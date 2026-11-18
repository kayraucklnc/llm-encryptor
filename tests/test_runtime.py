import io as io_module

import pytest
from rich.console import Console

from llm_encryptor import runtime, terminal
from llm_encryptor.backend import BackendError
from llm_encryptor.ollama import DEFAULT_HOST, is_running, list_models, pull_model


def make_io(answers=(), models=()):
    replies = iter(answers)
    asked = []

    def ask(prompt):
        asked.append(prompt)
        return next(replies)

    io = terminal.Io(
        console=Console(file=io_module.StringIO()),
        errors=Console(file=io_module.StringIO(), width=200),
        ask=ask,
        list_models=lambda host: list(models),
    )
    return io, asked


@pytest.fixture
def world(monkeypatch, tmp_path):
    """A machine where we decide what is installed and running."""
    state = {"running": False, "installed": True, "brew": True, "started": [], "ran": [], "pulled": []}
    monkeypatch.setattr(runtime, "is_running", lambda host: state["running"])
    monkeypatch.setattr(runtime, "_KNOWN_PLACES", ())
    monkeypatch.setattr(runtime.tempfile, "gettempdir", lambda: str(tmp_path))

    def which(name):
        if name == "ollama":
            return "/bin/ollama" if state["installed"] else None
        return "/bin/brew" if state["brew"] else None

    def popen(command, **kwargs):
        state["started"].append(command)
        state["running"] = True

    def run(command, check):
        state["ran"].append(command)
        state["installed"] = True

        class Done:
            returncode = 0

        return Done()

    def pull(model, host):
        state["pulled"].append(model)
        yield from ["pulling: 10%", "pulling: 10%", "success"]

    monkeypatch.setattr(runtime.shutil, "which", which)
    monkeypatch.setattr(runtime.subprocess, "Popen", popen)
    monkeypatch.setattr(runtime.subprocess, "run", run)
    monkeypatch.setattr(runtime, "pull_model", pull)
    return state


def test_nothing_to_do_when_the_server_is_running(world):
    world["running"] = True
    io, asked = make_io()
    runtime.ensure_ready(io, DEFAULT_HOST)
    assert not asked and not world["started"] and not world["pulled"]


def test_server_is_started_in_the_background(world):
    io, _ = make_io(models=["m"])
    runtime.ensure_ready(io, DEFAULT_HOST)
    assert world["started"] == [["/bin/ollama", "serve"]]
    assert "Starting Ollama" in io.errors.file.getvalue()


def test_missing_ollama_is_installed_after_asking(world):
    world["installed"] = False
    io, asked = make_io(answers=["y"], models=["m"])
    runtime.ensure_ready(io, DEFAULT_HOST)
    assert "Install it with Homebrew" in asked[0]
    assert world["ran"] == [["/bin/brew", "install", "ollama"]] and world["started"]


def test_declining_or_no_homebrew_explains_what_to_do(world):
    world["installed"] = False
    io, _ = make_io(answers=["n"])
    with pytest.raises(terminal.UserError, match="ollama.com/download"):
        runtime.ensure_ready(io, DEFAULT_HOST)
    world["brew"] = False
    with pytest.raises(terminal.UserError, match="not installed"):
        runtime.ensure_ready(make_io()[0], DEFAULT_HOST)
    assert not world["ran"]


def test_a_missing_model_is_downloaded_after_asking(world):
    io, asked = make_io(answers=[""], models=["other:1"])
    runtime.ensure_model(io, DEFAULT_HOST, "qwen2.5:0.5b")
    assert "qwen2.5:0.5b is not installed here" in asked[0] and world["pulled"] == ["qwen2.5:0.5b"]
    assert io.errors.file.getvalue().count("pulling: 10%") == 1
    with pytest.raises(terminal.UserError, match="ollama pull"):
        runtime.ensure_model(make_io(answers=["no"])[0], DEFAULT_HOST, "qwen2.5:0.5b")


def test_an_installed_model_is_left_alone(world):
    io, asked = make_io(models=["qwen3:4b", "llama3:latest"])
    runtime.ensure_model(io, DEFAULT_HOST, "qwen3:4b")
    runtime.ensure_model(io, DEFAULT_HOST, "llama3")           # Ollama stores it as llama3:latest
    assert not asked and not world["pulled"]


def test_failures_are_explained(world, monkeypatch):
    with pytest.raises(terminal.UserError, match="No Ollama server answers at http://box:1"):
        runtime.ensure_ready(make_io()[0], "http://box:1")

    def never(command, **kwargs):
        pass

    monkeypatch.setattr(runtime.subprocess, "Popen", never)
    with pytest.raises(terminal.UserError, match="did not start"):
        runtime._start(make_io()[0], "/bin/ollama", wait=lambda seconds: None)

    def broken(command, **kwargs):
        raise OSError(13, "Permission denied")

    monkeypatch.setattr(runtime.subprocess, "Popen", broken)
    with pytest.raises(terminal.UserError, match="Permission denied"):
        runtime._start(make_io()[0], "/bin/ollama")

    def failing_pull(model, host):
        raise BackendError("no space left")
        yield

    monkeypatch.setattr(runtime, "pull_model", failing_pull)
    with pytest.raises(terminal.UserError, match="no space left"):
        runtime.ensure_model(make_io(answers=["y"])[0], DEFAULT_HOST, "x:1")


# --- the Ollama calls it relies on ----------------------------------------


def test_models_are_listed_smallest_first():
    answer = {"models": [{"name": "big", "size": 9}, {"name": "small", "size": 1}, {"name": "mid", "size": 5}]}
    assert list_models(transport=lambda path, body: iter([answer])) == ["small", "mid", "big"]


def test_pull_reports_progress():
    updates = [{"status": "pulling manifest"}, {"status": "pulling abc", "total": 200, "completed": 50},
               {"status": "pulling abc", "total": 200, "completed": 200}, {"status": "success"}]
    lines = list(pull_model("m", transport=lambda path, body: iter(updates)))
    assert lines == ["pulling manifest", "pulling abc: 20%", "pulling abc: 100%", "success"]
    with pytest.raises(BackendError):
        list(pull_model("m", transport=lambda path, body: iter([{"error": "not found"}])))


def test_is_running(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", lambda url, timeout: io_module.BytesIO(b"{}"))
    assert is_running("http://host:1") is True

    def refuse(url, timeout):
        raise OSError("refused")

    monkeypatch.setattr("urllib.request.urlopen", refuse)
    assert is_running("http://host:1") is False
