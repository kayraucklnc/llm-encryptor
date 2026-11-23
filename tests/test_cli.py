import io as io_module

import pytest
from fakes import FakeBackend
from rich.console import Console

from llm_encryptor import cli
from llm_encryptor.codec import RevealError, reveal

PASSWORD = "correct horse battery staple"


def make_io(lines=(), hidden=(), stdin="", models=("fake:1",)):
    out, err = io_module.StringIO(), io_module.StringIO()
    answers, secrets = iter(lines), iter(hidden)
    created = []

    def make_backend(model, host):
        created.append((model, host))
        backend = FakeBackend(seed=1)
        backend.model = model
        return backend

    def ask(prompt):
        try:
            return next(answers)
        except StopIteration:
            raise EOFError from None

    io = cli.Io(
        console=Console(file=out, width=200, force_terminal=False),
        errors=Console(file=err, width=200, force_terminal=False),
        ask=ask,
        ask_hidden=lambda prompt: next(secrets),
        read_stdin=lambda: stdin,
        stdin_is_terminal=lambda: False,
        make_backend=make_backend,
        list_models=lambda host: list(models),
    )
    return io, out, err, created


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in (cli.PASSWORD_ENV, cli.HOST_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(cli.MODEL_ENV, "fake:1")


def test_hide_then_reveal(capsys):
    io, _, _, created = make_io(hidden=[PASSWORD, PASSWORD])
    assert cli.main(["--robust", "hide", "-s", "top secret", "-a", "Write about the sea."], io) == 0
    cover = capsys.readouterr().out
    assert created == [("fake:1", "http://localhost:11434")]

    io, _, _, _ = make_io(hidden=[PASSWORD], stdin=cover)
    assert cli.main(["reveal"], io) == 0
    assert capsys.readouterr().out.strip() == "top secret"


def test_hide_to_file_and_reveal_from_file(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv(cli.PASSWORD_ENV, PASSWORD)
    target = tmp_path / "cover.txt"
    io, _, _, _ = make_io(lines=["from the prompt"])
    assert cli.main(["--robust", "hide", "-a", "Write.", "-o", str(target)], io) == 0
    assert capsys.readouterr().out == ""
    assert reveal(target.read_text(), PASSWORD) == "from the prompt"

    io, _, _, _ = make_io()
    assert cli.main(["reveal", str(target)], io) == 0
    assert capsys.readouterr().out.strip() == "from the prompt"


def test_reveal_with_wrong_password_fails(capsys):
    io, _, _, _ = make_io(hidden=[PASSWORD, PASSWORD])
    cli.main(["--robust", "hide", "-s", "top secret", "-a", "Write."], io)
    cover = capsys.readouterr().out
    io, _, err, _ = make_io(hidden=["wrong"], stdin=cover)
    assert cli.main(["reveal"], io) == 1
    assert "No hidden message found" in err.getvalue()


def test_reveal_needs_text(tmp_path):
    io, _, err, _ = make_io(stdin="  \n")
    assert cli.main(["reveal"], io) == 1
    assert "no text" in err.getvalue()
    io, _, err, _ = make_io()
    assert cli.main(["reveal", str(tmp_path / "missing.txt")], io) == 1
    assert "Could not read" in err.getvalue()


def test_mismatched_passwords_are_refused():
    io, _, err, _ = make_io(hidden=["one", "two"])
    assert cli.main(["--robust", "hide", "-s", "x", "-a", "Write."], io) == 1
    assert "different" in err.getvalue()


def test_model_and_host_can_be_chosen(monkeypatch):
    monkeypatch.setenv(cli.PASSWORD_ENV, PASSWORD)
    io, _, _, created = make_io()
    cli.main(["--robust", "--model", "big:7", "--host", "box:9", "hide", "-s", "x", "-a", "Write."], io)
    io2, _, _, created2 = make_io()
    monkeypatch.setenv(cli.MODEL_ENV, "env:1")
    cli.main(["--robust", "hide", "-s", "x", "-a", "Write.", "--temperature", "0.5"], io2)
    assert created == [("big:7", "http://box:9")]
    assert created2 == [("env:1", "http://localhost:11434")]


def test_everyone_starts_with_the_same_model(monkeypatch):
    """Two people who never chose a model must be able to read each other's texts."""
    monkeypatch.delenv(cli.MODEL_ENV)
    monkeypatch.setenv(cli.PASSWORD_ENV, PASSWORD)
    io, _, _, created = make_io(models=("huge:70b", "other:1"))
    assert cli.main(["--robust", "hide", "-s", "x", "-a", "Write."], io) == 0
    assert created == [("qwen2.5:0.5b", "http://localhost:11434")]


def test_a_missing_model_is_fetched_before_use(monkeypatch):
    monkeypatch.setenv(cli.PASSWORD_ENV, PASSWORD)
    fetched = []
    io, _, _, _ = make_io()
    io.fetch_model = lambda io, host, model: fetched.append((host, model))
    assert cli.main(["--robust", "--model", "new:1", "hide", "-s", "x", "-a", "Write."], io) == 0
    assert fetched == [("http://localhost:11434", "new:1")]


def test_bad_temperature_is_refused(monkeypatch):
    monkeypatch.setenv(cli.PASSWORD_ENV, PASSWORD)
    io, _, err, _ = make_io()
    assert cli.main(["--robust", "hide", "-s", "x", "-a", "Write.", "--temperature", "0"], io) == 1
    assert "temperature" in err.getvalue()


# --- compact format (the default) ----------------------------------------


def test_compact_hide_then_reveal_with_the_same_model(capsys):
    io, _, err, _ = make_io(hidden=[PASSWORD, PASSWORD])
    assert cli.main(["hide", "-s", "top secret", "-a", "Write about the sea."], io) == 0
    cover = capsys.readouterr().out
    assert "the model fake:1" in err.getvalue()
    with pytest.raises(RevealError):
        reveal(cover, PASSWORD)  # not readable without the model

    io, _, err, _ = make_io(hidden=[PASSWORD], stdin=cover)
    assert cli.main(["reveal", "--model", "fake:1"], io) == 0
    assert capsys.readouterr().out.strip() == "top secret"
    assert "Reading with fake:1" in err.getvalue()


def test_compact_reveal_with_wrong_password_fails(capsys):
    io, _, _, _ = make_io(hidden=[PASSWORD, PASSWORD])
    cli.main(["hide", "-s", "top secret", "-a", "Write."], io)
    cover = capsys.readouterr().out
    io, _, err, _ = make_io(hidden=["wrong"], stdin=cover)
    assert cli.main(["reveal"], io) == 1
    assert "No hidden message found" in err.getvalue()


def test_reveal_explains_when_no_model_can_be_reached(capsys):
    from llm_encryptor.backend import BackendError

    class Unreachable(FakeBackend):
        def reset(self):
            raise BackendError("Could not reach Ollama")

    io, _, err, _ = make_io(hidden=[PASSWORD], stdin="Just some ordinary words. " * 30)
    io.make_backend = lambda model, host: Unreachable()
    assert cli.main(["reveal"], io) == 1
    assert "compact format could not be tried" in err.getvalue()


def test_models_are_listed():
    io, out, _, _ = make_io(models=("a:1", "b:2"))
    assert cli.main(["models"], io) == 0
    assert out.getvalue().split() == ["a:1", "b:2"]
    io, _, err, _ = make_io(models=())
    assert cli.main(["models"], io) == 1
    assert "ollama pull" in err.getvalue()


def test_names_hide_then_reveal_without_a_model(capsys):
    io, _, err, created = make_io(hidden=[PASSWORD, PASSWORD])
    assert cli.main(["--names", "hide", "-s", "meet at noon", "-a", "dinner on Friday"], io) == 0
    cover = capsys.readouterr().out.strip()
    assert len(cover.split()) <= 12 and "No model needed" in err.getvalue()
    reader, _, _, opened = make_io(hidden=[PASSWORD], stdin=cover)
    assert cli.main(["reveal"], reader) == 0
    assert capsys.readouterr().out.strip() == "meet at noon"
    assert len(created) == 1 and opened == []


def test_names_and_robust_cannot_be_combined():
    io, _, err, _ = make_io(hidden=[PASSWORD, PASSWORD])
    assert cli.main(["--names", "--robust", "hide", "-s", "x", "-a", "Write."], io) == 1
    assert "one of --names and --robust" in err.getvalue()
