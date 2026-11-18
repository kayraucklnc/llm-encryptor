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
    for name in (cli.PASSWORD_ENV, cli.MODEL_ENV, cli.HOST_ENV):
        monkeypatch.delenv(name, raising=False)


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


def test_no_models_is_explained():
    io, _, err, _ = make_io(models=())
    assert cli.main(["--robust", "hide", "-s", "x", "-a", "Write."], io) == 1
    assert "ollama pull" in err.getvalue()


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

    def unreachable(host):
        raise BackendError("Could not reach Ollama")

    io, _, err, _ = make_io(hidden=[PASSWORD], stdin="Just some ordinary words. " * 30)
    io.list_models = unreachable
    assert cli.main(["reveal"], io) == 1
    assert "compact format could not be tried" in err.getvalue()


def test_models_are_listed():
    io, out, _, _ = make_io(models=("a:1", "b:2"))
    assert cli.main(["models"], io) == 0
    assert out.getvalue().split() == ["a:1", "b:2"]
    io, _, err, _ = make_io(models=())
    assert cli.main(["models"], io) == 1
    assert "ollama pull" in err.getvalue()
