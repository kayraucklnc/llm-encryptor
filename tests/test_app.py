import io as io_module

import pytest
from fakes import FakeBackend
from rich.console import Console

from llm_encryptor import app, cli, terminal
from llm_encryptor.backend import BackendError
from llm_encryptor.codec import reveal

PASSWORD = "correct horse battery staple"


class Clipboard:
    def __init__(self, text=None, works=True):
        self.text, self.works = text, works

    def copy(self, text):
        if self.works:
            self.text = text
        return self.works

    def read(self):
        return self.text if self.works else None


@pytest.fixture
def clipboard(monkeypatch):
    board = Clipboard()
    monkeypatch.setattr(app, "copy_text", board.copy)
    monkeypatch.setattr(app, "read_clipboard", board.read)
    for name in (cli.PASSWORD_ENV, cli.MODEL_ENV, cli.HOST_ENV):
        monkeypatch.delenv(name, raising=False)
    return board


def run(lines, hidden=(), stdin="", models=("fake:1", "small:2"), argv=()):
    out = io_module.StringIO()
    answers, secrets = iter(lines), iter(hidden)
    created = []

    def make_backend(model, host):
        created.append(model)
        backend = FakeBackend(seed=1)
        backend.model = model
        return backend

    def ask(prompt):
        out.write(prompt)
        try:
            return next(answers)
        except StopIteration:
            raise EOFError from None

    io = terminal.Io(
        console=Console(file=out, width=200, force_terminal=False),
        errors=Console(file=io_module.StringIO(), width=200, force_terminal=False),
        ask=ask,
        ask_hidden=lambda prompt: next(secrets),
        read_stdin=lambda: stdin,
        stdin_is_terminal=lambda: True,
        make_backend=make_backend,
        list_models=lambda host: list(models),
    )
    return cli.main(list(argv), io), out.getvalue(), created


def test_menu_is_shown_and_quits(clipboard):
    code, shown, _ = run(["x", "q"])
    assert code == 0
    assert "1  Encrypt a message" in shown and "2  Decrypt a message" in shown
    assert "Model: fake:1" in shown and "Format: compact" in shown
    assert "Type 1, 2, 3, 4, 5 or q." in shown


def test_encrypt_copies_and_decrypt_reads_the_clipboard(clipboard):
    lines = ["1", "meet at noon", "write about Turkey", "", "2", "", "q"]
    code, shown, _ = run(lines, hidden=[PASSWORD, PASSWORD])
    assert code == 0
    assert "copied to your clipboard" in shown
    assert "the model fake:1" in shown
    assert "hidden message" in shown and "meet at noon" in shown
    assert clipboard.text and "meet at noon" not in clipboard.text


def test_a_change_gives_a_new_text_with_the_same_secret(clipboard):
    seen = []
    original = clipboard.copy
    clipboard_copy = lambda text: (seen.append(text), original(text))[1]
    app.copy_text = clipboard_copy
    lines = ["1", "s", "", "make it friendlier", "", "2", "", "q"]
    code, shown, _ = run(lines, hidden=[PASSWORD, PASSWORD])
    assert code == 0
    assert len(seen) == 2 and seen[0] != seen[1]
    assert shown.count("text to send") == 2 and "hidden message" in shown


def test_robust_format_can_be_read_without_a_model(clipboard):
    lines = ["4", "1", "s", "about the sea", "", "q"]
    code, shown, _ = run(lines, hidden=[PASSWORD, PASSWORD])
    assert code == 0
    assert "Format: robust" in shown and "No model needed" in shown
    assert reveal(clipboard.text, PASSWORD) == "s"


def test_model_can_be_chosen_by_number_or_name(clipboard):
    code, shown, created = run(["3", "2", "1", "s", "", "", "3", "fake:1", "3", "9", "q"], hidden=[PASSWORD, PASSWORD])
    assert code == 0
    assert "Model: small:2" in shown and created == ["small:2"]
    assert shown.count("Model: fake:1") >= 2
    assert "not one of the models" in shown


def test_no_models_is_explained(clipboard):
    code, _, _ = run(["q"], models=())
    assert code == 1
    code, shown, _ = run(["q"], argv=["--model", "given:1"], models=())
    assert code == 0 and "Model: given:1" in shown


def test_decrypt_from_a_file_and_from_pasted_text(clipboard, tmp_path):
    run(["1", "from a file", "", "", "q"], hidden=[PASSWORD, PASSWORD])
    saved = tmp_path / "msg.txt"
    saved.write_text(clipboard.text)
    code, shown, _ = run(["2", str(saved), "2", str(tmp_path / "missing"), "q"], hidden=[PASSWORD])
    assert code == 0 and "from a file" in shown and "Could not read" in shown

    text, clipboard.works = clipboard.text, False
    code, shown, _ = run(["2", "", "2", "", "q"], hidden=[PASSWORD], stdin=text)
    assert "No clipboard here" in shown and "from a file" in shown


def test_wrong_password_can_be_retried(clipboard):
    run(["1", "s3cret", "", "", "q"], hidden=[PASSWORD, PASSWORD])
    code, shown, _ = run(["2", "", "2", "", "q"], hidden=["wrong", PASSWORD])
    assert code == 0
    assert "No hidden message found" in shown and "s3cret" in shown


def test_empty_clipboard_and_mismatched_passwords_are_reported(clipboard):
    clipboard.text = "  "
    code, shown, _ = run(["2", "", "5", "1", "s", "q"], hidden=["one", "two", "a", "b"])
    assert code == 0
    assert "There is no text to read" in shown
    assert shown.count("The two passwords are different") == 2


def test_password_is_asked_once_and_can_be_changed(clipboard):
    lines = ["1", "first", "", "", "1", "second", "", "", "5", "2", "", "q"]
    code, shown, _ = run(lines, hidden=[PASSWORD, PASSWORD, "new one", "new one"])
    assert code == 0
    assert "No hidden message found" in shown  # the text was written with the old password


def test_without_a_clipboard_the_text_is_still_shown(clipboard):
    clipboard.works = False
    code, shown, _ = run(["1", "s", "", "", "q"], hidden=[PASSWORD, PASSWORD])
    assert code == 0 and "select it above to copy it" in shown


def test_model_trouble_and_interrupts_return_to_the_menu(clipboard, monkeypatch):
    def broken(self, request, on_progress):
        raise BackendError("Could not reach Ollama")

    monkeypatch.setattr(app.Session, "instruct", broken)
    code, shown, _ = run(["1", "s", "", "q"], hidden=[PASSWORD, PASSWORD])
    assert code == 0 and "Could not reach Ollama" in shown

    def interrupted(self, request, on_progress):
        raise KeyboardInterrupt

    monkeypatch.setattr(app.Session, "instruct", interrupted)
    code, shown, _ = run(["1", "s", "", "q"], hidden=[PASSWORD, PASSWORD])
    assert code == 0 and "Stopped." in shown


def test_end_of_input_leaves_quietly(clipboard):
    assert run(["1"])[0] == 130


# --- clipboard tools ------------------------------------------------------


def test_copy_and_paste_use_the_first_working_tool(monkeypatch):
    used = []

    class Done:
        stdout = b"pasted"

    def fake_run(command, input, check, capture_output):
        used.append(command[0])
        if command[0] in ("wl-copy", "wl-paste"):
            raise terminal.subprocess.CalledProcessError(1, command)
        return Done()

    monkeypatch.setattr(terminal.shutil, "which", lambda name: name if not name.startswith("pb") else None)
    monkeypatch.setattr(terminal.subprocess, "run", fake_run)
    assert terminal.copy_text("hello") is True
    assert terminal.read_clipboard() == "pasted"
    assert used == ["wl-copy", "xclip", "wl-paste", "xclip"]

    monkeypatch.setattr(terminal.shutil, "which", lambda name: None)
    assert terminal.copy_text("hello") is False
    assert terminal.read_clipboard() is None
