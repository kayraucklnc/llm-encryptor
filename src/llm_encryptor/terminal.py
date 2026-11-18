"""Shared terminal plumbing: input and output, clipboard, progress display."""

from __future__ import annotations

import getpass
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.live import Live
from rich.markup import escape
from rich.panel import Panel
from rich.text import Text

from .backend import Backend
from .crypto import Keys, derive_keys
from .embed import Progress
from .hide import validate_secret
from .ollama import OllamaBackend, list_models

PASSWORD_ENV = "LLM_ENCRYPTOR_PASSWORD"
_COPY_COMMANDS = (
    ("pbcopy",),
    ("wl-copy",),
    ("xclip", "-selection", "clipboard"),
    ("xsel", "--clipboard", "--input"),
)
_PASTE_COMMANDS = (
    ("pbpaste",),
    ("wl-paste", "--no-newline"),
    ("xclip", "-selection", "clipboard", "-o"),
    ("xsel", "--clipboard", "--output"),
)
_PREVIEW_CHARS = 1200


class UserError(Exception):
    """A problem the user can fix; shown without a traceback."""


@dataclass
class Io:
    """Everything that touches the terminal, so tests can replace it."""

    console: Console = field(default_factory=Console)
    errors: Console = field(default_factory=lambda: Console(stderr=True))
    ask: Callable[[str], str] = input
    ask_hidden: Callable[[str], str] = getpass.getpass
    read_stdin: Callable[[], str] = sys.stdin.read
    stdin_is_terminal: Callable[[], bool] = sys.stdin.isatty
    make_backend: Callable[[str, str], Backend] = OllamaBackend
    list_models: Callable[[str], list[str]] = list_models


def ask_secret(io: Io) -> str:
    while True:
        try:
            return validate_secret(io.ask("Secret to hide: ").strip())
        except ValueError as error:
            io.console.print(Text(str(error), style="yellow"))


def ask_keys(io: Io, confirm: bool) -> Keys:
    password = os.environ.get(PASSWORD_ENV)
    if not password:
        password = io.ask_hidden("Password (the reader needs the same one): ")
        if confirm and io.ask_hidden("Password again: ") != password:
            raise UserError("The two passwords are different.")
    return derive_keys(password)


def copy_text(text: str) -> bool:
    """Put text on the clipboard. False when no clipboard tool works here."""
    return _run_first(_COPY_COMMANDS, text.encode("utf-8")) is not None


def read_clipboard() -> str | None:
    """The clipboard's text, or None when no clipboard tool works here."""
    output = _run_first(_PASTE_COMMANDS, None)
    return None if output is None else output.decode("utf-8", "replace")


def _run_first(commands: tuple[tuple[str, ...], ...], data: bytes | None) -> bytes | None:
    for command in commands:
        if not shutil.which(command[0]):
            continue
        try:
            done = subprocess.run(command, input=data, check=True, capture_output=data is None)
        except (subprocess.CalledProcessError, OSError):
            continue  # e.g. a Wayland tool on an X11 session: try the next one
        return done.stdout or b""
    return None


def save_text(text: str, target: str, io: Io) -> None:
    try:
        Path(target).expanduser().write_text(text + "\n", encoding="utf-8")
    except OSError as error:
        raise UserError(f"Could not write {target}: {error.strerror}") from error
    io.errors.print(f"[green]Saved to {escape(target)}[/green]")


def reader_needs(backend: Backend, compact: bool) -> str:
    if not compact:
        return "To read it: the password. No model needed."
    model = getattr(backend, "model", "the same model")
    return f"To read it: the password and the model {model}."


class progress:
    """Shows the text growing while the model writes."""

    def __init__(self, io: Io) -> None:
        self._live = Live(Text("Starting…", style="dim"), console=io.errors, transient=True)

    def __enter__(self) -> Callable[[Progress], None]:
        self._live.__enter__()
        return self._update

    def __exit__(self, *_exc_info: object) -> None:
        self._live.__exit__(None, None, None)

    def _update(self, update: Progress) -> None:
        shown = update.text[-_PREVIEW_CHARS:]
        status = f"hidden {update.embedded}/{update.total} bits"
        self._live.update(Panel(Text(shown), title=status, border_style="dim"))
