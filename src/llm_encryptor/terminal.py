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
    # Gets the model server ready; set in cli so tests can leave it out.
    prepare: Callable[[Io, str], None] | None = None
    # Lets the user pick from a list with the arrow keys; None means typing a number.
    choose: Callable[[str, list[str], int], int | None] | None = None


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


_UP, _DOWN = (b"\x1b[A", b"\x1bOA", b"k"), (b"\x1b[B", b"\x1bOB", b"j")
_ENTER, _LEAVE = (b"\r", b"\n"), (b"q", b"\x1b")


def pick(
    title: str,
    items: list[str],
    start: int,
    read_key: Callable[[], bytes],
    write: Callable[[str], None],
    width: int = 80,
) -> int | None:
    """Arrow keys move, Enter chooses, a digit jumps straight there, q or Esc leaves."""
    at = min(max(start, 0), len(items) - 1)
    drawn = False
    while True:
        lines = [f"\x1b[1m{title}\x1b[0m  \x1b[2m↑↓ move · Enter choose · q back\x1b[0m"]
        for index, item in enumerate(items):
            label = f"{index + 1}  {item}"[: max(10, width - 4)]
            lines.append(f"\x1b[36;1m❯ {label}\x1b[0m" if index == at else f"  {label}")
        rewind = f"\x1b[{len(lines)}A" if drawn else ""
        write(rewind + "".join(f"\r\x1b[K{line}\n" for line in lines))
        drawn = True
        key = read_key()
        if key in _UP:
            at = (at - 1) % len(items)
        elif key in _DOWN:
            at = (at + 1) % len(items)
        elif key in _ENTER:
            return at
        elif key in _LEAVE:
            return None
        elif key.isdigit() and 1 <= int(key) <= len(items):
            return int(key) - 1
        elif key == b"\x03":
            raise KeyboardInterrupt
        elif key in (b"\x04", b""):
            raise EOFError


def arrow_menu(title: str, items: list[str], start: int = 0) -> int | None:
    """``pick`` on the real terminal, reading single key presses."""
    import termios
    import tty

    descriptor = sys.stdin.fileno()
    saved = termios.tcgetattr(descriptor)

    def write(text: str) -> None:
        sys.stdout.write(text)
        sys.stdout.flush()

    try:
        tty.setcbreak(descriptor)
        write("\x1b[?25l")
        return pick(
            title, items, start, lambda: os.read(descriptor, 8), write, shutil.get_terminal_size().columns
        )
    finally:
        write("\x1b[?25h")
        termios.tcsetattr(descriptor, termios.TCSADRAIN, saved)


def can_use_arrows() -> bool:
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return False
    try:
        import termios  # noqa: F401
    except ImportError:
        return False
    return True
