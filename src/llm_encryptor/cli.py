"""Command line: chat (default), hide, reveal."""

from __future__ import annotations

import argparse
import getpass
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.live import Live
from rich.markup import escape
from rich.panel import Panel
from rich.text import Text

from .backend import Backend, BackendError, Message
from .codec import RevealError, reveal_with_keys
from .crypto import Keys, derive_keys
from .embed import EmbedConfig, EmbedError, Progress
from .hide import hide, validate_secret, words_needed
from .ollama import DEFAULT_HOST, OllamaBackend, list_models
from .session import Session
from .words import split_words

PASSWORD_ENV = "LLM_ENCRYPTOR_PASSWORD"
MODEL_ENV = "LLM_ENCRYPTOR_MODEL"
HOST_ENV = "OLLAMA_HOST"
_CLIPBOARD_COMMANDS = (
    ("pbcopy",),
    ("wl-copy",),
    ("xclip", "-selection", "clipboard"),
    ("xsel", "--clipboard", "--input"),
)
_PREVIEW_CHARS = 1200

HELP = """\
Type what the text should be about, then keep asking for changes.
Every version carries the same secret.

  /copy          copy the latest text to the clipboard
  /save FILE     write the latest text to a file
  /secret        hide a different secret from now on
  /new           forget the conversation and start a fresh text
  /help          show this help
  /quit          leave"""


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


def main(argv: Sequence[str] | None = None, io: Io | None = None) -> int:
    io = io or Io()
    args = _parser().parse_args(argv)
    try:
        command = {"hide": _hide, "reveal": _reveal}.get(args.command, _chat)
        command(args, io)
    except (UserError, BackendError, EmbedError, RevealError, ValueError) as error:
        io.errors.print(Text(str(error), style="red"))
        return 1
    except (KeyboardInterrupt, EOFError):
        io.errors.print()
        return 130
    return 0


def _parser() -> argparse.ArgumentParser:
    model_options = argparse.ArgumentParser(add_help=False)
    # SUPPRESS keeps a value given before the command from being reset by the command's parser.
    unset = argparse.SUPPRESS
    model_options.add_argument(
        "--model", "-m", default=unset, help=f"Ollama model to write with (or ${MODEL_ENV})"
    )
    model_options.add_argument("--host", default=unset, help=f"Ollama address (or ${HOST_ENV})")
    model_options.add_argument(
        "--temperature", type=float, default=unset, help="how freely the model writes (default 1.0)"
    )

    parser = argparse.ArgumentParser(
        prog="llm-encryptor",
        description="Hide an encrypted message inside ordinary-looking text written by a local LLM.",
        parents=[model_options],
    )
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("chat", parents=[model_options], help="write and revise a cover text (default)")
    hide_parser = commands.add_parser("hide", parents=[model_options], help="hide a secret in one go")
    hide_parser.add_argument(
        "--secret", "-s", help="the secret; leave out to be asked (this way it stays out of shell history)"
    )
    hide_parser.add_argument("--about", "-a", required=True, help="what the cover text should say")
    hide_parser.add_argument("--out", "-o", type=Path, help="write the text here instead of printing it")
    reveal_parser = commands.add_parser("reveal", help="recover the secret from a text")
    reveal_parser.add_argument("file", nargs="?", type=Path, help="file with the text (default: paste it)")
    return parser


# --- commands -------------------------------------------------------------


def _chat(args: argparse.Namespace, io: Io) -> None:
    backend = _backend(args, io)
    io.console.print(Panel(HELP, title="llm-encryptor", border_style="cyan"))
    secret = _ask_secret(io)
    keys = _keys(io, confirm=True)
    session = Session(secret, keys, backend, _config(args))
    _print_size(io, secret)
    while True:
        line = io.ask("\nyou › ").strip()
        if not line:
            continue
        if line in ("/quit", "/exit", "/q"):
            return
        try:
            session = _handle(line, session, io)
        except UserError as error:
            io.console.print(Text(str(error), style="yellow"))


def _handle(line: str, session: Session, io: Io) -> Session:
    command, _, argument = line.partition(" ")
    if command == "/help":
        io.console.print(HELP)
    elif command == "/copy":
        _copy(_latest(session), io)
    elif command == "/save":
        _save(_latest(session), argument.strip(), io)
    elif command == "/secret":
        session = session.with_secret(_ask_secret(io))
        _print_size(io, session.secret)
    elif command == "/new":
        session = session.restarted()
        io.console.print("[dim]Started over. The secret and password are unchanged.[/dim]")
    elif command.startswith("/"):
        io.console.print(Text(f"Unknown command {command}. Try /help.", style="yellow"))
    else:
        session = _write(session, line, io)
    return session


def _write(session: Session, instruction: str, io: Io) -> Session:
    try:
        with _progress(io) as on_progress:
            session = session.instruct(instruction, on_progress)
    except (EmbedError, BackendError) as error:
        io.console.print(Text(str(error), style="red"))
        return session
    except KeyboardInterrupt:
        io.console.print("[yellow]Stopped. The previous text, if any, is unchanged.[/yellow]")
        return session
    cover = session.cover or ""
    io.console.print(Panel(Text(cover), title="text to send", border_style="green"))
    io.console.print(
        f"[green]✓[/green] {len(split_words(cover))} words · checked: it decodes back to "
        "your secret · [dim]/copy to copy it, or ask for changes[/dim]"
    )
    return session


def _hide(args: argparse.Namespace, io: Io) -> None:
    backend = _backend(args, io)
    secret = validate_secret(args.secret) if args.secret else _ask_secret(io)
    keys = _keys(io, confirm=True)
    conversation: list[Message] = [{"role": "user", "content": args.about}]
    with _progress(io) as on_progress:
        cover = hide(secret, keys, backend, conversation, _config(args), on_progress)
    if args.out:
        _save(cover, str(args.out), io)
    else:
        print(cover)


def _reveal(args: argparse.Namespace, io: Io) -> None:
    if args.file:
        try:
            text = args.file.read_text(encoding="utf-8")
        except OSError as error:
            raise UserError(f"Could not read {args.file}: {error.strerror}") from error
    else:
        if io.stdin_is_terminal():
            io.errors.print("Paste the text, then press Ctrl-D on an empty line:")
        text = io.read_stdin()
    if not text.strip():
        raise UserError("There is no text to read.")
    print(reveal_with_keys(text, _keys(io, confirm=False)))


# --- helpers --------------------------------------------------------------


def _config(args: argparse.Namespace) -> EmbedConfig:
    temperature = getattr(args, "temperature", EmbedConfig().temperature)
    if not 0 < temperature <= 2:
        raise UserError("The temperature must be above 0 and at most 2.")
    return EmbedConfig(temperature=temperature)


def _backend(args: argparse.Namespace, io: Io) -> Backend:
    host = getattr(args, "host", None) or os.environ.get(HOST_ENV) or DEFAULT_HOST
    if "://" not in host:
        host = f"http://{host}"
    model = getattr(args, "model", None) or os.environ.get(MODEL_ENV)
    if not model:
        installed = io.list_models(host)
        if not installed:
            raise UserError("Ollama has no models yet. Get one with `ollama pull qwen3:4b`.")
        model = installed[0]
    io.errors.print(f"[dim]Writing with {escape(model)}[/dim]")
    return io.make_backend(model, host)


def _ask_secret(io: Io) -> str:
    while True:
        try:
            return validate_secret(io.ask("Secret to hide: ").strip())
        except ValueError as error:
            io.console.print(Text(str(error), style="yellow"))


def _keys(io: Io, confirm: bool) -> Keys:
    password = os.environ.get(PASSWORD_ENV)
    if not password:
        password = io.ask_hidden("Password (the reader needs the same one): ")
        if confirm and io.ask_hidden("Password again: ") != password:
            raise UserError("The two passwords are different.")
    return derive_keys(password)


def _print_size(io: Io, secret: str) -> None:
    io.console.print(
        f"[dim]This secret needs a text of at least {words_needed(secret)} words, "
        "so ask for something that long.[/dim]"
    )


def _latest(session: Session) -> str:
    if session.cover is None:
        raise UserError("There is no text yet. Say what it should be about first.")
    return session.cover


def _copy(text: str, io: Io) -> None:
    for command in _CLIPBOARD_COMMANDS:
        if not shutil.which(command[0]):
            continue
        try:
            subprocess.run(command, input=text.encode("utf-8"), check=True)
        except (subprocess.CalledProcessError, OSError):
            continue  # e.g. a Wayland tool on an X11 session: try the next one
        io.console.print("[green]Copied.[/green]")
        return
    raise UserError("No working clipboard tool found. Use /save FILE instead.")


def _save(text: str, target: str, io: Io) -> None:
    if not target:
        raise UserError("Say where to save it: /save FILE")
    try:
        Path(target).expanduser().write_text(text + "\n", encoding="utf-8")
    except OSError as error:
        raise UserError(f"Could not write {target}: {error.strerror}") from error
    io.errors.print(f"[green]Saved to {escape(target)}[/green]")


class _progress:
    """Shows the text growing while the model writes."""

    def __init__(self, io: Io) -> None:
        self._live = Live(Text("Starting…", style="dim"), console=io.errors, transient=True)

    def __enter__(self) -> Callable[[Progress], None]:
        self._live.__enter__()
        return self._update

    def __exit__(self, *_exc_info: object) -> None:
        self._live.__exit__(None, None, None)

    def _update(self, progress: Progress) -> None:
        shown = progress.text[-_PREVIEW_CHARS:]
        status = f"hidden {progress.embedded}/{progress.total} bits"
        self._live.update(Panel(Text(shown), title=status, border_style="dim"))


if __name__ == "__main__":
    sys.exit(main())
