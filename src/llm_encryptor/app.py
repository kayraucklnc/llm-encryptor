"""The menu you get when you run llm-encryptor by itself."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from rich.panel import Panel
from rich.text import Text

from .backend import BackendError
from .codec import RevealError, reveal_with_keys
from .crypto import Keys
from .dense import read_dense
from .embed import EmbedError
from .session import Session
from .terminal import (
    Io,
    UserError,
    ask_keys,
    ask_secret,
    copy_text,
    progress,
    read_clipboard,
    reader_needs,
)
from .words import split_words

DEFAULT_TOPIC = "a friendly note to a friend about how the week has been"
_FORMATS = {
    True: "compact (short text; the reader needs the same model)",
    False: "robust (about three times longer; the reader needs no model)",
}


@dataclass(frozen=True)
class Settings:
    host: str
    model: str
    compact: bool = True
    keys: Keys | None = None


def run_app(io: Io, settings: Settings) -> None:
    while True:
        io.console.print(
            Panel(
                Text(
                    "1  Encrypt a message\n"
                    "2  Decrypt a message\n"
                    f"3  Model: {settings.model}\n"
                    f"4  Format: {_FORMATS[settings.compact]}\n"
                    "5  Change password\n"
                    "q  Quit"
                ),
                title="llm-encryptor",
                border_style="cyan",
            )
        )
        choice = io.ask("> ").strip().lower()
        if choice in ("q", "quit", "exit"):
            return
        action = _ACTIONS.get(choice)
        if action is None:
            io.console.print(Text("Type 1, 2, 3, 4, 5 or q.", style="yellow"))
            continue
        try:
            settings = action(io, settings)
        except (UserError, BackendError, EmbedError, RevealError) as error:
            io.console.print(Text(str(error), style="red"))
        except KeyboardInterrupt:
            io.console.print(Text("Stopped.", style="yellow"))


def _encrypt(io: Io, settings: Settings) -> Settings:
    secret = ask_secret(io)
    settings = _with_keys(io, settings, confirm=True)
    assert settings.keys is not None
    session = Session(
        secret, settings.keys, io.make_backend(settings.model, settings.host), compact=settings.compact
    )
    request = io.ask("What should the text be about? (Enter for a friendly note) ").strip()
    request = request or DEFAULT_TOPIC
    while request:
        with progress(io) as on_progress:
            session = session.instruct(request, on_progress)
        cover = session.cover or ""
        io.console.print(Panel(Text(cover), title="text to send", border_style="green"))
        copied = "copied to your clipboard" if copy_text(cover) else "select it above to copy it"
        io.console.print(f"[green]✓[/green] {len(split_words(cover))} words · {copied}")
        io.console.print(Text(reader_needs(session.backend, session.compact), style="dim"))
        request = io.ask("Ask for a change (e.g. make it friendlier), or Enter when done: ").strip()
    return settings


def _decrypt(io: Io, settings: Settings) -> Settings:
    source = io.ask("Copy the text, then press Enter (or type a file name): ").strip()
    text = _read_file(source) if source else read_clipboard()
    if text is None:
        io.console.print("No clipboard here. Paste the text, then press Ctrl-D on an empty line:")
        text = io.read_stdin()
    if not text.strip():
        raise UserError("There is no text to read.")
    settings = _with_keys(io, settings, confirm=False)
    assert settings.keys is not None
    try:
        secret = reveal_with_keys(text, settings.keys)  # The robust format needs no model.
    except RevealError:
        io.console.print(Text(f"Reading with {settings.model}…", style="dim"))
        try:
            secret = read_dense(text, settings.keys, io.make_backend(settings.model, settings.host))
        except RevealError as error:
            # A wrong password should not stick for the next attempt.
            io.console.print(Text(str(error), style="red"))
            return replace(settings, keys=None)
    io.console.print(Panel(Text(secret), title="hidden message", border_style="green"))
    return settings


def _choose_model(io: Io, settings: Settings) -> Settings:
    installed = io.list_models(settings.host)
    if not installed:
        raise UserError("Ollama has no models yet. Get one with `ollama pull qwen3:4b`.")
    for number, name in enumerate(installed, 1):
        io.console.print(Text(f"{number}  {name}"))
    answer = io.ask("Which one? ").strip()
    if answer.isdigit() and 1 <= int(answer) <= len(installed):
        return replace(settings, model=installed[int(answer) - 1])
    if answer in installed:
        return replace(settings, model=answer)
    raise UserError("That is not one of the models listed.")


def _switch_format(io: Io, settings: Settings) -> Settings:
    return replace(settings, compact=not settings.compact)


def _change_password(io: Io, settings: Settings) -> Settings:
    return _with_keys(io, replace(settings, keys=None), confirm=True)


def _with_keys(io: Io, settings: Settings, confirm: bool) -> Settings:
    """Ask for the password once; it is kept until changed or the app closes."""
    if settings.keys is not None:
        return settings
    return replace(settings, keys=ask_keys(io, confirm))


def _read_file(name: str) -> str:
    try:
        return Path(name).expanduser().read_text(encoding="utf-8")
    except OSError as error:
        raise UserError(f"Could not read {name}: {error.strerror}") from error


_ACTIONS = {
    "1": _encrypt,
    "2": _decrypt,
    "3": _choose_model,
    "4": _switch_format,
    "5": _change_password,
}
