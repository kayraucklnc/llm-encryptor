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
_DENSITIES = {
    False: "natural (reads best)",
    True: "tight (fewer words for longer secrets; reads less smoothly)",
}
_FORMATS = {
    True: "compact (short text; the reader needs the same model)",
    False: "robust (about three times longer; the reader needs no model)",
}


@dataclass(frozen=True)
class Settings:
    host: str
    model: str
    compact: bool = True
    tight: bool = False
    keys: Keys | None = None


def run_app(io: Io, settings: Settings) -> None:
    at = 0
    while True:
        items = [
            "Encrypt a message",
            "Decrypt a message",
            f"Model: {settings.model}",
            f"Format: {_FORMATS[settings.compact]}",
            f"Density: {_DENSITIES[settings.tight]}",
            "Change password",
            "Quit",
        ]
        choice = choose(io, "llm-encryptor", items, at)
        if choice is None or choice == len(items) - 1:
            return
        at = choice
        try:
            settings = _ACTIONS[choice](io, settings)
        except (UserError, BackendError, EmbedError, RevealError) as error:
            io.console.print(Text(str(error), style="red"))
        except KeyboardInterrupt:
            io.console.print(Text("Stopped.", style="yellow"))


def choose(io: Io, title: str, items: list[str], start: int = 0) -> int | None:
    """Let the user pick an item: with the arrow keys, or by typing its number."""
    if io.choose is not None:
        return io.choose(title, items, start)
    numbered = "\n".join(f"{number}  {item}" for number, item in enumerate(items, 1))
    io.console.print(Panel(Text(numbered), title=title, border_style="cyan"))
    while True:
        answer = io.ask("> ").strip().lower()
        if answer in ("q", "quit", "exit"):
            return None
        if answer.isdigit() and 1 <= int(answer) <= len(items):
            return int(answer) - 1
        io.console.print(Text(f"Type a number from 1 to {len(items)}, or q.", style="yellow"))


def _encrypt(io: Io, settings: Settings) -> Settings:
    secret = ask_secret(io)
    settings = _with_keys(io, settings, confirm=True)
    assert settings.keys is not None
    session = Session(
        secret,
        settings.keys,
        io.make_backend(settings.model, settings.host),
        compact=settings.compact,
        tight=settings.tight,
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
        raise UserError("Ollama has no models yet. Get one with `ollama pull qwen2.5:0.5b`.")
    current = installed.index(settings.model) if settings.model in installed else 0
    choice = choose(io, "Model (smallest first)", installed, current)
    return settings if choice is None else replace(settings, model=installed[choice])


def _switch_format(io: Io, settings: Settings) -> Settings:
    return replace(settings, compact=not settings.compact)


def _switch_density(io: Io, settings: Settings) -> Settings:
    if not settings.compact:
        raise UserError("Density only applies to the compact format (item 4).")
    return replace(settings, tight=not settings.tight)


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


_ACTIONS = (_encrypt, _decrypt, _choose_model, _switch_format, _switch_density, _change_password)
