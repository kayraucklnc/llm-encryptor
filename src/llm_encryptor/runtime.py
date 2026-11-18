"""Get Ollama ready without the user having to: install, start, fetch a model."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

from rich.text import Text

from .backend import BackendError
from .ollama import DEFAULT_HOST, is_running, pull_model
from .terminal import Io, UserError

# Small enough to download in a minute and to write a text in seconds.
DEFAULT_MODEL = "qwen2.5:0.5b"
DEFAULT_MODEL_SIZE = "about 400 MB"
_START_SECONDS = 30
_KNOWN_PLACES = (
    "/opt/homebrew/bin/ollama",
    "/usr/local/bin/ollama",
    "/Applications/Ollama.app/Contents/Resources/ollama",
)
_INSTALL_HELP = "Install Ollama from https://ollama.com/download and run llm-encryptor again."


def ensure_ready(io: Io, host: str) -> None:
    """Make sure an Ollama server answers at ``host`` and has a model to use."""
    if not is_running(host):
        if host != DEFAULT_HOST:
            raise UserError(f"No Ollama server answers at {host}.")
        _start(io, _find_ollama() or _install(io))
    if not io.list_models(host):
        _fetch_default_model(io, host)


def _find_ollama() -> str | None:
    found = shutil.which("ollama")
    return found or next((place for place in _KNOWN_PLACES if Path(place).exists()), None)


def _install(io: Io) -> str:
    brew = shutil.which("brew")
    if brew is None:
        raise UserError(f"Ollama is not installed. {_INSTALL_HELP}")
    if not _yes(io, "Ollama (the program that runs the models) is not installed. Install it with Homebrew now?"):
        raise UserError(_INSTALL_HELP)
    if subprocess.run([brew, "install", "ollama"], check=False).returncode != 0:
        raise UserError(f"Homebrew could not install Ollama. {_INSTALL_HELP}")
    installed = _find_ollama()
    if installed is None:
        raise UserError(f"Ollama was installed but cannot be found. {_INSTALL_HELP}")
    return installed


def _start(io: Io, ollama: str, wait: Callable[[float], None] = time.sleep) -> None:
    io.errors.print(Text("Starting Ollama in the background…", style="dim"))
    log = Path(tempfile.gettempdir()) / "llm-encryptor-ollama.log"
    try:
        with log.open("ab") as output:
            # Its own session: it keeps running after this app closes, so the next start is instant.
            subprocess.Popen(
                [ollama, "serve"], stdout=output, stderr=output, stdin=subprocess.DEVNULL,
                start_new_session=True,
            )  # fmt: skip
    except OSError as error:
        raise UserError(f"Could not start Ollama: {error.strerror}") from error
    for _ in range(_START_SECONDS * 4):
        if is_running(DEFAULT_HOST):
            return
        wait(0.25)
    raise UserError(f"Ollama did not start. Its messages are in {log}.")


def _fetch_default_model(io: Io, host: str) -> None:
    question = f"No model is installed yet. Download the small one ({DEFAULT_MODEL}, {DEFAULT_MODEL_SIZE})?"
    if not _yes(io, question):
        raise UserError(f"A model is needed. Get one with `ollama pull {DEFAULT_MODEL}`.")
    last = ""
    try:
        for status in pull_model(DEFAULT_MODEL, host):
            if status != last:
                io.errors.print(Text(status, style="dim"))
                last = status
    except BackendError as error:
        raise UserError(f"The download failed: {error}") from error


def _yes(io: Io, question: str) -> bool:
    return io.ask(f"{question} [Y/n] ").strip().lower() in ("", "y", "yes")
