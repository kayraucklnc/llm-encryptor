"""Command line. With no command it opens the menu; hide and reveal are for scripts."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from rich.markup import escape
from rich.text import Text

from .app import Settings, run_app
from .backend import Backend, BackendError, Message
from .codec import RevealError, reveal_with_keys
from .dense import DENSITIES, read_dense, write_dense
from .embed import EmbedConfig, EmbedError
from .hide import hide, validate_secret
from .ollama import DEFAULT_HOST
from .runtime import ensure_ready
from .terminal import (
    PASSWORD_ENV,
    Io,
    UserError,
    arrow_menu,
    ask_keys,
    ask_secret,
    can_use_arrows,
    progress,
    reader_needs,
    save_text,
)

MODEL_ENV = "LLM_ENCRYPTOR_MODEL"
HOST_ENV = "OLLAMA_HOST"
__all__ = ["HOST_ENV", "MODEL_ENV", "PASSWORD_ENV", "Io", "main"]


def main(argv: Sequence[str] | None = None, io: Io | None = None) -> int:
    io = io or Io(prepare=ensure_ready, choose=arrow_menu if can_use_arrows() else None)
    args = _parser().parse_args(argv)
    try:
        command = {"hide": _hide, "reveal": _reveal, "models": _models}.get(args.command, _menu)
        command(args, io)
    except (UserError, BackendError, EmbedError, RevealError, ValueError) as error:
        io.errors.print(Text(str(error), style="red"))
        return 1
    except (KeyboardInterrupt, EOFError):
        io.errors.print()
        return 130
    return 0


def _parser() -> argparse.ArgumentParser:
    shared = argparse.ArgumentParser(add_help=False)
    # SUPPRESS keeps a value given before the command from being reset by the command's parser.
    unset = argparse.SUPPRESS
    shared.add_argument("--model", "-m", default=unset, help=f"Ollama model to use (or ${MODEL_ENV})")
    shared.add_argument("--host", default=unset, help=f"Ollama address (or ${HOST_ENV})")
    shared.add_argument(
        "--robust",
        action="store_true",
        default=unset,
        help="write a longer text that can be read without a model",
    )
    shared.add_argument(
        "--density",
        choices=list(DENSITIES),
        default=unset,
        help="compact format: tight is shorter and reads less smoothly; max is shortest and nonsense",
    )
    shared.add_argument(
        "--temperature", type=float, default=unset, help="robust format: how freely the model writes"
    )

    parser = argparse.ArgumentParser(
        prog="llm-encryptor",
        description="Hide an encrypted message inside ordinary-looking text written by a local LLM. "
        "Run it without a command for the menu.",
        parents=[shared],
    )
    commands = parser.add_subparsers(dest="command")
    hide_parser = commands.add_parser("hide", parents=[shared], help="hide a secret in one go")
    hide_parser.add_argument(
        "--secret", "-s", help="the secret; leave out to be asked (this way it stays out of shell history)"
    )
    hide_parser.add_argument("--about", "-a", required=True, help="what the cover text should say")
    hide_parser.add_argument("--out", "-o", type=Path, help="write the text here instead of printing it")
    reveal_parser = commands.add_parser("reveal", parents=[shared], help="recover the secret from a text")
    reveal_parser.add_argument("file", nargs="?", type=Path, help="file with the text (default: paste it)")
    commands.add_parser("models", parents=[shared], help="list the models you can use")
    return parser


def _menu(args: argparse.Namespace, io: Io) -> None:
    _prepare(args, io)
    settings = Settings(_host(args), _model(args, io), _compact(args), _density(args))
    run_app(io, settings)


def _hide(args: argparse.Namespace, io: Io) -> None:
    backend = _backend(args, io, "Writing")
    secret = validate_secret(args.secret) if args.secret else ask_secret(io)
    keys = ask_keys(io, confirm=True)
    conversation: list[Message] = [{"role": "user", "content": args.about}]
    with progress(io) as on_progress:
        if _compact(args):
            cover = write_dense(secret, keys, backend, conversation, on_progress, density=_density(args))
        else:
            cover = hide(secret, keys, backend, conversation, _config(args), on_progress)
    io.errors.print(Text(reader_needs(backend, _compact(args)), style="dim"))
    if args.out:
        save_text(cover, str(args.out), io)
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
    keys = ask_keys(io, confirm=False)
    try:
        print(reveal_with_keys(text, keys))  # The robust format needs no model.
        return
    except RevealError:
        pass
    try:
        print(read_dense(text, keys, _backend(args, io, "Reading")))
    except BackendError as error:
        raise UserError(
            f"No message found without a model, and the compact format could not be tried: {error}"
        ) from error


def _models(args: argparse.Namespace, io: Io) -> None:
    _prepare(args, io)
    for name in _installed(io, _host(args)):
        io.console.print(Text(name))


def _config(args: argparse.Namespace) -> EmbedConfig:
    temperature = getattr(args, "temperature", EmbedConfig().temperature)
    if not 0 < temperature <= 2:
        raise UserError("The temperature must be above 0 and at most 2.")
    return EmbedConfig(temperature=temperature)


def _prepare(args: argparse.Namespace, io: Io) -> None:
    if io.prepare is not None:
        io.prepare(io, _host(args))


def _density(args: argparse.Namespace) -> str:
    return getattr(args, "density", "natural")


def _compact(args: argparse.Namespace) -> bool:
    return not getattr(args, "robust", False)


def _host(args: argparse.Namespace) -> str:
    host = getattr(args, "host", None) or os.environ.get(HOST_ENV) or DEFAULT_HOST
    return host if "://" in host else f"http://{host}"


def _installed(io: Io, host: str) -> list[str]:
    installed = io.list_models(host)
    if not installed:
        raise UserError("Ollama has no models yet. Get one with `ollama pull qwen3:4b`.")
    return installed


def _model(args: argparse.Namespace, io: Io) -> str:
    chosen = getattr(args, "model", None) or os.environ.get(MODEL_ENV)
    return chosen or _installed(io, _host(args))[0]


def _backend(args: argparse.Namespace, io: Io, purpose: str) -> Backend:
    _prepare(args, io)
    model = _model(args, io)
    io.errors.print(f"[dim]{purpose} with {escape(model)}[/dim]")
    return io.make_backend(model, _host(args))


if __name__ == "__main__":
    sys.exit(main())
