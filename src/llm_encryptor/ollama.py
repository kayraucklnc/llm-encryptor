"""Backend for a local Ollama server (https://ollama.com)."""

from __future__ import annotations

import http.client
import json
import math
import re
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator, Sequence

from .backend import BackendError, Choice, Message

DEFAULT_HOST = "http://localhost:11434"
_TIMEOUT_SECONDS = 300
_TOP_TOKENS = 20
# Makes the server pick almost evenly among its 40 likeliest tokens. top_p
# must be off: it would otherwise keep only the one overwhelming favourite.
_PICK_ANYTHING = {"temperature": 100.0, "top_k": 40, "top_p": 1.0, "min_p": 0.0}
# Reasoning models think before they answer; allow for that at the start.
_MAX_OPENING_CHUNKS = 8192
_CONTROL_TOKEN = re.compile(r"<\|\w+\|>|</s>|<eos>|<end_of_turn>")

Transport = Callable[[str, "dict | None"], Iterator[dict]]


def http_transport(host: str) -> Transport:
    """Send JSON to the server and yield each JSON line of its answer."""

    def send(path: str, body: dict | None) -> Iterator[dict]:
        data = None if body is None else json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            host.rstrip("/") + path, data, {"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response:
                for line in response:
                    if line.strip():
                        yield json.loads(line)
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", "replace")[:300]
            raise BackendError(f"Ollama answered {error.code}: {detail}") from error
        except (urllib.error.URLError, http.client.HTTPException, OSError) as error:
            raise BackendError(
                f"Could not reach Ollama at {host}. Start it with `ollama serve`."
            ) from error
        except json.JSONDecodeError as error:
            raise BackendError("Ollama sent an answer that could not be read.") from error

    return send


class OllamaBackend:
    def __init__(
        self,
        model: str,
        host: str = DEFAULT_HOST,
        transport: Transport | None = None,
        options: dict | None = None,
    ) -> None:
        self.model = model
        self._send = transport or http_transport(host)
        self._options = dict(options or {})

    def next_tokens(
        self, messages: Sequence[Message], prefix: str, attempt: int = 0
    ) -> Sequence[Choice]:
        """Ask for one token and read the alternatives the model weighed.

        Ollama reports nothing when the token it picked ends the reply. On a
        later attempt it picks at random instead, with the attempt number as
        the seed, so that it lands on a real token and reports the list.
        """
        picking = (
            {"temperature": 0.0, "seed": 0}
            if attempt == 0
            else {**_PICK_ANYTHING, "seed": attempt}
        )
        body = {
            "model": self.model,
            "messages": [*messages, {"role": "assistant", "content": prefix}],
            "stream": False,
            "think": False,
            "logprobs": True,
            "top_logprobs": _TOP_TOKENS,
            # Seeded: the token the server itself picks stays in its cache and
            # so affects the numbers of the next call. It must be the same for
            # writer and reader. The reported probabilities are raw.
            "options": {**self._options, "num_predict": 1, **picking},
        }
        for answer in self._send("/api/chat", body):
            _raise_on_error(answer)
            positions = answer.get("logprobs")
            if positions:
                return [_choice(item) for item in positions[0].get("top_logprobs", [])]
            if answer.get("message", {}).get("content"):
                raise BackendError(
                    "This Ollama does not report token probabilities. Update Ollama and try again."
                )
            return [Choice("", 1.0)]  # No token at all: the model chose to end its reply.
        raise BackendError("Ollama returned an empty answer.")

    def reset(self) -> None:
        """Unload the model. The next request loads it afresh, with nothing cached."""
        for answer in self._send("/api/generate", {"model": self.model, "keep_alive": 0}):
            _raise_on_error(answer)

    def start(self, messages: Sequence[Message], pieces: int = 8) -> str:
        """Stream a reply and stop once ``pieces`` pieces of visible text have arrived."""
        body = {
            "model": self.model,
            "messages": list(messages),
            "stream": True,
            "options": dict(self._options),
        }
        got: list[str] = []
        stream = self._send("/api/chat", body)
        try:
            for count, chunk in enumerate(stream):
                _raise_on_error(chunk)
                piece = chunk.get("message", {}).get("content", "")
                if piece:
                    got.append(piece)
                if len(got) >= pieces or chunk.get("done") or count >= _MAX_OPENING_CHUNKS:
                    break
        finally:
            close = getattr(stream, "close", None)
            if close is not None:
                close()
        return "".join(got)


def list_models(host: str = DEFAULT_HOST, transport: Transport | None = None) -> list[str]:
    send = transport or http_transport(host)
    for answer in send("/api/tags", None):
        return [model["name"] for model in answer.get("models", []) if model.get("name")]
    return []


def _choice(item: dict) -> Choice:
    token = item.get("token", "")
    text = "" if _CONTROL_TOKEN.fullmatch(token) else token
    return Choice(text, math.exp(item.get("logprob", -math.inf)))


def _raise_on_error(answer: dict) -> None:
    if "error" in answer:
        raise BackendError(f"Ollama error: {answer['error']}")
