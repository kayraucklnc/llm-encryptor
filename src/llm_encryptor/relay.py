"""A meeting point on the internet, for two computers that cannot reach each other.

A phone's hotspot, like most home and mobile networks, hides the computers on
it behind the provider's router, so two people on two such networks can never
call each other directly. Both can reach a public relay, though: an ntfy
server (https://ntfy.sh unless ``$LLM_ENCRYPTOR_RELAY`` names another; ``off``
turns it off), which hands whatever is posted to a topic to everyone listening
on it, with no account.

The topic and the key both come from the shared password and the ID, so only
the two of them meet there, and every post is sealed with AES-GCM: the relay
sees when and how much is sent, never what, by whom or with which model.
"""

from __future__ import annotations

import asyncio
import base64
import contextvars
import hashlib
import hmac
import http.client
import json
import os
import threading
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .crypto import Keys

DEFAULT_RELAY = "https://ntfy.sh"
RELAY_ENV = "LLM_ENCRYPTOR_RELAY"
_OFF = {"", "off", "no", "none"}
# ntfy turns a longer post into a file attachment, which the listener would not get.
MAX_POST_BYTES = 4096
_LABEL = b"llm-encryptor/relay/v1\x00"
_NONCE_LEN = 12
_POST_SECONDS = 15
# ntfy sends a keepalive every 45 s; silence well past that means the line is dead.
_IDLE_SECONDS = 100
_RETRY_SECONDS = (1, 2, 5, 10)


class RelayError(Exception):
    """The relay could not be reached, or would not take a post."""


def relay_url(setting: str | None = None) -> str | None:
    """The relay to use: ``setting``, else ``$LLM_ENCRYPTOR_RELAY``; None when it is off."""
    value = os.environ.get(RELAY_ENV, DEFAULT_RELAY) if setting is None else setting
    value = value.strip()
    return None if value.lower() in _OFF else value.rstrip("/")


@dataclass(frozen=True)
class Channel:
    """Where on the relay two people meet, and the key their posts are sealed with."""

    topic: str
    key: bytes = field(repr=False)

    @classmethod
    def between(cls, keys: Keys, place: str) -> Channel:
        """The channel for one password and one waiting side's address (``host:port``)."""
        root = hmac.new(keys.mac, _LABEL + place.encode("utf-8"), hashlib.sha256).digest()
        name = hmac.new(root, b"topic", hashlib.sha256).digest()[:15]
        topic = base64.b32encode(name).decode("ascii").lower()
        return cls(topic, hmac.new(root, b"key", hashlib.sha256).digest())

    def seal(self, message: dict) -> str:
        nonce = os.urandom(_NONCE_LEN)
        plain = json.dumps(message, ensure_ascii=False).encode("utf-8")
        sealed = AESGCM(self.key).encrypt(nonce, plain, self.topic.encode("ascii"))
        return base64.urlsafe_b64encode(nonce + sealed).decode("ascii").rstrip("=")

    def unseal(self, post: str) -> dict | None:
        """The message in a post, or None for anything not sealed with this channel's key."""
        try:
            raw = base64.urlsafe_b64decode(post + "=" * (-len(post) % 4))
            plain = AESGCM(self.key).decrypt(raw[:_NONCE_LEN], raw[_NONCE_LEN:], self.topic.encode("ascii"))
            message = json.loads(plain)
        except (ValueError, InvalidTag):
            return None
        return message if isinstance(message, dict) else None


def post(relay: str, topic: str, body: str) -> None:
    """Post one message to a topic. Blocks; run it in a thread."""
    request = urllib.request.Request(
        f"{relay}/{topic}",
        data=body.encode("utf-8"),
        # Not forwarded to phones through Google's push service; only listeners get it.
        headers={"Firebase": "no"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=_POST_SECONDS) as response:
            response.read()
    except urllib.error.HTTPError as error:
        if error.code == 429:
            raise RelayError("The relay says too many messages; wait a minute.") from error
        raise RelayError(f"The relay refused the message ({error.code}).") from error
    except (OSError, http.client.HTTPException) as error:
        raise RelayError("The relay could not be reached. Is this computer online?") from error


class Listener:
    """Listens on a topic in a thread of its own and hands each event to the loop.

    ``on_event`` gets ntfy's events as dicts, on ``loop``: an ``open`` one each
    time the relay is reached (its ``time`` is the relay's clock), then every
    post, the ones already there first. A dropped line is picked up again
    where it left off. Each runs in the context the listener was made in, so
    an app that keeps state there (Textual does) still finds it.
    """

    def __init__(
        self,
        relay: str,
        topic: str,
        on_event: Callable[[dict], None],
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self._relay, self._topic = relay, topic
        self._on_event, self._loop = on_event, loop
        self._context = contextvars.copy_context()
        self._stopped = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"relay-{topic[:6]}", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        """Stop handing on events; the thread ends at the relay's next keepalive."""
        self._stopped.set()

    def _run(self) -> None:
        since, tries = "all", 0
        while not self._stopped.is_set():
            try:
                url = f"{self._relay}/{self._topic}/json?since={since}"
                with urllib.request.urlopen(url, timeout=_IDLE_SECONDS) as response:
                    tries = 0
                    for line in response:
                        event = _event(line)
                        if event is None:
                            continue
                        if event.get("event") == "message" and isinstance(event.get("id"), str):
                            since = event["id"]
                        if not self._hand_on(event):
                            return
            except (OSError, ValueError, http.client.HTTPException):
                pass  # Offline for a moment, or the relay hung up: try again.
            self._stopped.wait(_RETRY_SECONDS[min(tries, len(_RETRY_SECONDS) - 1)])
            tries += 1

    def _hand_on(self, event: dict) -> bool:
        if self._stopped.is_set():
            return False
        try:
            self._loop.call_soon_threadsafe(self._on_event, event, context=self._context)
        except RuntimeError:  # The loop is gone: the app has closed.
            self._stopped.set()
            return False
        return True


def _event(line: bytes) -> dict | None:
    try:
        event = json.loads(line)
    except ValueError:
        return None
    return event if isinstance(event, dict) else None
