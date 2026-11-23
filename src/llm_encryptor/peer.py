"""A direct line between two people: one waits, the other connects by ID.

There is no server in between. An ID is just this computer's address and
port in a form that is easy to read out, so it works wherever one computer
can reach the other: on the same network, or over a VPN. Only cover texts
travel over the line, one JSON object per line.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import ipaddress
import json
import socket
from collections.abc import Callable

# A cover text is a few hundred words at most; anything much longer is not one.
MAX_TEXT_CHARS = 20_000
_LINE_LIMIT = 4 * MAX_TEXT_CHARS
_MAX_NAME_CHARS = 40


class LineError(Exception):
    """The other side could not be reached, or the ID makes no sense."""


def make_id(host: str, port: int) -> str:
    """An address as ten letters and digits, e.g. ``yiyaa-aj3hq``."""
    raw = ipaddress.IPv4Address(host).packed + port.to_bytes(2, "big")
    code = base64.b32encode(raw).decode("ascii").rstrip("=").lower()
    return f"{code[:5]}-{code[5:]}"


def parse_id(text: str) -> tuple[str, int]:
    """The address in an ID. ``host:port`` is accepted too."""
    text = text.strip()
    if ":" in text:
        host, _, port = text.rpartition(":")
        if host and port.isdigit() and 0 < int(port) < 65536:
            return host, int(port)
        raise LineError("That does not look like an address.")
    code = "".join(char for char in text.upper() if char.isalnum())
    try:
        raw = base64.b32decode(code + "=" * (-len(code) % 8))
    except (binascii.Error, ValueError) as error:
        raise LineError("That does not look like an ID.") from error
    if len(raw) != 6 or not int.from_bytes(raw[4:], "big"):
        raise LineError("That does not look like an ID.")
    return str(ipaddress.IPv4Address(raw[:4])), int.from_bytes(raw[4:], "big")


def local_address() -> str:
    """This computer's address on its network (no packet is sent to find it)."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("192.0.2.1", 9))  # an address reserved for documentation
        return probe.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        probe.close()


class Line:
    """One connection to one friend. Callbacks run on the event loop."""

    def __init__(
        self,
        name: str,
        on_text: Callable[[str], None],
        on_change: Callable[[], None] = lambda: None,
    ) -> None:
        self.name = name[:_MAX_NAME_CHARS] or "someone"
        self.friend: str | None = None
        self._on_text, self._on_change = on_text, on_change
        self._server: asyncio.Server | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._listening: asyncio.Task | None = None

    @property
    def connected(self) -> bool:
        return self._writer is not None

    async def open(self, port: int = 0, host: str | None = None) -> str:
        """Start waiting for a friend; returns the ID to give them."""
        host = host or local_address()
        self._server = await asyncio.start_server(self._accept, "0.0.0.0", port, limit=_LINE_LIMIT)
        return make_id(host, self._server.sockets[0].getsockname()[1])

    async def call(self, friend_id: str) -> None:
        """Connect to a friend who is waiting."""
        host, port = parse_id(friend_id)
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port, limit=_LINE_LIMIT), timeout=8
            )
        except (OSError, asyncio.TimeoutError) as error:
            raise LineError("Nobody answers at that ID. Is your friend's app open?") from error
        await self._adopt(reader, writer)

    async def send(self, text: str) -> None:
        if self._writer is None:
            raise LineError("Not connected.")
        await self._write({"type": "text", "text": text})

    async def close(self) -> None:
        if self._listening is not None:
            self._listening.cancel()
        self._drop()
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    async def _accept(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        if self._writer is not None:
            writer.close()  # One friend at a time.
            return
        await self._adopt(reader, writer)

    async def _adopt(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self._writer = writer
        await self._write({"type": "hello", "name": self.name})
        self._listening = asyncio.create_task(self._listen(reader))

    async def _write(self, message: dict) -> None:
        assert self._writer is not None
        try:
            self._writer.write(json.dumps(message).encode("utf-8") + b"\n")
            await self._writer.drain()
        except OSError as error:
            self._drop()
            raise LineError("The connection was lost.") from error

    async def _listen(self, reader: asyncio.StreamReader) -> None:
        """Take in what the friend sends. Nothing from the other side is trusted."""
        try:
            while line := await reader.readline():
                self._take(line)
        except (OSError, ValueError, asyncio.LimitOverrunError):
            pass  # A broken line or one far too long: hang up.
        self._drop()

    def _take(self, line: bytes) -> None:
        try:
            message = json.loads(line)
        except ValueError:
            return
        if not isinstance(message, dict):
            return
        if message.get("type") == "hello" and isinstance(message.get("name"), str):
            self.friend = message["name"].strip()[:_MAX_NAME_CHARS] or "friend"
            self._on_change()
        elif (
            message.get("type") == "text"
            and isinstance(message.get("text"), str)
            and 0 < len(message["text"]) <= MAX_TEXT_CHARS
        ):
            self._on_text(message["text"])

    def _drop(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
            self._on_change()
