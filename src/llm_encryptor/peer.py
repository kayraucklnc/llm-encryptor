"""A line between two people: one waits, the other connects by ID.

An ID is the waiting computer's address and port in a form that is easy to
read out. The caller first tries that address directly, which works on the
same network or over a VPN. When it cannot get through (two phone hotspots,
two homes), both sides meet on a public relay instead (see ``relay.py``),
in a place only the same password and ID lead to. Only cover texts travel,
one JSON object at a time.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import ipaddress
import json
import secrets
import socket
from collections.abc import Callable

from .crypto import Keys
from .relay import MAX_POST_BYTES, Channel, Listener, RelayError, post, relay_url

# A cover text is a few hundred words at most; anything much longer is not one.
MAX_TEXT_CHARS = 20_000
_LINE_LIMIT = 4 * MAX_TEXT_CHARS
_MAX_NAME_CHARS = 40
_MAX_MODEL_CHARS = 120
# On the same network an answer takes milliseconds; past this it is not coming.
_DIRECT_SECONDS = 4
_REACH_SECONDS = 15
# The caller asks again this often, in case the other side was not listening yet.
_KNOCK_SECONDS = 20
_CALL_SECONDS = 180
_BYE_SECONDS = 3
# Posts the relay still holds from longer ago than this belong to an old chat.
_STALE_SECONDS = 600

NOBODY = "Nobody answers at that ID. Is your friend's app open and waiting, with the same password?"


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


class _Direct:
    """The line straight to the other computer."""

    def __init__(self, writer: asyncio.StreamWriter) -> None:
        self._writer = writer

    async def send(self, message: dict) -> None:
        try:
            self._writer.write(json.dumps(message).encode("utf-8") + b"\n")
            await self._writer.drain()
        except OSError as error:
            raise LineError("The connection was lost.") from error

    def close(self) -> None:
        self._writer.close()


class _Meeting:
    """Two sides finding each other on the relay, then talking through it.

    Each side posts under a random session name, so it can skip its own posts
    and the waiting side can tell one caller from another. The caller says
    hello; the waiting side answers it by name, and from then on each listens
    only to the other.
    """

    def __init__(self, line: Line, relay: str, channel: Channel, calling: bool) -> None:
        self._line, self._relay, self.channel, self._calling = line, relay, channel, calling
        self.session = secrets.token_hex(8)
        self.peer: str | None = None
        self._seen: set[str] = set()  # ntfy may hand the same post over twice
        self._now: int | None = None  # the relay's clock when first reached
        self._reached = asyncio.Event()
        self._answers: set[asyncio.Task] = set()
        self.stopped = False
        self._listener = Listener(relay, channel.topic, self._event, asyncio.get_running_loop())
        self._listener.start()

    async def reached(self) -> None:
        try:
            await asyncio.wait_for(self._reached.wait(), timeout=_REACH_SECONDS)
        except asyncio.TimeoutError as error:
            self.stop()
            raise LineError("The relay could not be reached. Is this computer online?") from error

    async def send(self, message: dict) -> None:
        body = self.channel.seal({**message, "s": self.session, "to": self.peer})
        if len(body.encode("ascii")) > MAX_POST_BYTES:
            raise LineError("That text is too long for the relay; choose a shorter length.")
        try:
            await asyncio.to_thread(post, self._relay, self.channel.topic, body)
        except RelayError as error:
            raise LineError(str(error)) from error

    async def knock(self) -> None:
        await self.send(self._line.hello())

    def parted(self) -> None:
        """The friend left: a caller is done here, the waiting side waits for the next one."""
        if self._calling:
            self.stop()
        self.peer = None

    def stop(self) -> None:
        self.stopped = True
        self._listener.stop()

    def _event(self, event: dict) -> None:
        when = event.get("time")
        if not isinstance(when, int):
            return
        if event.get("event") == "open":
            self._now = when if self._now is None else self._now
            return self._reached.set()
        post_id = event.get("id")
        if event.get("event") != "message" or not isinstance(post_id, str) or post_id in self._seen:
            return
        self._seen.add(post_id)
        if self._now is not None and when < self._now - _STALE_SECONDS:
            return
        message = self.channel.unseal(str(event.get("message", "")))
        if message is not None:
            self._take(message)

    def _take(self, message: dict) -> None:
        sender, hello = message.get("s"), message.get("type") == "hello"
        if not isinstance(sender, str) or sender == self.session:
            return
        if self.peer is None:
            if not hello or self._line.connected:
                return
            if self._calling and message.get("to") != self.session:
                return  # Another caller's hello, or one not meant for us.
            self.peer = sender
            self._line._join(_Relayed(self))
        elif sender != self.peer:
            return  # One friend at a time.
        if hello and not self._calling:
            # Answered every time, in case an earlier answer went astray.
            answer = asyncio.ensure_future(self._answer())
            self._answers.add(answer)
            answer.add_done_callback(self._answers.discard)
        self._line._handle(message)

    async def _answer(self) -> None:
        with contextlib.suppress(LineError):
            await self.knock()


class _Relayed:
    """The line through the relay, once both sides have met there."""

    def __init__(self, meeting: _Meeting) -> None:
        self._meeting = meeting

    async def send(self, message: dict) -> None:
        await self._meeting.send(message)

    def close(self) -> None:
        self._meeting.parted()


class Line:
    """One connection to one friend. Callbacks run on the event loop."""

    def __init__(
        self,
        name: str,
        on_text: Callable[[str], None],
        on_change: Callable[[], None] = lambda: None,
        model: str = "",
        relay: str | None = None,
    ) -> None:
        self.name = name[:_MAX_NAME_CHARS] or "someone"
        self.model = model
        self.friend: str | None = None
        # The model the friend writes and reads with; both sides need the same one.
        self.friend_model: str | None = None
        # The relay to meet on when a direct line fails; None when there is none.
        self.relay = relay_url(relay)
        self._on_text, self._on_change = on_text, on_change
        self._server: asyncio.Server | None = None
        self._place: str | None = None  # this side's own address, once waiting
        self._link: _Direct | _Relayed | None = None
        self._listening: asyncio.Task | None = None
        self._meeting: _Meeting | None = None
        self._greeted: asyncio.Event | None = None

    @property
    def connected(self) -> bool:
        return self._link is not None

    @property
    def relayed(self) -> bool:
        return isinstance(self._link, _Relayed)

    async def open(self, port: int = 0, host: str | None = None) -> str:
        """Start waiting for a friend; returns the ID to give them."""
        host = host or local_address()
        self._server = await asyncio.start_server(self._accept, "0.0.0.0", port, limit=_LINE_LIMIT)
        port = self._server.sockets[0].getsockname()[1]
        self._place = f"{host}:{port}"
        return make_id(host, port)

    async def wait(self, keys: Keys) -> None:
        """Wait on the relay too, for a friend on another network."""
        if self.relay is None or self._place is None:
            return
        await self._meet(Channel.between(keys, self._place), calling=False)

    async def call(
        self,
        friend_id: str,
        keys: Keys | None = None,
        progress: Callable[[str], None] = lambda _: None,
    ) -> None:
        """Connect to a friend who is waiting: directly, else through the relay."""
        host, port = parse_id(friend_id)
        if await self._call_directly(host, port):
            return
        if keys is None or self.relay is None:
            raise LineError(NOBODY)
        progress("not on this network · meeting through the relay…")
        meeting = await self._meet(Channel.between(keys, f"{host}:{port}"), calling=True)
        loop = asyncio.get_running_loop()
        give_up = loop.time() + _CALL_SECONDS
        while not self.connected:
            await meeting.knock()
            if not await self._hear_hello(_KNOCK_SECONDS) and loop.time() > give_up:
                self._stop_meeting()
                raise LineError(NOBODY)

    async def send(self, text: str) -> None:
        if self._link is None:
            raise LineError("Not connected.")
        await self._write({"type": "text", "text": text})

    async def close(self) -> None:
        if self.relayed:
            # Nobody else would tell the friend: the relay has no line to drop.
            with contextlib.suppress(LineError, asyncio.TimeoutError):
                await asyncio.wait_for(self._write({"type": "bye"}), timeout=_BYE_SECONDS)
        if self._listening is not None:
            self._listening.cancel()
        self._drop()
        self._stop_meeting()
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    def hello(self) -> dict:
        return {"type": "hello", "name": self.name, "model": self.model}

    async def _meet(self, channel: Channel, calling: bool) -> _Meeting:
        meeting = self._meeting
        if meeting is not None and meeting.channel == channel and not meeting.stopped:
            return meeting
        self._stop_meeting()  # Another password, say: start over.
        self._meeting = _Meeting(self, self.relay, channel, calling)
        try:
            await self._meeting.reached()
        except LineError:
            self._meeting = None
            raise
        return self._meeting

    def _stop_meeting(self) -> None:
        if self._meeting is not None:
            self._meeting.stop()
            self._meeting = None

    async def _call_directly(self, host: str, port: int) -> bool:
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port, limit=_LINE_LIMIT), timeout=_DIRECT_SECONDS
            )
        except (OSError, asyncio.TimeoutError):
            return False
        await self._adopt(reader, writer)
        if await self._hear_hello(_DIRECT_SECONDS):
            return True
        self._drop()  # Something answered at that address, but not the app.
        return False

    async def _hear_hello(self, seconds: float) -> bool:
        self._greeted = self._greeted or asyncio.Event()
        if self.connected and self.friend is not None:
            return True
        try:
            await asyncio.wait_for(self._greeted.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            return False
        return self.connected

    async def _accept(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        if self.connected:
            writer.close()  # One friend at a time.
            return
        await self._adopt(reader, writer)

    async def _adopt(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self._join(_Direct(writer))
        await self._write(self.hello())
        self._listening = asyncio.create_task(self._listen(reader))

    def _join(self, link: _Direct | _Relayed) -> None:
        self.friend = self.friend_model = None
        self._link = link

    async def _write(self, message: dict) -> None:
        assert self._link is not None
        try:
            await self._link.send(message)
        except LineError:
            if isinstance(self._link, _Direct):
                self._drop()
            raise

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
        if isinstance(message, dict):
            self._handle(message)

    def _handle(self, message: dict) -> None:
        kind = message.get("type")
        if kind == "hello" and isinstance(message.get("name"), str):
            self.friend = message["name"].strip()[:_MAX_NAME_CHARS] or "friend"
            model = message.get("model")
            self.friend_model = model[:_MAX_MODEL_CHARS] if isinstance(model, str) and model else None
            if self._greeted is not None:
                self._greeted.set()
            self._on_change()
        elif kind == "text" and isinstance(message.get("text"), str) and 0 < len(message["text"]) <= MAX_TEXT_CHARS:
            self._on_text(message["text"])
        elif kind == "bye":
            self._drop()

    def _drop(self) -> None:
        if self._link is None:
            return
        link, self._link = self._link, None
        if self._greeted is not None:
            self._greeted.clear()
        link.close()
        self._on_change()
