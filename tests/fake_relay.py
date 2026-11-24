"""A tiny ntfy server on this computer, speaking the part of its HTTP API the app uses."""

from __future__ import annotations

import itertools
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse


class FakeRelay:
    def __init__(self, clock_offset: int = 0) -> None:
        self.posts: list[dict] = []
        self.offset = clock_offset  # The relay's clock needn't agree with ours.
        self._ids = itertools.count(1)
        self._changed = threading.Condition()
        self._closing = False
        relay = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def log_message(self, *args) -> None:
                pass

            def do_POST(self) -> None:
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                relay.add(self.path.strip("/"), body.decode("utf-8").strip())
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"{}")

            def do_GET(self) -> None:
                url = urlparse(self.path)
                topic = url.path.strip("/").removesuffix("/json")
                since = parse_qs(url.query).get("since", ["all"])[0]
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.end_headers()
                relay.stream(self.wfile, topic, since)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def now(self) -> int:
        return int(time.time()) + self.offset

    def add(self, topic: str, message: str, when: int | None = None) -> None:
        with self._changed:
            self.posts.append({
                "id": f"m{next(self._ids)}", "time": self.now() if when is None else when,
                "event": "message", "topic": topic, "message": message,
            })  # fmt: skip
            self._changed.notify_all()

    def stream(self, out, topic: str, since: str) -> None:
        def send(event: dict) -> None:
            out.write(json.dumps(event).encode() + b"\n")
            out.flush()

        try:
            send({"id": "open", "time": self.now(), "event": "open", "topic": topic})
            ids = [post["id"] for post in self.posts]
            done = 0 if since == "all" else ids.index(since) + 1 if since in ids else len(ids)
            while not self._closing:
                with self._changed:
                    fresh = self.posts[done:]
                    if not fresh:
                        self._changed.wait(0.1)
                        continue
                done += len(fresh)
                for post in fresh:
                    if post["topic"] == topic:
                        send(post)
        except OSError:
            pass  # The listener went away.

    def close(self) -> None:
        self._closing = True
        self.server.shutdown()
        self.server.server_close()
