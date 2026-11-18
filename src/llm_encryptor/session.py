"""A chat about one cover text: every instruction yields a new version."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from .backend import Backend, Message
from .crypto import Keys
from .dense import write_dense
from .embed import DEFAULT_CONFIG, EmbedConfig, ProgressCallback
from .hide import hide, system_message, validate_secret


@dataclass(frozen=True)
class Session:
    secret: str = field(repr=False)
    keys: Keys = field(repr=False)
    backend: Backend
    config: EmbedConfig = DEFAULT_CONFIG
    conversation: tuple[Message, ...] = ()
    cover: str | None = None
    # Compact: short texts, but the reader needs the same model.
    compact: bool = True

    def instruct(self, instruction: str, on_progress: ProgressCallback | None = None) -> Session:
        """Write or revise the cover text; the secret stays embedded."""
        request: Message = {"role": "user", "content": instruction}
        asked = (*self.conversation, request)
        if self.compact:
            opening_messages = [system_message(self.secret), *asked]
            cover = write_dense(self.secret, self.keys, self.backend, opening_messages, on_progress)
        else:
            cover = hide(self.secret, self.keys, self.backend, asked, self.config, on_progress)
        answer: Message = {"role": "assistant", "content": cover}
        answered = (*asked, answer)
        return replace(self, conversation=answered, cover=cover)

    def with_secret(self, secret: str) -> Session:
        return replace(self, secret=validate_secret(secret))

    def with_backend(self, backend: Backend) -> Session:
        return replace(self, backend=backend)

    def with_mode(self, compact: bool) -> Session:
        return replace(self, compact=compact)

    def restarted(self) -> Session:
        return replace(self, conversation=(), cover=None)
