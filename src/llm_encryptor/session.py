"""A chat about one cover text: every instruction yields a new version."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from .backend import Backend, Message
from .crypto import Keys
from .dense import write_dense
from .embed import DEFAULT_CONFIG, EmbedConfig, ProgressCallback
from .hide import hide


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
            cover = write_dense(self.secret, self.keys, self.backend, asked, on_progress)
        else:
            cover = hide(self.secret, self.keys, self.backend, asked, self.config, on_progress)
        answer: Message = {"role": "assistant", "content": cover}
        answered = (*asked, answer)
        return replace(self, conversation=answered, cover=cover)
