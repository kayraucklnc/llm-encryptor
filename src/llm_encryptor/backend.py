"""What the app needs from a language model."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, TypedDict


class Message(TypedDict):
    role: str
    content: str


@dataclass(frozen=True)
class Choice:
    """One way the model might continue. Empty text means the reply ends here."""

    text: str
    probability: float


class BackendError(RuntimeError):
    """The model could not be reached or refused the request."""


class Backend(Protocol):
    def start(self, messages: Sequence[Message]) -> str:
        """The first few words of a reply to ``messages``."""
        ...

    def next_tokens(self, messages: Sequence[Message], prefix: str) -> Sequence[Choice]:
        """The likeliest next tokens of a reply that so far reads ``prefix``."""
        ...
