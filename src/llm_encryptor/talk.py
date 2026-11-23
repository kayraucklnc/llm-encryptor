"""What the chat app asks of the model, and the one dial it shows."""

from __future__ import annotations

from .backend import Message

DEFAULT_TOPIC = "how the week has been"
# The length dial, from longest and most natural to shortest and strangest:
# (density, what to call it, what it gives).
LENGTHS: tuple[tuple[str, str, str], ...] = (
    ("natural", "long", "≈ 100 words"),
    ("tight", "medium", "≈ 70 words"),
    ("short", "short", "≈ 30 words"),
    ("max", "shortest", "≈ 20 words · garbled"),
)
DEFAULT_LENGTH = 2


def request(topic: str = "", answering: str | None = None) -> list[Message]:
    """What to ask the model for the visible text.

    With a text to answer, the visible text is a reply to it, so that whoever
    watches the line sees one conversation; ``topic`` then steers the reply.
    Without one, ``topic`` is what the first visible text is about.
    """
    topic = " ".join(topic.split())
    if not answering:
        return [{"role": "user", "content": topic or DEFAULT_TOPIC}]
    asked = f'A friend just texted you this:\n"{" ".join(answering.split())}"\nText back a reply to it.'
    if topic:
        asked += f" Bring the reply round to: {topic}."
    return [{"role": "user", "content": asked}]
