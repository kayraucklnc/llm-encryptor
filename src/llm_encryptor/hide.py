"""Hide a secret in a cover text written to the user's instructions."""

from __future__ import annotations

from collections.abc import Sequence

from .backend import Backend, Message
from .codec import RevealError, reveal_with_keys, to_bits
from .crypto import MAX_SECRET_BYTES, OVERHEAD, Keys, SecretBox
from .embed import DEFAULT_CONFIG, EmbedConfig, EmbedError, ProgressCallback, embed
from .words import BITS_PER_BLOCK, WORDS_PER_BLOCK

_ATTEMPTS = 2

_SYSTEM_PROMPT = (
    "You are a skilled writer. Write exactly what the user asks for, as natural, "
    "flowing prose in plain text. Output only the text itself: no preamble, no title, "
    "no headings, no lists, no notes about the task. The text must be at least "
    "{words} words long; do not wrap up before that. When the user asks for changes, "
    "write the complete new version of the text."
)


def payload_bit_count(secret: str) -> int:
    return 8 * (len(secret.encode("utf-8")) + OVERHEAD)


def validate_and_size(secret: str, compact: bool) -> str:
    """A line telling the user how long the text for this secret will be."""
    validate_secret(secret)
    if compact:
        return "Compact format: expect very roughly 15 words per character, plus 100."
    return f"Word-hash format: the text needs at least {words_needed(secret)} words."


def words_needed(secret: str) -> int:
    """How many words of cover text the hidden payload takes up."""
    blocks = -(-payload_bit_count(secret) // BITS_PER_BLOCK)
    return blocks * WORDS_PER_BLOCK + 1


def validate_secret(secret: str) -> str:
    size = len(secret.encode("utf-8"))
    if size == 0:
        raise ValueError("The secret must not be empty.")
    if size > MAX_SECRET_BYTES:
        raise ValueError(
            f"The secret is {size} bytes; the limit is {MAX_SECRET_BYTES} "
            "(every byte needs eight words of cover text)."
        )
    return secret


def system_message(secret: str) -> Message:
    target = int(words_needed(secret) * 1.1) + 20
    return {"role": "system", "content": _SYSTEM_PROMPT.format(words=target)}


def hide(
    secret: str,
    keys: Keys,
    backend: Backend,
    conversation: Sequence[Message],
    config: EmbedConfig = DEFAULT_CONFIG,
    on_progress: ProgressCallback | None = None,
) -> str:
    """Return a cover text that follows ``conversation`` and carries ``secret``.

    The result is checked by decoding it again before it is returned.
    """
    validate_secret(secret)
    messages = [system_message(secret), *conversation]
    box = SecretBox(keys.enc)
    for _ in range(_ATTEMPTS):
        payload = to_bits(box.seal(secret.encode("utf-8")))
        cover = embed(backend, messages, payload, keys.mac, config, on_progress)
        if _decodes_to(cover, keys, secret):
            return cover
    raise EmbedError("The finished text did not decode back to the secret. Please try again.")


def _decodes_to(cover: str, keys: Keys, secret: str) -> bool:
    try:
        return reveal_with_keys(cover, keys) == secret
    except RevealError:
        return False
