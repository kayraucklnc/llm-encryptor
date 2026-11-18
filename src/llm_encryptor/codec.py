"""Recover a hidden secret from cover text. Needs only the password."""

from __future__ import annotations

from collections.abc import Sequence

from .crypto import MAX_SECRET_BYTES, OVERHEAD, Keys, SecretBox, derive_keys
from .words import split_words, text_bits

# The payload may start after a few free words, so the reader tries each start.
MAX_LEAD_WORDS = 64


class RevealError(ValueError):
    """No secret could be recovered from the text with this password."""


def to_bits(data: bytes) -> tuple[int, ...]:
    return tuple((byte >> shift) & 1 for byte in data for shift in range(7, -1, -1))


def from_bits(bits: Sequence[int]) -> bytes:
    """Pack bits into bytes, dropping any incomplete final byte."""
    usable = len(bits) - len(bits) % 8
    return bytes(
        sum(bit << (7 - i) for i, bit in enumerate(bits[start : start + 8]))
        for start in range(0, usable, 8)
    )


def reveal_with_keys(text: str, keys: Keys) -> str:
    words = split_words(text)
    box = SecretBox(keys.enc)
    for lead in range(MAX_LEAD_WORDS + 1):
        blob = from_bits(text_bits(keys.mac, words[lead:]))
        if len(blob) <= OVERHEAD:
            break
        longest = min(len(blob), OVERHEAD + MAX_SECRET_BYTES)
        for length in range(OVERHEAD + 1, longest + 1):
            plaintext = box.open(blob[:length])
            if plaintext is not None:
                return plaintext.decode("utf-8")
    raise RevealError(
        "No hidden message found. The password may be wrong, or words in the text were changed."
    )


def reveal(text: str, password: str) -> str:
    return reveal_with_keys(text, derive_keys(password))
