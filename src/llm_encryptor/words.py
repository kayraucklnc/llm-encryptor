"""Turn text into words, and groups of words into key-dependent bits.

Words are taken in blocks; a keyed hash of each block gives its bits. Pooling
several words per block matters: many single words are forced by grammar
("Have you ever"), but a block almost always leaves the writer a choice.
Only letters and digits count, ignoring case and accents, so re-wrapping the
text or changing its punctuation and capitalisation does not change the bits.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import unicodedata
from collections.abc import Sequence

# Part of the format: sender and reader must agree on both.
WORDS_PER_BLOCK = 3
BITS_PER_BLOCK = 2

_WORD = re.compile(r"[^\W_]+")


def clean(text: str) -> str:
    """Canonical form for text we generate (keeps case and punctuation)."""
    return unicodedata.normalize("NFKC", text)


def split_words(text: str) -> list[str]:
    """The words that carry bits: letters and digits, without case or accents."""
    decomposed = unicodedata.normalize("NFKD", text)
    bare = "".join(char for char in decomposed if not unicodedata.combining(char))
    return _WORD.findall(bare.casefold())


def has_word(text: str) -> bool:
    return _WORD.search(text) is not None


def is_word_char(char: str) -> bool:
    return _WORD.fullmatch(char) is not None


def complete_word_ends(text: str) -> list[int]:
    """Offsets just after each word that is followed by a non-word character."""
    return [m.end() for m in _WORD.finditer(text) if m.end() < len(text)]


def joins_cleanly(text: str, continuation: str) -> bool:
    """True when appending the continuation leaves the last word of text intact."""
    if not continuation:
        return False
    first = continuation[0]
    if unicodedata.combining(first):
        return False
    return not (text and is_word_char(text[-1]) and is_word_char(first))


def block_bits(mac_key: bytes, block: Sequence[str]) -> tuple[int, ...]:
    """The bits carried by one block of words."""
    message = "\x00".join(block).encode("utf-8")
    first_byte = hmac.new(mac_key, message, hashlib.sha256).digest()[0]
    return tuple((first_byte >> (7 - i)) & 1 for i in range(BITS_PER_BLOCK))


def text_bits(mac_key: bytes, words: Sequence[str]) -> list[int]:
    """Bits of every complete block, in order."""
    bits: list[int] = []
    for start in range(0, len(words) - WORDS_PER_BLOCK + 1, WORDS_PER_BLOCK):
        bits.extend(block_bits(mac_key, words[start : start + WORDS_PER_BLOCK]))
    return bits
