"""Pack a secret by the word instead of by the letter.

A fixed list holds 4096 everyday words, so a listed word costs 13 bits (one
bit to say "a word" and twelve to say which) however long it is, where the
letter code of textcode needs four or five bits a letter. The space before a
word comes free. Anything else (a name, a number, a comma) is written letter
by letter, at one bit more than usual. A secret that gains nothing this way
is packed by the letter throughout, behind a single marker bit.

The list (data/secret_words.txt) is part of the format: its order must never
change for texts already sent.
"""

from __future__ import annotations

import re
from importlib import resources

from .textcode import char_bits, pack_bits, read_char, unpack_bits

WORDS: tuple[str, ...] = tuple(
    resources.files(__package__).joinpath("data/secret_words.txt").read_text(encoding="utf-8").split()
)
_INDEX = {word: at for at, word in enumerate(WORDS)}
_INDEX_BITS = 12
_BY_LETTER, _BY_WORD = "0", "1"
_LETTER, _LISTED = "0", "1"
_RUN = re.compile(r"[A-Za-z']+")


def pack_words(text: str) -> str:
    """The shorter of the word packing and the letter packing, as 0s and 1s."""
    by_word = _BY_WORD + "".join(_items(text))
    by_letter = _BY_LETTER + pack_bits(text)
    return by_word if len(by_word) < len(by_letter) else by_letter


def unpack_words(bits: str) -> str:
    """Undo ``pack_words``. Every bit must belong to the message, or it raises ValueError."""
    if not bits:
        raise ValueError("Nothing to unpack.")
    if bits[0] == _BY_LETTER:
        return unpack_bits(bits[1:])
    parts: list[str] = []
    at = 1
    while at < len(bits):
        if bits[at] == _LISTED:
            number = bits[at + 1 : at + 1 + _INDEX_BITS]
            if len(number) < _INDEX_BITS:
                raise ValueError("The packed message is cut off.")
            # A listed word brings the space before it, except at the very start.
            parts.append((" " if parts else "") + WORDS[int(number, 2)])
            at += 1 + _INDEX_BITS
        else:
            read = read_char(bits, at + 1)
            if read is None:
                raise ValueError("The packed message is cut off.")
            char, at = read
            parts.append(char)
    return "".join(parts)


def _items(text: str) -> list[str]:
    items: list[str] = []
    at = 0
    while at < len(text):
        word = _listed_word(text, at)
        if word is None:
            items.append(_LETTER + char_bits(text[at]))
            at += 1
        else:
            items.append(_LISTED + format(_INDEX[word], f"0{_INDEX_BITS}b"))
            at += len(word) + (1 if at else 0)
    return items


def _listed_word(text: str, at: int) -> str | None:
    """The listed word that starts the text at 0, or follows the space at ``at``."""
    if at and text[at] != " ":
        return None
    run = _RUN.match(text, at + 1 if at else 0)
    return run.group() if run and run.group() in _INDEX else None
