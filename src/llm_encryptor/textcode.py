"""Pack a short message into fewer bits before it is encrypted.

Every hidden bit costs about a word of cover text, so plain UTF-8 (eight bits
a character) is wasteful. This is a fixed Huffman code tuned for everyday
messages: common letters take three to five bits. Any other character is
still allowed; it is written as an escape followed by its UTF-8 bytes. The
table is part of the format and must never change for texts already sent.
"""

from __future__ import annotations

import heapq

_ESCAPE = "\x00"
_COMMON = {
    " ": 1800, "e": 1020, "t": 750, "a": 650, "o": 620, "i": 570, "n": 570, "s": 530,
    "h": 500, "r": 500, "d": 350, "l": 330, "u": 230, "c": 220, "m": 200, "w": 200,
    "f": 180, "g": 170, "y": 170, "p": 150, "b": 130, "v": 80, "k": 70, "j": 15,
    "x": 15, "q": 10, "z": 10, ".": 100, ",": 100, "'": 40, "?": 30, "!": 30,
    "-": 20, ":": 20, "/": 10, "\n": 10, "@": 5,
}  # fmt: skip
_WEIGHTS = {
    **_COMMON,
    **{digit: 25 for digit in "0123456789"},
    **{letter.upper(): max(5, weight // 15) for letter, weight in _COMMON.items() if letter.isalpha()},
    **{letter: 10 for letter in "çğıöşüÇĞİÖŞÜ"},
    _ESCAPE: 5,
}


def _build() -> dict[str, str]:
    """A Huffman code for the table; ties are broken by symbol so it is stable."""
    heap: list[tuple[int, str, tuple]] = [
        (weight, symbol, (symbol,)) for symbol, weight in sorted(_WEIGHTS.items())
    ]
    heapq.heapify(heap)
    while len(heap) > 1:
        weight_a, name_a, tree_a = heapq.heappop(heap)
        weight_b, name_b, tree_b = heapq.heappop(heap)
        heapq.heappush(heap, (weight_a + weight_b, min(name_a, name_b), (tree_a, tree_b)))
    codes: dict[str, str] = {}

    def walk(tree: tuple, prefix: str) -> None:
        if len(tree) == 1:
            codes[tree[0]] = prefix
            return
        walk(tree[0], prefix + "0")
        walk(tree[1], prefix + "1")

    walk(heap[0][2], "")
    return codes


_CODES = _build()
_SYMBOLS = {code: symbol for symbol, code in _CODES.items()}
# Filler for the last byte: the start of the longest code, which is too long
# to be completed by seven bits and so can never be mistaken for a character.
_FILLER = max(_CODES.values(), key=lambda code: (len(code), code))
_PACKED, _RAW = "0", "1"


def pack(text: str) -> bytes:
    """The shorter of the packed form and plain UTF-8, marked by its first bit."""
    bits = _PACKED + "".join(_encode(char) for char in text)
    bits += _FILLER[: -len(bits) % 8]
    packed = int(bits, 2).to_bytes(len(bits) // 8, "big")
    raw = bytes([0x80]) + text.encode("utf-8")
    return packed if len(packed) <= len(raw) else raw


def unpack(data: bytes) -> str:
    """Undo ``pack``. Raises ValueError (or UnicodeDecodeError) on damaged data."""
    if not data:
        raise ValueError("Nothing to unpack.")
    if data[0] & 0x80:
        if data[0] != 0x80:
            raise ValueError("Unknown packing.")
        return data[1:].decode("utf-8")
    bits = bin(int.from_bytes(data, "big"))[2:].zfill(8 * len(data))[1:]
    chars: list[str] = []
    at = 0
    while at < len(bits):
        end = at + 1
        while end <= len(bits) and bits[at:end] not in _SYMBOLS:
            end += 1
        if end > len(bits):
            break  # Only filler is left.
        symbol = _SYMBOLS[bits[at:end]]
        at = end
        if symbol == _ESCAPE:
            symbol, at = _escaped(bits, at)
        chars.append(symbol)
    return "".join(chars)


def _encode(char: str) -> str:
    if char in _CODES and char != _ESCAPE:
        return _CODES[char]
    return _CODES[_ESCAPE] + "".join(f"{byte:08b}" for byte in char.encode("utf-8"))


def _escaped(bits: str, at: int) -> tuple[str, int]:
    """Read one UTF-8 character written out in full."""
    first = bits[at : at + 8]
    if len(first) < 8:
        raise ValueError("The packed message is cut off.")
    length = 1 if first[0] == "0" else len(first) - len(first.lstrip("1"))
    chunk = bits[at : at + 8 * length]
    if not 1 <= length <= 4 or len(chunk) < 8 * length:
        raise ValueError("The packed message is damaged.")
    return int(chunk, 2).to_bytes(length, "big").decode("utf-8"), at + 8 * length
