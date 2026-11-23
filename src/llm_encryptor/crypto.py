"""Password-based authenticated encryption for the hidden payload.

Secrecy comes from here, not from the language model: scrypt turns the
password into keys and AES-256-CCM encrypts and authenticates the secret.
The nonce and tag are kept short because every payload bit costs one word
of cover text.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass, field

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.ciphers.aead import AESCCM

NONCE_LEN = 8
TAG_LEN = 8
OVERHEAD = NONCE_LEN + TAG_LEN
# Every byte costs words of cover text, so secrets are kept short.
MAX_SECRET_BYTES = 1024

_SALT = b"llm-encryptor/v1"
_SCRYPT_N = 2**15
_SCRYPT_R = 8
_SCRYPT_P = 1
_SCRYPT_MAXMEM = 2**27
_KEY_LEN = 32


@dataclass(frozen=True)
class Keys:
    """Keys derived from one password.

    ``enc`` and ``mac`` serve the word-hash format (encrypt, hash words);
    ``dense`` encrypts in the compact format.
    """

    enc: bytes = field(repr=False)
    mac: bytes = field(repr=False)
    dense: bytes = field(repr=False, default=b"")


def derive_keys(password: str) -> Keys:
    if not password:
        raise ValueError("The password must not be empty.")
    # Normalised so the same password typed on another keyboard gives the same keys.
    raw = hashlib.scrypt(
        unicodedata.normalize("NFKC", password).encode("utf-8"),
        salt=_SALT,
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
        maxmem=_SCRYPT_MAXMEM,
        dklen=3 * _KEY_LEN,
    )
    return Keys(
        enc=raw[:_KEY_LEN], mac=raw[_KEY_LEN : 2 * _KEY_LEN], dense=raw[2 * _KEY_LEN :]
    )


class SecretBox:
    """Seals and opens payloads with one encryption key."""

    def __init__(self, enc_key: bytes) -> None:
        self._aead = AESCCM(enc_key, tag_length=TAG_LEN)

    def seal(self, plaintext: bytes) -> bytes:
        nonce = os.urandom(NONCE_LEN)
        return nonce + self._aead.encrypt(nonce, plaintext, None)

    def open(self, blob: bytes) -> bytes | None:
        """Return the plaintext, or None when the blob is not authentic."""
        if len(blob) <= OVERHEAD:
            return None
        try:
            return self._aead.decrypt(blob[:NONCE_LEN], blob[NONCE_LEN:], None)
        except InvalidTag:
            return None


DENSE_SALT_LEN = 2
DENSE_TAG_LEN = 4
DENSE_OVERHEAD = DENSE_SALT_LEN + DENSE_TAG_LEN
_DENSE_NONCE_LEN = 13
_DENSE_LABEL = b"llm-encryptor/dense/v1\x00"


class DenseBox:
    """Sealing for the compact format, where every bit of overhead shows.

    Only two random bytes travel with the message. The rest of the nonce is
    a hash of the text's opening words, which the reader sees anyway, so two
    messages share a nonce only if both the opening and those bytes repeat.
    The tag is four bytes: it tells a real message from noise (a reader that
    tries a thousand readings of a text is fooled about once in 100,000
    texts), but it is too short to stop a determined forger.
    """

    def __init__(self, dense_key: bytes) -> None:
        self._aead = AESCCM(dense_key, tag_length=DENSE_TAG_LEN)

    def seal(self, plaintext: bytes, opening: str, salt_len: int = DENSE_SALT_LEN) -> bytes:
        salt = os.urandom(salt_len)
        return salt + self._aead.encrypt(_dense_nonce(opening, salt), plaintext, None)

    def open(self, blob: bytes, opening: str, salt_len: int = DENSE_SALT_LEN) -> bytes | None:
        """Return the plaintext, or None when the blob is not authentic.

        A text with a short opening carries more random bytes instead, because
        a few opening words repeat too easily to keep nonces apart.
        """
        if len(blob) <= salt_len + DENSE_TAG_LEN:
            return None
        salt, sealed = blob[:salt_len], blob[salt_len:]
        try:
            return self._aead.decrypt(_dense_nonce(opening, salt), sealed, None)
        except InvalidTag:
            return None


def _dense_nonce(opening: str, salt: bytes) -> bytes:
    digest = hashlib.sha256(_DENSE_LABEL + opening.encode("utf-8")).digest()
    return digest[: _DENSE_NONCE_LEN - len(salt)] + salt


LEAN_TAG_BITS = 32
_LEAN_LABEL = b"llm-encryptor/lean/v1\x00"


class LeanBox:
    """Sealing with nothing but a tag added, for the shortest texts.

    No random bytes travel with the message: the tag is a keyed hash of the
    message and the opening words, and it doubles as the nonce (the idea of
    SIV mode). The message is a string of bits, not bytes, so nothing is
    padded either. The price of sending no random bytes: the same message
    under the same password and opening seals the same way every time. Two
    different messages share a nonce only if their openings match and their
    32-bit tags collide.

    The reader checks the tag at every possible length of every reading of a
    text, a few hundred thousand checks at worst, so a text that holds no
    message passes for one about once in 10,000 such texts.
    """

    def __init__(self, dense_key: bytes) -> None:
        self._mac_key = hmac.new(dense_key, _LEAN_LABEL + b"mac", hashlib.sha256).digest()
        self._enc_key = hmac.new(dense_key, _LEAN_LABEL + b"enc", hashlib.sha256).digest()

    def seal(self, bits: str, opening: str) -> str:
        """The tag followed by the encrypted bits, as a string of 0s and 1s."""
        if not bits:
            raise ValueError("There is nothing to seal.")
        about = _about(opening)
        message = int(bits, 2)
        tag = next(tag for length, tag in self._tags(about, message, len(bits)) if length == len(bits))
        return _as_bits(tag, LEAN_TAG_BITS) + _as_bits(message ^ self._stream(about, tag, len(bits)), len(bits))

    def open(self, bits: str, opening: str) -> Iterator[str]:
        """Every message that a leading part of ``bits`` holds.

        The reader does not know where the message ends, so each length is
        checked against the tag. Normally exactly one fits, or none.
        """
        count = len(bits) - LEAN_TAG_BITS
        if count < 1:
            return
        about = _about(opening)
        tag = int(bits[:LEAN_TAG_BITS], 2)
        plain = int(bits[LEAN_TAG_BITS:], 2) ^ self._stream(about, tag, count)
        for length, found in self._tags(about, plain, count):
            if found == tag:
                yield _as_bits(plain >> (count - length), length)

    def _tags(self, about: bytes, message: int, count: int) -> Iterator[tuple[int, int]]:
        """The tag of each leading part of a ``count``-bit message: (length, tag).

        A tag covers the opening, the bits (the last byte filled up with
        zeros) and their number. Hashing the bytes once and branching off at
        every length keeps a long text from costing a hash of everything per
        length.
        """
        data = (message << (-count % 8)).to_bytes((count + 7) // 8, "big")
        whole = hmac.new(self._mac_key, about, hashlib.sha256)
        for length in range(1, count + 1):
            done, extra = divmod(length, 8)
            branch = whole.copy()
            if extra:
                branch.update(bytes([data[done] & (0xFF << (8 - extra)) & 0xFF]))
            else:
                whole.update(data[done - 1 : done])
                branch = whole.copy()
            branch.update(length.to_bytes(4, "big"))
            yield length, int.from_bytes(branch.digest()[:4], "big")

    def _stream(self, about: bytes, tag: int, length: int) -> int:
        """The first ``length`` bits of the keystream for this tag and opening."""
        start = hashlib.sha256(_LEAN_LABEL + b"iv" + about + tag.to_bytes(4, "big")).digest()[:16]
        size = (length + 7) // 8
        stream = Cipher(algorithms.AES(self._enc_key), modes.CTR(start)).encryptor().update(bytes(size))
        return int.from_bytes(stream, "big") >> (8 * size - length)


def _about(opening: str) -> bytes:
    return hashlib.sha256(_LEAN_LABEL + opening.encode("utf-8")).digest()


def _as_bits(number: int, length: int) -> str:
    return format(number, "b").zfill(length) if length else ""
