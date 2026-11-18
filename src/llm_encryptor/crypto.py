"""Password-based authenticated encryption for the hidden payload.

Secrecy comes from here, not from the language model: scrypt turns the
password into keys and AES-256-CCM encrypts and authenticates the secret.
The nonce and tag are kept short because every payload bit costs one word
of cover text.
"""

from __future__ import annotations

import hashlib
import os
import unicodedata
from dataclasses import dataclass, field

from cryptography.exceptions import InvalidTag
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
    """Keys derived from one password: one to encrypt, one to hash words."""

    enc: bytes = field(repr=False)
    mac: bytes = field(repr=False)


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
        dklen=2 * _KEY_LEN,
    )
    return Keys(enc=raw[:_KEY_LEN], mac=raw[_KEY_LEN:])


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
