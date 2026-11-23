import time

import pytest

from llm_encryptor.crypto import (
    LEAN_TAG_BITS,
    NONCE_LEN,
    OVERHEAD,
    LeanBox,
    SecretBox,
    derive_keys,
)


def test_same_password_gives_same_keys(keys):
    assert derive_keys("correct horse battery staple") == keys


def test_different_passwords_give_different_keys(keys, other_keys):
    assert keys.enc != other_keys.enc
    assert keys.mac != other_keys.mac
    assert keys.enc != keys.mac


def test_password_is_the_same_however_it_was_typed():
    assert derive_keys("caf\u00e9") == derive_keys("cafe\u0301")


def test_keys_do_not_show_in_repr(keys):
    assert keys.enc.hex() not in repr(keys) and "enc" not in repr(keys)


def test_empty_password_is_refused():
    with pytest.raises(ValueError):
        derive_keys("")


def test_seal_then_open_returns_plaintext(keys):
    box = SecretBox(keys.enc)
    blob = box.seal(b"meet at noon")
    assert len(blob) == len(b"meet at noon") + OVERHEAD
    assert box.open(blob) == b"meet at noon"


def test_sealing_twice_gives_different_blobs(keys):
    box = SecretBox(keys.enc)
    assert box.seal(b"same") != box.seal(b"same")


def test_wrong_key_does_not_open(keys, other_keys):
    blob = SecretBox(keys.enc).seal(b"meet at noon")
    assert SecretBox(other_keys.enc).open(blob) is None


@pytest.mark.parametrize("index", [0, NONCE_LEN, -1])
def test_tampered_blob_does_not_open(keys, index):
    box = SecretBox(keys.enc)
    blob = bytearray(box.seal(b"meet at noon"))
    blob[index] ^= 1
    assert box.open(bytes(blob)) is None


def test_too_short_blob_does_not_open(keys):
    assert SecretBox(keys.enc).open(b"x" * OVERHEAD) is None


def test_lean_box_needs_no_random_bytes(keys, other_keys):
    box = LeanBox(keys.dense)
    sealed = box.seal("0110100", "Once upon a time")
    assert len(sealed) == 7 + LEAN_TAG_BITS and set(sealed) <= {"0", "1"}
    assert sealed[LEAN_TAG_BITS:] != "0110100"
    assert list(box.open(sealed, "Once upon a time")) == ["0110100"]
    # The same message under the same opening seals the same way; another opening does not.
    assert box.seal("0110100", "Once upon a time") == sealed
    assert box.seal("0110100", "Another opening")[:LEAN_TAG_BITS] != sealed[:LEAN_TAG_BITS]
    assert list(box.open(sealed, "Another opening")) == []
    assert list(LeanBox(other_keys.dense).open(sealed, "Once upon a time")) == []


def test_lean_box_finds_the_message_before_whatever_follows_it(keys):
    box = LeanBox(keys.dense)
    sealed = box.seal("1" * 40, "x")
    assert list(box.open(sealed + "0101101110", "x")) == ["1" * 40]
    assert list(box.open(sealed[:-1], "x")) == []
    assert list(box.open(sealed[:LEAN_TAG_BITS], "x")) == []
    flipped = sealed[:5] + ("1" if sealed[5] == "0" else "0") + sealed[6:]
    assert list(box.open(flipped, "x")) == []


def test_lean_box_tells_messages_of_different_lengths_apart(keys):
    box = LeanBox(keys.dense)
    assert box.seal("0", "x")[:LEAN_TAG_BITS] != box.seal("00", "x")[:LEAN_TAG_BITS]
    with pytest.raises(ValueError):
        box.seal("", "x")


@pytest.mark.parametrize("length", [1, 7, 8, 9, 16, 17, 63, 64, 200])
def test_lean_box_opens_every_length(keys, length):
    box = LeanBox(keys.dense)
    message = ("10110" * 40)[:length]
    assert list(box.open(box.seal(message, "x") + "110", "x")) == [message]


def test_lean_box_stays_quick_on_a_long_text(keys):
    box = LeanBox(keys.dense)
    began = time.perf_counter()
    assert list(box.open("01" * 4000, "x")) == []
    assert time.perf_counter() - began < 0.5


@pytest.mark.parametrize("tag_bits", [16, 20, 25])
def test_lean_box_with_a_shorter_tag(keys, other_keys, tag_bits):
    box = LeanBox(keys.dense, tag_bits)
    sealed = box.seal("0110100111", "Once upon a time")
    assert len(sealed) == 10 + tag_bits
    assert list(box.open(sealed + "01", "Once upon a time")) == ["0110100111"]
    assert list(box.open(sealed[:tag_bits], "Once upon a time")) == []
    assert list(box.open(sealed, "Another opening")) == []
    assert list(LeanBox(other_keys.dense, tag_bits).open(sealed, "Once upon a time")) == []
    # A tag of another length is a different sealing, not a cut of the same one.
    assert LeanBox(keys.dense).seal("0110100111", "Once upon a time")[:tag_bits] != sealed[:tag_bits]


def test_lean_box_refuses_a_tag_size_it_cannot_make(keys):
    for tag_bits in (0, 33):
        with pytest.raises(ValueError):
            LeanBox(keys.dense, tag_bits)
