import pytest

from llm_encryptor.crypto import NONCE_LEN, OVERHEAD, SecretBox, derive_keys


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
