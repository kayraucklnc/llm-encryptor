import pytest

from llm_encryptor.codec import (
    MAX_LEAD_WORDS,
    RevealError,
    from_bits,
    reveal,
    reveal_with_keys,
    to_bits,
)


def test_bits_round_trip():
    data = bytes(range(256))
    assert from_bits(to_bits(data)) == data


def test_to_bits_is_most_significant_first():
    assert to_bits(b"\x80\x01") == (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1)


def test_from_bits_drops_an_incomplete_byte():
    assert from_bits([1] * 8 + [1, 0, 1]) == b"\xff"


def test_plain_text_has_no_secret(keys):
    with pytest.raises(RevealError):
        reveal_with_keys("Just an ordinary paragraph about nothing in particular. " * 20, keys)


def test_short_text_has_no_secret(keys):
    with pytest.raises(RevealError):
        reveal_with_keys("Too short.", keys)


def test_reveal_takes_a_password():
    with pytest.raises(RevealError):
        reveal("Nothing hidden here at all.", "some password")


def test_lead_limit_is_positive():
    assert MAX_LEAD_WORDS > 0


def test_a_very_long_text_is_searched_in_bounded_time(keys):
    import time

    started = time.monotonic()
    with pytest.raises(RevealError):
        reveal_with_keys("lorem ipsum dolor sit amet " * 4000, keys)
    assert time.monotonic() - started < 20
