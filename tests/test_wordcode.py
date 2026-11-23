import random

import pytest

from llm_encryptor.textcode import pack_bits
from llm_encryptor.wordcode import WORDS, pack_words, unpack_words

MESSAGES = [
    "meet at noon",
    "a",
    "",
    "Meet me at the old bridge at 9pm.",
    "the package is under the bench, 9pm sharp",
    "call me when you land, I'll pick you up",
    " at",
    "at ",
    "meet  at",
    "meet,at noon",
    "'at' the door",
    "çay at 5, İstiklal",
    "日本語のメッセージ",
    "X7#kq9!zP",
]


def test_the_word_list_is_fixed():
    """The list is part of the format: its size and order must never change."""
    assert len(WORDS) == len(set(WORDS)) == 4096
    assert WORDS[:4] == ("the", "a", "an", "and") and "noon" in WORDS and "I'll" in WORDS


@pytest.mark.parametrize("text", MESSAGES)
def test_round_trip(text):
    bits = pack_words(text)
    assert set(bits) <= {"0", "1"}
    assert unpack_words(bits) == text


def test_a_listed_word_costs_thirteen_bits_whatever_its_length():
    assert len(pack_words("meet at noon")) == 1 + 3 * 13
    assert len(pack_words("tomorrow afternoon")) == 1 + 2 * 13
    assert len(pack_words("meet at noon")) < len(pack_bits("meet at noon"))


def test_text_without_listed_words_costs_one_bit_more_than_by_the_letter():
    for text in ("X7#kq9!zP", "日本語のメッセージ", "çay"):
        assert pack_words(text) == "0" + pack_bits(text)


def test_random_texts_round_trip():
    rng = random.Random(5)
    pieces = [*WORDS[:300], " ", " ", " ", ",", ".", "9", "'", "X", "é", "\n"]
    for _ in range(300):
        text = "".join(rng.choice(pieces) for _ in range(rng.randint(1, 12)))
        assert unpack_words(pack_words(text)) == text


@pytest.mark.parametrize("bits", ["", "0", pack_words("meet at noon")[:-1], pack_words("meet at noon") + "1"])
def test_cut_off_or_damaged_bits_are_refused(bits):
    with pytest.raises(ValueError):
        unpack_words(bits)
