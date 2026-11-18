from llm_encryptor.words import (
    BITS_PER_BLOCK,
    WORDS_PER_BLOCK,
    block_bits,
    complete_word_ends,
    has_word,
    is_word_char,
    joins_cleanly,
    split_words,
    text_bits,
)


def test_words_ignore_case_punctuation_and_layout():
    assert split_words("Hello, World!\n\nIt's 2024_now") == ["hello", "world", "it", "s", "2024", "now"]


def test_words_ignore_unicode_form():
    assert split_words("café ﬁne") == split_words("café fine")


def test_words_ignore_accents_and_dotted_capitals():
    assert split_words("İstanbul’da ÇAY içtik") == ["istanbul", "da", "cay", "ictik"]
    assert split_words("café") == split_words("CAFE")


def test_curly_and_straight_quotes_give_the_same_words():
    assert split_words("don’t “stop”") == split_words("don't \"stop\"")


def test_complete_word_ends_skips_an_unfinished_last_word():
    assert complete_word_ends(" one two thr") == [4, 8]
    assert complete_word_ends(" one two.") == [4, 8]


def test_word_helpers():
    assert has_word(" ok ")
    assert not has_word(" ... \n")
    assert is_word_char("é") and is_word_char("7")
    assert not is_word_char("_") and not is_word_char("-")


def test_joins_cleanly():
    assert joins_cleanly("friend", " and")
    assert joins_cleanly("friend", ", and")
    assert joins_cleanly("friend.\n\n", "And")
    assert joins_cleanly("", "And")
    assert not joins_cleanly("friend", "ly")
    assert not joins_cleanly("friend", "")
    assert not joins_cleanly("cafe", "́ au lait")


def test_a_block_gives_the_agreed_number_of_bits(keys):
    bits = block_bits(keys.mac, ["one", "two", "three"])
    assert len(bits) == BITS_PER_BLOCK
    assert set(bits) <= {0, 1}
    assert bits == block_bits(keys.mac, ["one", "two", "three"])


def test_bits_depend_on_the_key_and_every_word(keys, other_keys):
    blocks = [[f"a{i}", "same", "words"] for i in range(40)]
    mine = [block_bits(keys.mac, block) for block in blocks]
    theirs = [block_bits(other_keys.mac, block) for block in blocks]
    assert mine != theirs
    assert len(set(mine)) == 2**BITS_PER_BLOCK


def test_text_bits_cover_whole_blocks_only(keys):
    words = [f"w{i}" for i in range(2 * WORDS_PER_BLOCK + 1)]
    bits = text_bits(keys.mac, words)
    assert len(bits) == 2 * BITS_PER_BLOCK
    assert tuple(bits[:BITS_PER_BLOCK]) == block_bits(keys.mac, words[:WORDS_PER_BLOCK])
    assert text_bits(keys.mac, words[: WORDS_PER_BLOCK - 1]) == []
