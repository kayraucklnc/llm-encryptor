import pytest

from llm_encryptor.textcode import _CODES, _FILLER, pack, unpack

MESSAGES = [
    "meet at noon",
    "a",
    "",
    "The eagle lands at dawn, bring 2 keys!",
    "çay at 5, İstiklal",
    "日本語のメッセージ",
    "emoji 🫖 and tabs\t\x00 too",
    "line one\nline two",
    "x" * 300,
]


@pytest.mark.parametrize("message", MESSAGES)
def test_round_trip(message):
    assert unpack(pack(message)) == message


def test_everyday_text_takes_about_half_the_space():
    assert len(pack("meet at noon")) <= 7
    assert len(pack("the eagle lands at dawn")) <= 13


def test_text_it_cannot_shrink_costs_one_byte_more():
    message = "日本語のメッセージ"
    assert len(pack(message)) == len(message.encode("utf-8")) + 1


def test_the_code_is_prefix_free_and_the_filler_is_safe():
    codes = sorted(_CODES.values())
    assert not any(codes[i + 1].startswith(codes[i]) for i in range(len(codes) - 1))
    assert len(_FILLER) >= 8


def test_the_table_does_not_change():
    """Texts already sent depend on these exact codes."""
    assert _CODES[" "] == "110" and _CODES["e"] == "1111"
    assert pack("meet at noon").hex() == _GOLDEN


@pytest.mark.parametrize("damaged", [b"", b"\x90abc", b"\x80\xff"])
def test_damaged_data_is_refused(damaged):
    with pytest.raises(ValueError):
        unpack(damaged)


_GOLDEN = "5bff59eb22aa33"
