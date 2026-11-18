import random

import pytest
from fakes import FakeBackend

from llm_encryptor.backend import Choice
from llm_encryptor.crypto import DENSE_OVERHEAD, DenseBox
from llm_encryptor.dense import (
    LEVELS,
    OPENING_WORDS,
    WOBBLE,
    DenseError,
    _Interval,
    flatten,
    options,
    read_dense,
    split_opening,
    write_dense,
)
from llm_encryptor.embed import EmbedError, Progress
from llm_encryptor.session import Session

ASK = [{"role": "user", "content": "Write about the sea."}]


@pytest.mark.parametrize("secret", ["hi", "Meet me at the old bridge at 9pm.", "şifre: çay 🫖 123"])
def test_secret_round_trips(keys, secret):
    backend = FakeBackend(seed=1)
    text = write_dense(secret, keys, backend, ASK)
    assert read_dense(text, keys, backend) == secret


def test_wrong_password_or_model_recovers_nothing(keys, other_keys):
    text = write_dense("top secret", keys, FakeBackend(seed=2), ASK)
    with pytest.raises(DenseError):
        read_dense(text, other_keys, FakeBackend(seed=2))
    with pytest.raises(DenseError):
        read_dense(text, keys, FakeBackend(seed=3))


def test_spacing_and_line_breaks_may_change(keys):
    backend = FakeBackend(seed=4)
    text = write_dense("top secret", keys, backend, ASK)
    reflowed = "\n\n  " + text.replace(". ", ".\n\n").replace(" ", "  ", 5) + "  \n"
    assert read_dense(reflowed, keys, backend) == "top secret"


def test_text_after_the_message_is_ignored(keys):
    backend = FakeBackend(seed=5)
    text = write_dense("top secret", keys, backend, ASK)
    assert read_dense(text + " Zzz qqq, talk soon!", keys, backend) == "top secret"


def test_a_changed_word_recovers_nothing(keys):
    backend = FakeBackend(seed=6)
    words = write_dense("top secret", keys, backend, ASK).split(" ")
    words[OPENING_WORDS + 3] = "tampered"
    with pytest.raises(DenseError):
        read_dense(" ".join(words), keys, backend)


def test_too_short_a_text_is_refused(keys):
    with pytest.raises(DenseError, match="too short"):
        read_dense("Only a few words here.", keys, FakeBackend())


def test_progress_reaches_the_whole_payload(keys):
    seen: list[Progress] = []
    write_dense("hi", keys, FakeBackend(seed=7), ASK, on_progress=seen.append)
    assert seen[-1].embedded == seen[-1].total == 8 * (2 + DENSE_OVERHEAD)


def test_writer_tries_again_when_its_text_does_not_read_back(keys):
    class NoisyAtFirst(FakeBackend):
        """Shifts every probability during the first pass, as a busy server does."""

        passes = 0

        def reset(self):
            NoisyAtFirst.passes += 1

        def next_tokens(self, messages, prefix):
            choices = super().next_tokens(messages, prefix)
            if NoisyAtFirst.passes == 1:
                return [Choice(c.text, c.probability * (0.5 if i % 2 else 1.6)) for i, c in enumerate(choices)]
            return choices

    backend = NoisyAtFirst(seed=8)
    text = write_dense("top secret", keys, backend, ASK)
    assert NoisyAtFirst.passes >= 3
    assert read_dense(text, keys, backend) == "top secret"


def test_writer_gives_up_when_nothing_reads_back(keys):
    class NeverTheSame(FakeBackend):
        def next_tokens(self, messages, prefix):
            return [Choice(c.text, random.random()) for c in super().next_tokens(messages, prefix)]

    with pytest.raises(EmbedError, match="did not decode"):
        write_dense("top secret", keys, NeverTheSame(seed=9), ASK)


def test_silent_model_is_an_error(keys):
    class Silent(FakeBackend):
        def start(self, messages, pieces=8):
            return "Too short."

    with pytest.raises(EmbedError, match="did not start"):
        write_dense("hi", keys, Silent(), ASK)


def test_model_with_nothing_to_say_is_an_error(keys):
    class Stops(FakeBackend):
        def next_tokens(self, messages, prefix):
            return [Choice("", 1.0)]

    with pytest.raises(EmbedError, match="stopped before"):
        write_dense("hi", keys, Stops(), ASK)


def test_a_probability_near_a_level_is_read_both_ways(keys):
    """The reader sees one probability just across a boundary and still succeeds."""
    boundary = LEVELS[1][0]

    class Shifted(FakeBackend):
        shift = 0.0

        def next_tokens(self, messages, prefix):
            choices = super().next_tokens(messages, prefix)
            if len(prefix) % 29:
                return choices
            return [Choice(c.text, boundary + self.shift if c.probability == 0.25 else c.probability) for c in choices]

    writer, reader = Shifted(seed=10), Shifted(seed=10)
    writer.shift, reader.shift = boundary * WOBBLE / 3, -boundary * WOBBLE / 3
    text = write_dense("hi", keys, writer, ASK)
    assert read_dense(text, keys, reader) == "hi"


def test_compact_session_revises_and_is_shorter(keys):
    backend = FakeBackend(seed=11)
    compact = Session("meet at noon", keys, backend).instruct("Write about Turkey.")
    revised = compact.instruct("Make it friendlier.")
    robust = Session("meet at noon", keys, backend, compact=False).instruct("Write about Turkey.")
    assert read_dense(revised.cover, keys, backend) == "meet at noon"
    assert compact.cover != revised.cover
    assert len(compact.cover.split()) < len(robust.cover.split())
    assert compact.with_mode(False).compact is False
    assert compact.with_backend(FakeBackend(seed=12)).backend is not backend


# --- pieces ---------------------------------------------------------------


def test_opening_is_the_first_words():
    text = " ".join(f"w{i}" for i in range(40)) + "."
    opening, rest = split_opening("  " + text.replace(" ", "\n", 3))
    assert opening == " ".join(f"w{i}" for i in range(OPENING_WORDS))
    assert rest.startswith(f" w{OPENING_WORDS} ")


def test_flatten():
    assert flatten("a \n\n b\tc") == "a b c"


def test_options_are_sorted_and_cannot_be_confused():
    choices = [Choice(" inside", 0.3), Choice(" in", 0.3), Choice("\n\n", 0.3), Choice("", 0.3),
               Choice(" �", 0.3), Choice(" zero", 0.0), Choice(" at", 0.3)]
    found = options(choices, "We met")
    assert [option.token for option in found] == [" at", " in"]


def test_shares_follow_the_levels():
    choices = [Choice(" a", 0.7), Choice(" b", 0.2), Choice(" c", 0.07), Choice(" d", 0.01)]
    assert [option.share for option in options(choices, "x.")] == [16, 4, 2, 0]


def test_a_word_is_only_finished_the_likely_way():
    choices = [Choice("ly", 0.2), Choice("ship", 0.6), Choice(" and", 0.2)]
    shares = {option.token: option.share for option in options(choices, "friend")}
    assert shares == {" and": 4, "ly": 0, "ship": 16}
    assert [o.token for o in options(choices, "friend", first=True)] == [" and"]


def test_borderline_probabilities_are_flagged():
    edge = LEVELS[-1][0]
    choices = [Choice(" a", edge * (1 + WOBBLE / 2)), Choice(" b", edge * (1 - WOBBLE / 2)), Choice(" c", 0.9)]
    found = {option.token: (option.share, option.other) for option in options(choices, "x.")}
    assert found == {" a": (1, 0), " b": (0, 1), " c": (16, None)}


def test_the_likeliest_token_gets_through_when_none_stands_out():
    choices = [Choice(f" w{i}", 0.01 + i / 1000) for i in range(5)]
    assert [option.share for option in options(choices, "x.")] == [0, 0, 0, 0, 1]


def test_interval_bits():
    half = _Interval().narrowed(1, 1, 2)          # the upper half: 0.1...
    assert half.agreed_bits(8) == (1,)
    assert half.narrowed(0, 1, 4).agreed_bits(8) == (1, 0, 0)
    assert _Interval().narrowed(1, 2, 4).agreed_bits(8) == ()   # straddles one half


def test_dense_box(keys, other_keys):
    box = DenseBox(keys.dense)
    blob = box.seal(b"meet at noon", "Once upon a time")
    assert len(blob) == len(b"meet at noon") + DENSE_OVERHEAD
    assert box.open(blob, "Once upon a time") == b"meet at noon"
    assert box.open(blob, "Another opening") is None
    assert DenseBox(other_keys.dense).open(blob, "Once upon a time") is None
    assert box.open(blob[:DENSE_OVERHEAD], "Once upon a time") is None
    assert box.seal(b"same", "x") != box.seal(b"same", "x")
