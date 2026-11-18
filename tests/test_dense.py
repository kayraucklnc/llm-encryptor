import random

import pytest
from fakes import FakeBackend

from llm_encryptor.backend import Choice
from llm_encryptor.crypto import DENSE_OVERHEAD, DenseBox
from llm_encryptor.dense import (
    MAX,
    NATURAL,
    NATURAL_OPENING,
    TIGHT,
    WOBBLE,
    DenseError,
    _Interval,
    _other_share,
    flatten,
    options,
    read_dense,
    shares,
    split_opening,
    write_dense,
)
from llm_encryptor.embed import EmbedError, Progress
from llm_encryptor.session import Session
from llm_encryptor.textcode import pack

ASK = [{"role": "user", "content": "Write about the sea."}]


@pytest.mark.parametrize("secret", ["hi", "Meet me at the old bridge at 9pm.", "şifre: çay 🫖 123"])
def test_secret_round_trips(keys, secret):
    backend = FakeBackend(seed=1)
    text = write_dense(secret, keys, backend, ASK)
    assert read_dense(text, keys, backend) == secret


def test_tight_texts_read_back_without_being_told(keys):
    backend = FakeBackend(seed=21)
    secret = "a longer secret, so that the difference shows clearly"
    tight = write_dense(secret, keys, backend, ASK, density="tight")
    assert read_dense(tight, keys, backend) == secret


def test_max_density_uses_every_token_and_stops_at_once(keys):
    backend = FakeBackend(seed=22)
    secret = "a longer secret, so that the difference shows clearly"
    natural = write_dense(secret, keys, backend, ASK)
    shortest = write_dense(secret, keys, backend, ASK, density="max")
    assert read_dense(shortest, keys, backend) == secret
    assert len(shortest.split()) < len(natural.split())
    choices = [Choice(" a", 0.9), Choice(" b", 0.05), Choice(" c", 0.001), Choice(" d", 0.0001)]
    assert shares(options(choices, "x."), MAX) == [1, 1, 1, 0]
    pieces = options([Choice("ly", 0.01), Choice(" and", 0.9)], "friend")
    assert shares(pieces, MAX) == [1, 1] and shares(pieces, NATURAL) == [16, 0]
    choices = [Choice(" a", 0.7), Choice(" b", 0.2), Choice(" c", 0.02), Choice(" d", 0.005)]
    assert shares(options(choices, "x."), TIGHT) == [2, 1, 1, 0]


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
    words[NATURAL_OPENING + 3] = "tampered"
    with pytest.raises(DenseError):
        read_dense(" ".join(words), keys, backend)


def test_too_short_a_text_is_refused(keys):
    with pytest.raises(DenseError, match="too short"):
        read_dense("Far too short.", keys, FakeBackend())


def test_progress_reaches_the_whole_payload(keys):
    seen: list[Progress] = []
    write_dense("hi", keys, FakeBackend(seed=7), ASK, on_progress=seen.append)
    assert seen[-1].embedded == seen[-1].total == 8 * (len(pack("hi")) + DENSE_OVERHEAD)


def test_writer_tries_again_when_its_text_does_not_read_back(keys):
    class NoisyAtFirst(FakeBackend):
        """Shifts every probability during the first pass, as a busy server does."""

        passes = 0

        def reset(self):
            NoisyAtFirst.passes += 1

        def next_tokens(self, messages, prefix, attempt=0):
            choices = super().next_tokens(messages, prefix, attempt)
            if NoisyAtFirst.passes == 1:
                return [Choice(c.text, c.probability * (0.5 if i % 2 else 1.6)) for i, c in enumerate(choices)]
            return choices

    backend = NoisyAtFirst(seed=8)
    text = write_dense("top secret", keys, backend, ASK)
    assert NoisyAtFirst.passes >= 3
    assert read_dense(text, keys, backend) == "top secret"


def test_writer_gives_up_when_nothing_reads_back(keys):
    class NeverTheSame(FakeBackend):
        def next_tokens(self, messages, prefix, attempt=0):
            return [Choice(c.text, random.random()) for c in super().next_tokens(messages, prefix, attempt)]

    with pytest.raises(EmbedError, match="did not decode"):
        write_dense("top secret", keys, NeverTheSame(seed=9), ASK)


def test_silent_model_is_an_error(keys):
    class Silent(FakeBackend):
        def start(self, messages, pieces=8):
            return "Too short."

    with pytest.raises(EmbedError, match="very short reply"):
        write_dense("hi", keys, Silent(), ASK)


def test_a_too_short_reply_is_asked_for_again(keys):
    asked = []

    class ShortAtFirst(FakeBackend):
        def start(self, messages, pieces=8):
            asked.append(list(messages))
            return "Hello." if len(asked) == 1 else super().start(messages, pieces)

    backend = ShortAtFirst(seed=3)
    text = write_dense("hi", keys, backend, ASK)
    assert read_dense(text, keys, backend) == "hi"
    retry = asked[1]
    assert retry[-2] == {"role": "assistant", "content": "Hello."}
    assert "too short" in retry[-1]["content"]


def test_a_model_that_wants_to_stop_is_asked_what_else_it_could_write(keys):
    class StopsOften(FakeBackend):
        """Refuses to list tokens on the first two attempts at every third step."""

        def next_tokens(self, messages, prefix, attempt=0):
            if len(prefix) % 3 == 0 and attempt < 2:
                return [Choice("", 1.0)]
            return super().next_tokens(messages, prefix, attempt)

    backend = StopsOften(seed=13)
    text = write_dense("top secret", keys, backend, ASK)
    assert read_dense(text, keys, backend) == "top secret"


def test_model_with_nothing_to_say_is_an_error(keys):
    class Stops(FakeBackend):
        def next_tokens(self, messages, prefix, attempt=0):
            return [Choice("", 1.0)]

    with pytest.raises(EmbedError, match="stopped before"):
        write_dense("hi", keys, Stops(), ASK)


def test_a_probability_near_a_level_is_read_both_ways(keys):
    """The reader sees one probability just across a boundary and still succeeds."""
    boundary = NATURAL.levels[1][0]

    class Shifted(FakeBackend):
        shift = 0.0

        def next_tokens(self, messages, prefix, attempt=0):
            choices = super().next_tokens(messages, prefix, attempt)
            if len(prefix.split()) != NATURAL_OPENING + 2:
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


# --- pieces ---------------------------------------------------------------


def test_opening_is_the_first_words():
    text = " ".join(f"w{i}" for i in range(40)) + "."
    opening, rest = split_opening("  " + text.replace(" ", "\n", 3))
    assert opening == " ".join(f"w{i}" for i in range(NATURAL_OPENING))
    assert rest.startswith(f" w{NATURAL_OPENING} ")


def test_flatten():
    assert flatten("a \n\n b\tc") == "a b c"


def test_options_are_sorted_and_cannot_be_confused():
    choices = [Choice(" inside", 0.3), Choice(" in", 0.3), Choice("\n\n", 0.3), Choice("", 0.3),
               Choice(" �", 0.3), Choice(" zero", 0.0), Choice(" at", 0.3)]
    found = options(choices, "We met")
    assert [option.token for option in found] == [" at", " in"]


def test_words_already_said_in_that_order_are_not_offered_again():
    written = "We were part of a group, who were part of a"
    choices = [Choice(" group", 0.9), Choice(" team", 0.05), Choice(",", 0.05)]
    assert [o.token for o in options(choices, written)] == [" team", ","]
    # ...also when the repeat is completed by a word piece
    assert [o.token for o in options([Choice("oup", 0.9), Choice("ove", 0.1)], written + " gr")] == ["ove"]
    # ...but with nothing else to say, the repeat is allowed rather than getting stuck
    assert [o.token for o in options([Choice(" group", 1.0)], written)] == [" group"]
    assert [o.token for o in options([Choice(" who", 1.0), Choice(" x", 0.1)], "who, who, who,")] == [" x"]


def test_a_text_going_nowhere_is_started_again(keys):
    class Looping(FakeBackend):
        """Until its second opening, offers a single choice at every step: no bits fit."""

        openings = 0

        def start(self, messages, pieces=8):
            Looping.openings += 1
            return super().start(messages, pieces)

        def next_tokens(self, messages, prefix, attempt=0):
            choices = super().next_tokens(messages, prefix, attempt)
            return choices[:1] if Looping.openings < 2 else choices

    backend = Looping(seed=14)
    text = write_dense("top secret", keys, backend, ASK)
    assert Looping.openings == 2
    assert read_dense(text, keys, backend) == "top secret"


def test_shares_follow_the_levels():
    choices = [Choice(" a", 0.7), Choice(" b", 0.2), Choice(" c", 0.07), Choice(" d", 0.01)]
    assert shares(options(choices, "x.")) == [16, 4, 2, 0]


def test_a_word_is_only_finished_the_likely_way():
    choices = [Choice("ly", 0.2), Choice("ship", 0.6), Choice(" and", 0.2)]
    found = options(choices, "friend")
    assert dict(zip((o.token for o in found), shares(found))) == {" and": 4, "ly": 0, "ship": 16}
    assert [o.token for o in options(choices, "friend", first=True)] == [" and"]


def test_borderline_probabilities_are_flagged():
    edge = NATURAL.levels[-1][0]
    choices = [Choice(" a", edge * (1 + WOBBLE / 2)), Choice(" b", edge * (1 - WOBBLE / 2)), Choice(" c", 0.9)]
    found = options(choices, "x.")
    assert shares(found) == [1, 0, 16]
    assert [_other_share(option) for option in found] == [0, 1, None]


def test_the_likeliest_token_gets_through_when_none_stands_out():
    choices = [Choice(f" w{i}", 0.01 + i / 1000) for i in range(5)]
    assert shares(options(choices, "x.")) == [0, 0, 0, 0, 1]


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
