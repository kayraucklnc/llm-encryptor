import math
import random
from dataclasses import replace

import pytest
from fakes import VOCABULARY, FakeBackend

from llm_encryptor import dense
from llm_encryptor.backend import Choice
from llm_encryptor.crypto import DENSE_OVERHEAD, LEAN_TAG_BITS, DenseBox
from llm_encryptor.dense import (
    MAX,
    NATURAL,
    NATURAL_OPENING,
    SHORT,
    TIGHT,
    WOBBLE,
    DenseError,
    _code,
    _Interval,
    _opening,
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
from llm_encryptor.textcode import pack, pack_bits
from llm_encryptor.wordcode import pack_words

ASK = [{"role": "user", "content": "Write about the sea."}]
# A request that names no subject, so any opening will do.
ANYTHING = [{"role": "user", "content": "Write a short note."}]


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
    compact = Session("meet at noon", keys, backend, density="natural").instruct("Write about Turkey.")
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


# --- short density and the leaner sealing ----------------------------------


@pytest.mark.parametrize("secret", ["hi", "Meet me at the old bridge at 9pm.", "şifre: çay 🫖 123"])
def test_short_density_round_trips_and_is_shorter(keys, secret):
    backend = FakeBackend(seed=31)
    short = write_dense(secret, keys, backend, ASK, density="short")
    assert read_dense(short, keys, backend) == secret
    assert len(short.split()) < len(write_dense(secret, keys, backend, ASK).split())


def test_short_density_ends_its_sentence_at_once(keys):
    backend = FakeBackend(seed=32)
    opening = _opening(backend, ASK, SHORT.opening_words)
    for seed in range(20):
        seen: list[Progress] = []
        text = _code("top secret", keys, backend, opening, seen.append, random.Random(seed), SHORT)
        assert text.endswith(".")
        # From the moment the message is in: at most a word after a comma, then the full stop.
        assert 1 <= len([p for p in seen if p.embedded == p.total]) <= 3


def test_short_and_max_carry_only_a_tag_besides_the_secret(keys):
    seen: list[Progress] = []
    write_dense("hi", keys, FakeBackend(seed=33), ASK, seen.append, density="max")
    assert seen[-1].total == len(pack_bits("hi")) + LEAN_TAG_BITS
    # Short packs the secret by the word and makes do with a shorter tag.
    write_dense("meet at noon", keys, FakeBackend(seed=33), ASK, seen.append, density="short")
    assert seen[-1].total == len(pack_words("meet at noon")) + SHORT.tag_bits == 1 + 3 * 13 + 20


def test_short_shares_are_measured_against_the_likeliest_token():
    choices = [Choice(" a", 0.6), Choice(" b", 0.05), Choice(" c", 0.03), Choice(" d", 0.2)]
    assert shares(options(choices, "x."), SHORT) == [1, 1, 0, 1]
    # Where the model has no favourite, everything it lists is fair game, down to a floor.
    choices = [Choice(" a", 0.1), Choice(" b", 0.016), Choice(" c", 0.012)]
    assert shares(options(choices, "x."), SHORT) == [1, 1, 0]
    pieces = options([Choice("ly", 0.2), Choice("ship", 0.3), Choice(" and", 0.5)], "friend")
    assert dict(zip((o.token for o in pieces), shares(pieces, SHORT))) == {" and": 1, "ly": 0, "ship": 1}


def test_short_density_does_not_leave_a_word_half_written():
    """Where the model wants to go on with the word, a comma may not cut the word off."""
    unfinished = options([Choice("its", 0.6), Choice(",", 0.2), Choice(" made", 0.1), Choice("ight", 0.4)], "the stra")
    assert dict(zip((o.token for o in unfinished), shares(unfinished, SHORT))) == {
        " made": 0, ",": 0, "ight": 1, "its": 1,
    }  # fmt: skip
    assert dict(zip((o.token for o in unfinished), shares(unfinished, TIGHT)))[","] == 1
    # A finished word may be followed by anything about as likely as extending it.
    done = options([Choice("ship", 0.3), Choice(" and", 0.5), Choice(",", 0.1)], "friend")
    assert dict(zip((o.token for o in done), shares(done, SHORT))) == {" and": 1, ",": 0, "ship": 1}
    # The rule has no sharp edge: a token near it is one the reader reads both ways.
    near = options([Choice("its", 0.6), Choice(",", 0.3 * (1 + WOBBLE))], "the stra")
    assert shares(near, SHORT) == [1, 1] and _other_share(near[0], SHORT, 0.6, 0.6) == 0


class NewOpenings(FakeBackend):
    """Starts every text differently, as a real model does."""

    def __init__(self, seed: int = 0) -> None:
        super().__init__(seed)
        self.openings = 0

    def start(self, messages, pieces=8):
        self.openings += 1
        words = random.Random(self.openings).choices(VOCABULARY, k=pieces)
        return " ".join(words).capitalize()


def test_short_density_keeps_the_likeliest_of_its_drafts(keys, monkeypatch):
    real, written = dense._draft, []
    surprises = iter([5.0, 3.0, 9.0, 1.0, 7.0, 8.0])

    def scored(*args):
        _, text = real(*args)
        written.append(text)
        return next(surprises), text

    monkeypatch.setattr(dense, "_draft", scored)
    backend = NewOpenings(seed=40)
    text = write_dense("top secret", keys, backend, ANYTHING, density="short")
    assert backend.openings == SHORT.drafts == 6 and len(set(written)) == 6
    assert text == written[3]


def test_the_model_finds_an_ordinary_text_less_surprising(keys):
    backend = FakeBackend(seed=41)
    opening = _opening(backend, ASK, SHORT.opening_words)
    surprise, text = dense._draft("top secret", keys, backend, opening, None, random.Random(1), SHORT)
    _, steps, to_the_end = dense._replay(text, backend, SHORT.opening_words, dense.CHAT_CONTINUE, True)
    assert to_the_end
    assert surprise == pytest.approx(sum(-math.log2(found[at].probability) for found, at in steps))


def test_a_slow_model_writes_fewer_drafts(keys):
    backend = NewOpenings(seed=42)
    ticks = iter(range(0, 10_000, 20))   # every look at the clock is 20 seconds later
    text = write_dense("top secret", keys, backend, ANYTHING, density="short", clock=lambda: next(ticks))
    assert backend.openings == 1 and read_dense(text, keys, backend) == "top secret"


def test_short_density_is_written_and_read_as_a_text_message(keys):
    class Listening(FakeBackend):
        def __init__(self, seed):
            super().__init__(seed)
            self.started, self.continued = [], []

        def start(self, messages, pieces=8):
            self.started.append(messages[0]["content"])
            return super().start(messages, pieces)

        def next_tokens(self, messages, prefix, attempt=0):
            self.continued.append(messages[0]["content"])
            return super().next_tokens(messages, prefix, attempt)

    backend = Listening(seed=43)
    text = write_dense("top secret", keys, backend, ASK, density="short")
    assert all("text message to a friend" in prompt for prompt in backend.started)
    assert all("texting a friend" in prompt for prompt in backend.continued)
    backend.continued.clear()
    backend.resets = 0
    assert read_dense(text, keys, backend) == "top secret"
    # A brief text is read as a short one first: one pass, with the same instructions.
    assert backend.resets == 1 and all("texting a friend" in prompt for prompt in backend.continued)
    # A longer text is read as the usual prose first.
    long_secret = "a longer secret, so that the text is clearly not a brief one"
    essay = write_dense(long_secret, keys, backend, ASK)
    backend.continued.clear()
    backend.resets = 0
    assert len(essay.split()) > 80 and read_dense(essay, keys, backend) == long_secret
    assert backend.resets == 1 and all("middle of a long piece" in prompt for prompt in backend.continued)


def test_short_density_keeps_to_plain_characters():
    choices = [
        Choice(" [", 0.3), Choice(" (", 0.3), Choice(" 🙂", 0.3), Choice(' "', 0.3), Choice(" #", 0.3),
        Choice(" at", 0.3), Choice(" 7", 0.3), Choice(",", 0.3), Choice(" don’t", 0.3), Choice(" çay", 0.3),
    ]  # fmt: skip
    assert {o.token for o in options(choices, "See you.", plain=True)} == {" at", " 7", ",", " don’t", " çay"}
    assert len(options(choices, "See you.")) == len(choices)
    # No apostrophe left hanging ("don' t"): on its own it would also hide "'t" from the choices.
    apostrophes = [Choice("'", 0.3), Choice("’", 0.3), Choice("'t", 0.3), Choice(" and", 0.3)]
    assert {o.token for o in options(apostrophes, "We don", plain=True)} == {"'t", " and"}
    assert "'t" not in {o.token for o in options(apostrophes, "We don")}
    # No capital letter in the middle of a word either.
    pieces = [Choice("ATES", 0.4), Choice("ates", 0.4), Choice(" Ankara", 0.2)]
    assert {o.token for o in options(pieces, "captiv", plain=True)} == {"ates", " Ankara"}
    # No sentence of one or two words: a run of those reads as noise.
    stops = [Choice(".", 0.3), Choice("?", 0.3), Choice(" tonight", 0.3)]
    assert {o.token for o in options(stops, "See you there. Have fun", plain=True)} == {" tonight"}
    assert {o.token for o in options(stops, "See you there. Have fun then", plain=True)} == {".", "?", " tonight"}
    # With nothing plain on offer there is nothing to choose from.
    assert options([Choice(" [", 0.6), Choice(" (", 0.4)], "x.", plain=True) == []
    # What is not plain does not set the bar for what is.
    crowded = options([Choice(" 🙂", 0.9), Choice(" at", 0.05), Choice(" on", 0.04)], "See you.", plain=True)
    assert dict(zip((o.token for o in crowded), shares(crowded, SHORT))) == {" at": 1, " on": 1}


def test_short_texts_carry_no_markup_even_from_a_model_that_offers_it(keys):
    class Decorating(FakeBackend):
        def next_tokens(self, messages, prefix, attempt=0):
            choices = super().next_tokens(messages, prefix, attempt)
            return [Choice(" [name]", 0.5), Choice(" 🙂", 0.4), *choices]

    backend = Decorating(seed=46)
    text = write_dense("top secret", keys, backend, ASK, density="short")
    assert "[" not in text and "🙂" not in text and read_dense(text, keys, backend) == "top secret"
    assert "[" in write_dense("top secret", keys, backend, ASK)


def test_the_opening_is_asked_for_again_until_it_names_the_subject(keys):
    class Scripted(FakeBackend):
        def __init__(self, *replies):
            super().__init__()
            self.replies, self.asked = list(replies), 0

        def start(self, messages, pieces=8):
            self.asked += 1
            return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]

    subject = dense._subject([
        {"role": "user", "content": "Write a friendly note about dinner on Friday"},
        {"role": "assistant", "content": "an earlier text"},
        {"role": "user", "content": "Make it shorter please."},
    ])
    assert subject == {"dinner", "friday", "shorter"}
    backend = Scripted("Hey, how is it going today", "Hey, what about the dinners we planned")
    assert _opening(backend, ASK, 5, True, subject) == "Hey, what about the dinners" and backend.asked == 2
    # A start on the subject that is not plain loses to a plain one off the subject...
    backend = Scripted("Dinner [time] is at seven sharp", "Hey, how is it going today")
    assert _opening(backend, ASK, 5, True, subject) == "Hey, how is it going"
    # ...and with no subject to look for, the first plain start is taken.
    backend = Scripted("Hey, how is it going today", "Dinner is at seven sharp then")
    assert _opening(backend, ASK, 5, True) == "Hey, how is it going" and backend.asked == 1
    assert dense._subject([{"role": "user", "content": "write a short note"}]) == frozenset()


def test_a_short_text_asked_for_again_starts_differently(keys):
    """Sealed without random bytes, the same opening would give the same text again."""
    backend = NewOpenings(seed=47)
    first = Session("top secret", keys, backend, density="short").instruct("Write a short note.")
    revised = first.instruct("Make it shorter please.")
    assert read_dense(revised.cover, keys, backend) == "top secret"
    assert revised.cover.split()[:5] != first.cover.split()[:5]
    # A start like an earlier text's is passed over.
    twice = NewOpenings(seed=47)
    twice.start = lambda messages, pieces=8: "Sea calm bright while old ships moved slowly past green hills."
    earlier = ["Sea calm bright while old ships."]
    assert _opening(twice, ANYTHING, 5, True, frozenset(), earlier) == "Sea calm bright while old"   # nothing better
    assert _opening(NewOpenings(seed=47), ANYTHING, 5, True, frozenset(), [first.cover]) != " ".join(first.cover.split()[:5])


def test_short_texts_open_on_the_subject_asked_for(keys):
    class TwoStarts(FakeBackend):
        asked = 0

        def start(self, messages, pieces=8):
            TwoStarts.asked += 1
            start = "Hey how is it going today" if TwoStarts.asked % 2 else "The sea was calm and bright"
            return start + " while old ships moved slowly past green hills and people walked along narrow streets."

    text = write_dense("hi", keys, TwoStarts(seed=45), ASK, density="short")
    assert text.startswith("The sea was calm and")
    # Only the short density looks for the subject: longer openings get to it anyway.
    TwoStarts.asked = 0
    assert write_dense("hi", keys, TwoStarts(seed=45), ASK).startswith("Hey how is it going today")


def test_a_marked_up_opening_is_asked_for_again(keys):
    class Decorated(NewOpenings):
        def start(self, messages, pieces=8):
            plain = super().start(messages, pieces)
            return "Hey [Friend's name], hope you are doing really well" if self.openings == 1 else plain

    backend = Decorated(seed=44)
    assert "[" not in _opening(backend, ASK, SHORT.opening_words, plain=True) and backend.openings == 2
    assert "[" in _opening(Decorated(seed=44), ASK, SHORT.opening_words)

    class AlwaysDecorated(FakeBackend):
        def start(self, messages, pieces=8):
            return "Hey [Friend's name], hope you are doing really well"

    assert _opening(AlwaysDecorated(), ASK, 5, plain=True) == "Hey [Friend's name], hope"


def test_a_probability_near_a_relative_level_is_read_both_ways(keys):
    boundary = SHORT.levels[0][0] * 0.4  # the toy model's likeliest token has 0.4

    class Shifted(FakeBackend):
        shift = 0.0

        def next_tokens(self, messages, prefix, attempt=0):
            choices = super().next_tokens(messages, prefix, attempt)
            if len(prefix.split()) != SHORT.opening_words + 2:
                return choices
            return [Choice(c.text, boundary + self.shift if c.probability == 0.06 else c.probability) for c in choices]

    writer, reader = Shifted(seed=34), Shifted(seed=34)
    writer.shift, reader.shift = boundary * WOBBLE / 3, -boundary * WOBBLE / 3
    text = write_dense("hi", keys, writer, ASK, density="short")
    assert read_dense(text, keys, reader) == "hi"
    edge = [Choice(" a", 0.4), Choice(" b", boundary * (1 + WOBBLE)), Choice(" c", boundary * (1 - WOBBLE))]
    assert [_other_share(option, SHORT, 0.4) for option in options(edge, "x.")] == [None, 0, 1]


def test_nothing_may_follow_a_short_text(keys):
    """A short text is the model's to its last word: that tells it from other brief texts."""
    backend = FakeBackend(seed=48)
    text = write_dense("top secret", keys, backend, ASK, density="short")
    assert read_dense(text, keys, backend) == "top secret"
    with pytest.raises(DenseError):
        read_dense(text + " Zzz qqq, talk soon!", keys, backend)
    # A brief text of another density is not taken for a short one, and is still found.
    brief = write_dense("hi", keys, backend, ASK, density="max")
    assert len(brief.split()) < 80 and read_dense(brief, keys, backend) == "hi"
    assert read_dense(brief + " Zzz qqq, talk soon!", keys, backend) == "hi"


def test_wrong_password_recovers_nothing_from_a_short_text(keys, other_keys):
    text = write_dense("top secret", keys, FakeBackend(seed=35), ASK, density="short")
    with pytest.raises(DenseError):
        read_dense(text, other_keys, FakeBackend(seed=35))


def test_max_texts_from_before_the_leaner_sealing_still_read(keys):
    backend = FakeBackend(seed=36)
    opening = _opening(backend, ASK, MAX.opening_words)
    old = _code("top secret", keys, backend, opening, None, random.Random(1), replace(MAX, lean=False))
    assert read_dense(old, keys, backend) == "top secret"
    new = _code("top secret", keys, backend, opening, None, random.Random(1), MAX)
    assert read_dense(new, keys, backend) == "top secret"
    assert len(new.split()) < len(old.split())
