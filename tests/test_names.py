import random
import re

import pytest
from fakes import FakeBackend

from llm_encryptor.backend import Choice
from llm_encryptor.codec import RevealError
from llm_encryptor.crypto import derive_keys
from llm_encryptor.embed import Progress
from llm_encryptor.hide import hide
from llm_encryptor.names import (
    LEADS,
    MIN_TAG_BITS,
    NAME_BITS,
    NAMES,
    PLAIN_LEAD,
    names_for,
    read_names,
    reveal_offline,
    write_names,
)
from llm_encryptor.session import Session
from llm_encryptor.wordcode import pack_words
from llm_encryptor.words import split_words

ASK = [{"role": "user", "content": "a note about the weekend"}]


class Completes(FakeBackend):
    """A model that finishes any opening line with scripted words, one token at a time."""

    def __init__(self, *tails: str) -> None:
        super().__init__()
        self.tails = list(tails)
        self.tail = ""
        self.asked: list[list[dict]] = []
        self.leads: list[str] = []

    def next_tokens(self, messages, prefix, attempt=0):
        lead = next(lead for lead in LEADS if prefix.startswith(lead))
        written = prefix[len(lead) :]
        if not written:
            self.tail = self.tails.pop(0) if self.tails else " the party:"
            self.asked.append(list(messages))
            self.leads.append(lead)
        token = re.match(r"\s*\S+", self.tail[len(written) :])
        return [Choice(token.group(), 0.9), Choice(" other", 0.1)] if token else [Choice("", 1.0)]


def test_the_name_list_is_fixed():
    """The list is part of the format: its size and order must never change."""
    assert len(NAMES) == len({name.casefold() for name in NAMES}) == 2**NAME_BITS
    assert list(NAMES) == sorted(NAMES, key=str.casefold)
    assert NAMES[0] == "Aaron" and "Kayra" in NAMES and "Will" not in NAMES and "Rose" not in NAMES
    assert all(name.isascii() and name.isalpha() and name[0].isupper() for name in NAMES)
    # The reader finds each name as one word, and none hides in the lines the app writes itself.
    assert all(split_words(name) == [name.casefold()] for name in NAMES)
    listed = {name.casefold() for name in NAMES}
    assert not listed & set(split_words(" ".join([*LEADS, PLAIN_LEAD, "and"])))


@pytest.mark.parametrize("secret", ["hi", "meet at noon", "Meet me at the old bridge at 9pm.", "şifre: çay 🫖 123"])
def test_secret_round_trips(keys, secret):
    text = write_names(secret, keys, Completes(" dinner on Saturday:"), ASK)
    assert read_names(text, keys) == secret


def test_twelve_characters_take_six_names(keys):
    names = names_for("meet at noon", keys)
    assert len(names) == 6 and all(name in NAMES for name in names)
    backend = Completes(" dinner:")
    text = write_names("meet at noon", keys, backend, ASK)
    assert text == f"{backend.leads[0]} dinner: " + ", ".join(names[:-1]) + f" and {names[-1]}."
    assert len(text.split()) <= 13


def test_every_bit_is_used_and_the_tag_fills_the_last_name(keys):
    for secret in ("a", "hi", "meet at noon", "the package is under the bench"):
        spare = NAME_BITS * len(names_for(secret, keys)) - len(pack_words(secret))
        assert MIN_TAG_BITS <= spare < MIN_TAG_BITS + NAME_BITS


def test_reading_needs_no_model_and_survives_reformatting(keys):
    text = write_names("meet at noon", keys, Completes(), ASK)
    mangled = text.upper().replace(", ", "\n- ").replace(" and ", " & ")
    assert read_names(mangled, keys) == "meet at noon"


def test_wrong_password_or_changed_names_recover_nothing(keys, other_keys):
    names = names_for("meet at noon", keys)
    text = write_names("meet at noon", keys, Completes(), ASK)
    with pytest.raises(RevealError):
        read_names(text, other_keys)
    other = next(name for name in NAMES if name not in names)
    for changed in (text.replace(names[2], other), text.replace(f"{names[1]}, ", ""), text + f" {other}"):
        with pytest.raises(RevealError):
            read_names(changed, keys)
    with pytest.raises(RevealError):
        read_names("No names in here at all.", keys)


def test_the_same_secret_gives_the_same_names_and_another_secret_other_names(keys):
    assert names_for("meet at noon", keys) == names_for("meet at noon", keys)
    assert names_for("meet at noon", keys) != names_for("meet at nine", keys)
    assert names_for("meet at noon", keys) != names_for("meet at noon", derive_keys("another"))


def test_the_model_finishes_the_line_before_the_names(keys):
    backend = Completes(" the trip to Turkey:\nAnna, Bob")
    text = write_names("hi", keys, backend, [{"role": "user", "content": "a trip to Turkey"}])
    assert text.startswith(f"{backend.leads[0]} the trip to Turkey: ")
    assert backend.asked[0][-1] == {"role": "user", "content": "a trip to Turkey"}
    assert "Never write a name" in backend.asked[0][0]["content"]


def test_the_line_starts_differently_from_one_text_to_the_next(keys):
    starts = set()
    for seed in range(20):
        backend = Completes()
        write_names("hi", keys, backend, ASK, rng=random.Random(seed))
        starts.add(backend.leads[0])
    assert len(starts) > 1 and starts <= set(LEADS)


@pytest.mark.parametrize(
    "bad",
    [
        f" dinner with {NAMES[7]}:",              # a name of the list would be read as part of the message
        " the [occasion]:",                      # the model copied its instructions
        " the user's request:",
        ":",                                     # nothing added
        " a very long evening that just keeps going and going:",
        " something that never comes to an end " * 3,
    ],
)
def test_an_unusable_line_is_tried_again_with_another_start(keys, bad):
    backend = Completes(bad, " dinner:")
    text = write_names("hi", keys, backend, ASK)
    assert text.startswith(f"{backend.leads[1]} dinner: ") and backend.leads[0] != backend.leads[1]


def test_a_line_may_end_in_a_full_stop_and_gets_its_colon(keys):
    backend = Completes(" dinner on Friday. See you")
    assert write_names("hi", keys, backend, ASK).startswith(f"{backend.leads[0]} dinner on Friday: ")


def test_a_model_that_never_gives_a_usable_line_gets_a_plain_one(keys):
    for backend in (Completes(*[f" dinner with {NAMES[7]}:"] * 5), FakeBackend(seed=2)):
        text = write_names("hi", keys, backend, ASK)
        assert text.startswith(f"{PLAIN_LEAD} ") and read_names(text, keys) == "hi"


def test_progress_is_reported_once_the_text_is_there(keys):
    seen: list[Progress] = []
    text = write_names("hi", keys, Completes(), ASK, seen.append)
    assert seen[-1].text == text and seen[-1].embedded == seen[-1].total > 0


def test_reveal_offline_reads_names_and_the_robust_format(keys):
    assert reveal_offline(write_names("meet at noon", keys, Completes(), ASK), keys) == "meet at noon"
    robust = hide("hi", keys, FakeBackend(seed=3), [{"role": "user", "content": "Write about the sea."}])
    assert reveal_offline(robust, keys) == "hi"
    with pytest.raises(RevealError):
        reveal_offline("Nothing here, Anna.", keys)


def test_names_session_revises_the_line_and_keeps_the_names(keys):
    backend = Completes(" Saturday:", " this lovely Saturday:")
    first = Session("meet at noon", keys, backend, names=True).instruct("a note about Saturday")
    second = first.instruct("Make it friendlier.")
    assert read_names(second.cover, keys) == "meet at noon"
    assert first.cover != second.cover
    assert first.cover.split(": ")[1] == second.cover.split(": ")[1]
    assert backend.asked[1][-1] == {"role": "user", "content": "Make it friendlier."}
