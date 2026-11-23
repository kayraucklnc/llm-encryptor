"""The names format: the secret picks people's names, and the reader needs no model.

A list of names is about the only ordinary text in which every word may be
anything: nobody can say that a guest list should have held Marco instead of
Yuki. So the encrypted secret is cut into pieces of ten bits and each piece
picks one of 1024 first names. The model only finishes the line that
introduces the list. Reading is a table lookup: find the names in the text, turn them
back into bits, decrypt.

That makes for the shortest texts by far (six names for a 12-character
secret), and they survive anything that leaves the names spelled as they
were. The price: what the text says is always "here are some people".

The name list (data/names.txt) is part of the format: its order must never
change for texts already sent.
"""

from __future__ import annotations

import random
import re
from collections.abc import Sequence
from importlib import resources

from .backend import Backend, Message
from .codec import RevealError, reveal_with_keys
from .crypto import Keys, LeanBox
from .embed import EmbedError, Progress, ProgressCallback
from .wordcode import pack_words, unpack_words
from .words import split_words

NAMES: tuple[str, ...] = tuple(
    resources.files(__package__).joinpath("data/names.txt").read_text(encoding="utf-8").split()
)
NAME_BITS = 10
# The tag takes at least this many bits, and then whatever is left of the last name.
MIN_TAG_BITS = 16
_INDEX = {name.casefold(): at for at, name in enumerate(NAMES)}

LEAD_PROMPT = (
    "You complete the opening line of a short message that introduces a list of people. "
    "Continue the line with a few plain words that name the occasion the user mentions, "
    "then end the line with a colon. Never write a name."
)
# The line starts with one of these, so that it makes sense whatever the model adds.
LEADS = (
    "Here's who is coming to",
    "Who's coming to",
    "Everyone who is in for",
    "People joining us for",
    "The group for",
)
PLAIN_LEAD = "Here is who is coming:"
_LEAD_TOKENS = 12
_LEAD_WORDS = 12
_LINE_END = re.compile(r"[:.!?\n]")
# Left in by a model that copied its instructions instead of following them.
_NOT_A_LINE = re.compile(r"[\[\]<>{}|*_#\"]|\b(user|occasion|request)\b", re.IGNORECASE)


def names_for(secret: str, keys: Keys) -> list[str]:
    """The names that carry ``secret``, in order."""
    message = pack_words(secret)
    tag_bits = MIN_TAG_BITS + -(len(message) + MIN_TAG_BITS) % NAME_BITS
    sealed = _box(keys, tag_bits).seal(message, _label(tag_bits))
    return [NAMES[int(sealed[at : at + NAME_BITS], 2)] for at in range(0, len(sealed), NAME_BITS)]


def read_names(text: str, keys: Keys) -> str:
    """Recover the secret from the names in a text."""
    bits = "".join(format(_INDEX[word], f"0{NAME_BITS}b") for word in split_words(text) if word in _INDEX)
    # The writer made the tag as long as it took to fill the last name.
    for tag_bits in range(MIN_TAG_BITS, MIN_TAG_BITS + NAME_BITS):
        message = _box(keys, tag_bits).open_whole(bits, _label(tag_bits))
        if message is not None:
            try:
                return unpack_words(message)
            except ValueError:
                continue
    raise RevealError(
        "No hidden message found in the names. The password may be wrong, "
        "or a name was changed, added or removed."
    )


def write_names(
    secret: str,
    keys: Keys,
    backend: Backend,
    conversation: Sequence[Message],
    on_progress: ProgressCallback | None = None,
    rng: random.Random | None = None,
) -> str:
    """A line written to ``conversation``, then the names that carry ``secret``."""
    names = names_for(secret, keys)
    lead = _lead_in(backend, conversation, rng or random.SystemRandom())
    text = f"{lead} {_listed(names)}."
    try:
        intact = read_names(text, keys) == secret
    except RevealError:
        intact = False
    if not intact:
        raise EmbedError("The finished text did not decode back to the secret. Please try again.")
    if on_progress is not None:
        on_progress(Progress(text, NAME_BITS * len(names), NAME_BITS * len(names)))
    return text


def reveal_offline(text: str, keys: Keys) -> str:
    """Read a text in either format that needs no model: names, then robust."""
    try:
        return read_names(text, keys)
    except RevealError:
        return reveal_with_keys(text, keys)


def _box(keys: Keys, tag_bits: int) -> LeanBox:
    return LeanBox(keys.dense, tag_bits)


def _label(tag_bits: int) -> str:
    return f"names/v1/{tag_bits}"


def _listed(names: Sequence[str]) -> str:
    if len(names) < 2:
        return "".join(names)
    return ", ".join(names[:-1]) + f" and {names[-1]}"


def _lead_in(backend: Backend, conversation: Sequence[Message], rng: random.Random) -> str:
    """A line to introduce the list: a fixed start, finished by the model."""
    asked: list[Message] = [{"role": "system", "content": LEAD_PROMPT}, *conversation]
    for lead in rng.sample(LEADS, len(LEADS)):
        line = _finished(backend, asked, lead)
        if line is not None:
            return line
    return PLAIN_LEAD


def _finished(backend: Backend, asked: Sequence[Message], lead: str) -> str | None:
    """``lead`` continued by the model's likeliest words up to a colon; None if unusable."""
    line = lead
    for _ in range(_LEAD_TOKENS):
        choices = [choice for choice in backend.next_tokens(asked, line) if choice.text]
        if not choices:
            return None
        line += max(choices, key=lambda choice: (choice.probability, choice.text)).text
        added = line[len(lead) :]
        end = _LINE_END.search(added)
        if end is not None:
            return _usable(lead, added[: end.start()])
    return None  # The model went on and on: a cut-off line would not make sense.


def _usable(lead: str, added: str) -> str | None:
    line = " ".join(f"{lead}{added}".split())
    said_something = bool(split_words(added))
    # A name of the list in the line would be read as part of the message.
    has_name = any(word in _INDEX for word in split_words(line))
    if not said_something or has_name or len(line.split()) > _LEAD_WORDS or _NOT_A_LINE.search(line):
        return None
    return line + ":"
