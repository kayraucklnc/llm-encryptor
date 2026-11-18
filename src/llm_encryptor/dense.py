"""The compact format: bits are coded in the model's own token probabilities.

This is arithmetic coding run backwards. The encrypted secret, read as a
binary fraction, picks a point between 0 and 1. At every step the model's
likely next tokens share out the current interval in proportion to their
probability, and the token whose share contains the point is written. Likely
tokens narrow the interval a little (few bits), unlikely ones a lot (many
bits), so the text carries about as many bits as the model has real choices.

The reader replays the same model on the same text, follows the same
intervals, and reads the secret off the final interval. That needs the same
model, and the text must arrive with its words and punctuation unchanged
(spacing and line breaks may change).

Two computers need not give exactly the same probabilities. So probabilities
are only used in a few coarse levels, and where one lies close to a level
boundary the reader tries both sides; the encryption tag tells which reading
is right.

The first words of the text are written freely to the user's instructions and
carry nothing; they set topic and tone, and the coded part simply continues
them. So the reader needs no prompt.
"""

from __future__ import annotations

import itertools
import random
import re
from collections.abc import Sequence
from dataclasses import dataclass

from .backend import Backend, Choice, Message
from .codec import RevealError
from .crypto import DENSE_OVERHEAD, MAX_SECRET_BYTES, DenseBox, Keys
from .embed import EmbedError, Progress, ProgressCallback
from .textcode import pack, unpack
from .words import complete_word_ends, is_word_char

# All of these are part of the format: sender and reader must agree on them.
OPENING_WORDS = 16
# A token's share of the interval, by its probability: (at least, shares).
# Flatter shares were tried and did not shorten the text: this model usually
# has one clear favourite, so there is little to flatten.
LEVELS = ((0.5, 16), (0.25, 8), (0.12, 4), (0.06, 2), (0.03, 1))
OPENING_PROMPT = (
    "You are a skilled writer. Write what the user asks for as natural, flowing prose "
    "in plain text. Name the subject within the first ten words. Always write at least "
    "150 words, even when the request is a single word or a greeting: then write a "
    "short piece inspired by it. Output only the text itself: no preamble, no title, "
    "no headings, no lists, no questions back to the user."
)
# Small models sometimes answer a short request with a word or two anyway.
_INSIST: Message = {
    "role": "user",
    "content": "That was too short. Write the full text now: at least 150 words of prose.",
}
# A word is only continued by a token at least this likely (avoids misspellings).
WORD_PIECE_LEVEL = 0.25
# How far, relatively, a probability may differ between writer and reader.
WOBBLE = 0.03
CONTINUE: tuple[Message, ...] = (
    {
        "role": "system",
        "content": "You are a writer in the middle of a long piece. Continue it naturally from "
        "where it stops, staying on its topic and in its tone and language. Write plain "
        "prose only. Never comment on the text and never wrap it up.",
    },
    {"role": "user", "content": "Write a long, detailed piece of at least 800 words."},
)

_OPENING_PIECES = 4 * OPENING_WORDS
_OPENING_TRIES = 5
_LOOKUP_TRIES = 12
_WRITE_TRIES = 3
_PADDING_BITS = 8192
_MAX_BITS = 8 * (DENSE_OVERHEAD + MAX_SECRET_BYTES + 1)
_TOKENS_PER_BIT = 8
_TAIL_TOKENS = 60
_MAX_READINGS = 1000
_MAX_DOUBTS_FLIPPED = 4
_SENTENCE_END = re.compile(r"[.!?][\"'”’)\]]*$")
_SPACE = re.compile(r"\s+")


class DenseError(RevealError):
    """The text could not be read in the compact format."""


@dataclass(frozen=True)
class _Interval:
    """The numbers from low/scale up to (low + width)/scale."""

    low: int = 0
    width: int = 1
    scale: int = 1

    def narrowed(self, before: int, share: int, total: int) -> _Interval:
        return _Interval(
            self.low * total + self.width * before, self.width * share, self.scale * total
        )

    def agreed_bits(self, limit: int) -> tuple[int, ...]:
        """The leading binary digits shared by every number in the interval."""
        first = (self.low << limit) // self.scale
        last = (((self.low + self.width) << limit) - 1) // self.scale
        agreed = limit - (first ^ last).bit_length()
        return tuple((first >> (limit - 1 - i)) & 1 for i in range(agreed))


def flatten(text: str) -> str:
    """Text with every run of spaces and line breaks as one space."""
    return _SPACE.sub(" ", text)


def split_opening(text: str) -> tuple[str, str]:
    """The free opening words of a text, and everything after them."""
    flat = flatten(text).strip()
    ends = complete_word_ends(flat)
    if len(ends) < OPENING_WORDS:
        raise DenseError("The text is too short to hold a message.")
    cut = ends[OPENING_WORDS - 1]
    return flat[:cut], flat[cut:]


@dataclass(frozen=True)
class Option:
    """A token that may come next. ``extends_word`` marks a piece that would
    continue the word being written."""

    token: str
    probability: float
    extends_word: bool = False


def _share(probability: float, extends_word: bool) -> int:
    if extends_word and probability < WORD_PIECE_LEVEL:
        return 0
    for at_least, share in LEVELS:
        if probability >= at_least:
            return share
    return 0


def options(choices: Sequence[Choice], written: str, first: bool = False) -> list[Option]:
    """Which tokens may come next, in a fixed order.

    Deterministic, because sender and reader must get the same answer. No
    token may read as the start of another once spacing is flattened;
    otherwise the reader could not tell which one was written. The first coded
    token may not extend the last opening word, or the reader would cut the
    opening in a different place.
    """
    seen = flatten(written)
    mid_word = is_word_char(written[-1:])
    keyed = sorted(
        (flatten(written + choice.text), choice.text, choice.probability)
        for choice in choices
        if choice.probability > 0 and choice.text.strip() and "\ufffd" not in choice.text
        and not (first and is_word_char(choice.text[0]))
    )
    kept: list[Option] = []
    keys: list[str] = []
    for key, token, probability in keyed:
        if key.rstrip() == seen.rstrip() or any(key.startswith(other) for other in keys):
            continue
        keys.append(key)
        kept.append(Option(token, probability, mid_word and is_word_char(token[0])))
    return kept


def shares(found: Sequence[Option]) -> list[int]:
    """How the interval is shared among the options."""
    result = [_share(o.probability, o.extends_word) for o in found]
    if found and not any(result):
        # Nothing stands out: let the likeliest token through.
        best = max(range(len(found)), key=lambda i: (found[i].probability, found[i].token))
        result[best] = 1
    return result


def _other_share(option: Option) -> int | None:
    """The share this option would get just across a nearby level boundary."""
    here = _share(option.probability, option.extends_word)
    nearby = {
        _share(option.probability * (1 - WOBBLE), option.extends_word),
        _share(option.probability * (1 + WOBBLE), option.extends_word),
    }
    return next(iter(nearby - {here}), None)


def write_dense(
    secret: str,
    keys: Keys,
    backend: Backend,
    conversation: Sequence[Message],
    on_progress: ProgressCallback | None = None,
    rng: random.Random | None = None,
) -> str:
    """A text that opens as ``conversation`` asks and carries ``secret``."""
    rng = rng or random.SystemRandom()
    opening = _opening(backend, [{"role": "system", "content": OPENING_PROMPT}, *conversation])
    # What a model server has cached changes its numbers slightly. Writer and
    # reader therefore both start from a freshly loaded model, and every text
    # is read back before it is returned, and written again if that fails.
    for _ in range(_WRITE_TRIES):
        text = _code(secret, keys, backend, opening, on_progress, rng)
        try:
            if read_dense(text, keys, backend) == secret:
                return text
        except DenseError:
            continue
    raise EmbedError("The finished text did not decode back to the secret. Please try again.")


def _code(
    secret: str,
    keys: Keys,
    backend: Backend,
    text: str,
    on_progress: ProgressCallback | None,
    rng: random.Random,
) -> str:
    notify = on_progress or (lambda _progress: None)
    payload = DenseBox(keys.dense).seal(pack(secret), text)
    wanted = 8 * len(payload)
    # The point to home in on: the payload, then random digits so that the
    # words after the payload are still sampled the way the model would.
    digits = wanted + _PADDING_BITS
    point = (int.from_bytes(payload, "big") << _PADDING_BITS) | rng.getrandbits(_PADDING_BITS)

    interval = _Interval()
    done_at: int | None = None
    backend.reset()
    for step in range(_TOKENS_PER_BIT * wanted + _TAIL_TOKENS):
        found = _lookup(backend, text, first=step == 0)
        choices = [(o.token, share) for o, share in zip(found, shares(found)) if share]
        if not choices:
            break
        total = sum(count for _, count in choices)
        # Which share of the interval holds the point?
        offset = point * interval.scale - (interval.low << digits)
        slot = offset * total // (interval.width << digits)
        before = 0
        token, count = choices[-1]
        for token, count in choices:
            if slot < before + count:
                break
            before += count
        interval = interval.narrowed(before, count, total)
        text += token
        agreed = min(len(interval.agreed_bits(wanted)), wanted)
        notify(Progress(text, agreed, wanted))
        if agreed == wanted:
            done_at = step if done_at is None else done_at
            if _SENTENCE_END.search(text.rstrip()) or step - done_at >= _TAIL_TOKENS:
                return text.rstrip()
    raise EmbedError(
        "The model stopped before the whole message was in the text. Try again."
    )


def read_dense(text: str, keys: Keys, backend: Backend) -> str:
    """Recover the secret from a compact-format text by replaying the model."""
    written, rest = split_opening(text)
    opening = written
    steps: list[tuple[list[Option], int]] = []
    backend.reset()
    while rest.strip():
        found = _lookup(backend, written, first=written == opening)
        index = next(
            (
                i
                for i, option in enumerate(found)
                if _continues(flatten(written + option.token), flatten(written), rest)
            ),
            None,
        )
        if index is None:
            break  # Nothing the model could have written here: the message ended earlier.
        token = found[index].token
        steps.append((found, index))
        rest = rest[len(flatten(written + token)) - len(flatten(written)) :]
        written += token

    box = DenseBox(keys.dense)
    for reading in _readings(steps):
        secret = _open(box, _bits(steps, reading), opening)
        if secret is not None:
            return secret
    raise DenseError(
        "No hidden message found. The password or model may be wrong, "
        "or the text was changed."
    )


def _readings(steps: Sequence[tuple[list[Option], int]]):
    """Ways to read the shares: as seen first, then with borderline ones moved.

    A token that was written must have had a share, so a written token we see
    below the lowest level is moved up without counting as a doubt.
    """
    base = [shares(found) for found, _ in steps]
    doubts = []
    for at, (found, chosen) in enumerate(steps):
        for i, option in enumerate(found):
            other = _other_share(option)
            if other is None:
                continue
            if i == chosen and base[at][i] == 0:
                base[at][i] = other
            elif not (i == chosen and other == 0):
                doubts.append((at, i, other))
    tried = 0
    for size in range(min(_MAX_DOUBTS_FLIPPED, len(doubts)) + 1):
        for flipped in itertools.combinations(doubts, size):
            if tried >= _MAX_READINGS:
                return
            tried += 1
            reading = [list(row) for row in base]
            for at, i, other in flipped:
                reading[at][i] = other
            yield reading


def _bits(steps: Sequence[tuple[list[Option], int]], reading: Sequence[Sequence[int]]) -> bytes:
    interval = _Interval()
    for (_, chosen), row in zip(steps, reading):
        if row[chosen] == 0:
            break
        interval = interval.narrowed(sum(row[:chosen]), row[chosen], sum(row))
    bits = interval.agreed_bits(_MAX_BITS)
    return bytes(
        sum(bit << (7 - i) for i, bit in enumerate(bits[start : start + 8]))
        for start in range(0, len(bits) - len(bits) % 8, 8)
    )


def _open(box: DenseBox, blob: bytes, opening: str) -> str | None:
    for length in range(DENSE_OVERHEAD + 1, len(blob) + 1):
        plaintext = box.open(blob[:length], opening)
        if plaintext is not None:
            try:
                return unpack(plaintext)
            except ValueError:
                continue
    return None


def _continues(new_flat: str, old_flat: str, rest: str) -> bool:
    """Does the text still to be read begin with what this token adds?"""
    added = new_flat[len(old_flat) :]
    return rest.startswith(added) or rest == added.rstrip()


def _lookup(backend: Backend, written: str, first: bool = False) -> list[Option]:
    # When the model would rather stop, ask again (the same way on both sides)
    # until it shows what else it could write.
    for attempt in range(_LOOKUP_TRIES):
        found = options(backend.next_tokens(CONTINUE, written, attempt), written, first)
        if found:
            return found
    return []


def _opening(backend: Backend, messages: Sequence[Message]) -> str:
    asked = list(messages)
    for _ in range(_OPENING_TRIES):
        sample = flatten(backend.start(asked, _OPENING_PIECES)).strip()
        ends = complete_word_ends(sample)
        if len(ends) >= OPENING_WORDS:
            return sample[: ends[OPENING_WORDS - 1]]
        if sample:
            asked = [*messages, {"role": "assistant", "content": sample}, _INSIST]
    raise EmbedError(
        "The model only gave a very short reply, too short to start a text. "
        "Say more about what to write (for example: write about a trip to Turkey), "
        "or pick a larger model."
    )
