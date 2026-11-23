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
from .crypto import (
    DENSE_SALT_LEN,
    DENSE_TAG_LEN,
    MAX_SECRET_BYTES,
    DenseBox,
    Keys,
    LeanBox,
)
from .embed import EmbedError, Progress, ProgressCallback
from .textcode import pack, pack_bits, unpack, unpack_bits
from .words import complete_word_ends, is_word_char, split_words


# All of these are part of the format: sender and reader must agree on them.
@dataclass(frozen=True)
class Density:
    """How the interval is shared among the next tokens.

    ``levels`` gives a token's share by its probability: (at least, shares).
    ``piece_level`` is how likely a token must be to continue a word (this
    avoids misspellings). With ``relative`` both are fractions of the likeliest
    token's probability, ``floor`` is the least probability that counts at
    all, and nothing may cut a word short that is less than ``piece_level``
    times as likely as going on with it. ``finish_sentence`` keeps writing to the end of the sentence once the
    message is in; ``quick_ending`` gets there by the likeliest words instead
    of by chance.
    """

    levels: tuple[tuple[float, int], ...]
    piece_level: float = 0.25
    finish_sentence: bool = True
    # The free words at the start, and the random bytes sent along. Fewer
    # opening words repeat more easily, so they come with more random bytes.
    opening_words: int = 16
    salt_len: int = DENSE_SALT_LEN
    relative: bool = False
    floor: float = 0.0
    quick_ending: bool = False
    # How many texts to write before keeping the shortest.
    drafts: int = 1
    # Sealed with a tag only (crypto.LeanBox): no random bytes, no padding.
    # A density that also has a ``salt_len`` was sealed the older way before,
    # and the reader still accepts that.
    lean: bool = False


# The reader tries each of these, so the writer is free to choose.
DENSITIES = {
    # Follows the model's own preferences: reads best.
    "natural": Density(((0.5, 16), (0.25, 8), (0.12, 4), (0.06, 2), (0.03, 1))),
    # Every plausible word is about equally likely: more bits per word.
    "tight": Density(((0.3, 2), (0.01, 1))),
    # Every token within reach of the likeliest one counts the same. Where the
    # model is sure, little else gets through; where it has no favourite (a
    # name, a detail, a turn of phrase) everything it lists does. About twice
    # the bits per word of natural, and it still reads as ordinary prose.
    "short": Density(
        ((0.05, 1),),
        piece_level=0.5,
        relative=True,
        floor=0.01,
        quick_ending=True,
        drafts=3,
        opening_words=8,
        salt_len=0,
        lean=True,
    ),
    # Every token the model lists counts the same, however unlikely, and the
    # text stops the moment the message is in: shortest, and it reads as nonsense.
    "max": Density(
        ((0.0005, 1),),
        piece_level=0.0,
        finish_sentence=False,
        opening_words=4,
        salt_len=4,
        lean=True,
    ),
}
NATURAL, TIGHT, SHORT, MAX = (DENSITIES[name] for name in ("natural", "tight", "short", "max"))
NATURAL_OPENING = NATURAL.opening_words
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
# A run of this many words may not appear twice: it stops the model going in circles.
NO_REPEAT_WORDS = 3
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

_OPENING_PIECES = 64
_OPENING_TRIES = 5
_LOOKUP_TRIES = 12
_WRITE_TRIES = 3
_PADDING_BITS = 8192
_MAX_BITS = 8 * (8 + DENSE_TAG_LEN + MAX_SECRET_BYTES + 1)
_TOKENS_PER_BIT = 3
_TAIL_TOKENS = 60
# A healthy text hides a bit every token or two; give up on one that hides none for this long.
_STALL_TOKENS = 40
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


def split_opening(text: str, words: int = NATURAL_OPENING) -> tuple[str, str]:
    """The free opening words of a text, and everything after them."""
    flat = flatten(text).strip()
    ends = complete_word_ends(flat)
    if len(ends) < words:
        raise DenseError("The text is too short to hold a message.")
    cut = ends[words - 1]
    return flat[:cut], flat[cut:]


@dataclass(frozen=True)
class Option:
    """A token that may come next. ``extends_word`` marks a piece that would
    continue the word being written."""

    token: str
    probability: float
    extends_word: bool = False


def _share(
    probability: float, extends_word: bool, density: Density, top: float = 1.0, word_top: float = 0.0
) -> int:
    """A token's share. ``top`` is the probability of the likeliest token on
    offer and ``word_top`` that of the likeliest one continuing the word."""
    scale = top if density.relative else 1.0
    least = density.piece_level * (scale if extends_word else word_top)
    if probability < max(density.floor, least):
        return 0
    for at_least, share in density.levels:
        if probability >= at_least * scale:
            return share
    return 0


def _top(found: Sequence[Option], density: Density) -> tuple[float, float]:
    """The probabilities the shares are measured against: of the likeliest
    token, and of the likeliest one that continues the word being written.

    The second matters only with relative levels. Where the model wants to go
    on with a word, a comma or a new word must be nearly as likely as that,
    or the word would be left half-written.
    """
    top = max((option.probability for option in found), default=1.0)
    if not density.relative:
        return top, 0.0
    return top, max((option.probability for option in found if option.extends_word), default=0.0)


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
    fresh = _without_repeats(kept, written)
    return fresh or kept


def _without_repeats(found: Sequence[Option], written: str) -> list[Option]:
    """Drop tokens that would make the text say the same few words again.

    A model that starts repeating itself becomes certain of every next word,
    and a certain word hides nothing: the text would grow without carrying
    more of the message.
    """
    words = split_words(written)
    runs = {tuple(words[i : i + NO_REPEAT_WORDS]) for i in range(len(words) - NO_REPEAT_WORDS + 1)}
    fresh = []
    for option in found:
        after = split_words(written + option.token)
        last = tuple(after[-NO_REPEAT_WORDS:])
        # A token that adds no letters (punctuation) cannot start a repeat.
        if after == words or len(last) < NO_REPEAT_WORDS or last not in runs:
            fresh.append(option)
    return fresh


def shares(found: Sequence[Option], density: Density = NATURAL) -> list[int]:
    """How the interval is shared among the options."""
    top, word_top = _top(found, density)
    result = [_share(o.probability, o.extends_word, density, top, word_top) for o in found]
    if found and not any(result):
        # Nothing stands out: let the likeliest token through.
        best = max(range(len(found)), key=lambda i: (found[i].probability, found[i].token))
        result[best] = 1
    return result


def _other_share(
    option: Option, density: Density = NATURAL, top: float = 1.0, word_top: float = 0.0
) -> int | None:
    """The share this option would get just across a nearby level boundary."""
    here = _share(option.probability, option.extends_word, density, top, word_top)
    # Measured against another token, two probabilities wobble instead of one.
    wobble = 2 * WOBBLE if density.relative else WOBBLE
    nearby = {
        _share(option.probability * (1 - wobble), option.extends_word, density, top, word_top),
        _share(option.probability * (1 + wobble), option.extends_word, density, top, word_top),
    }
    return next(iter(nearby - {here}), None)


def write_dense(
    secret: str,
    keys: Keys,
    backend: Backend,
    conversation: Sequence[Message],
    on_progress: ProgressCallback | None = None,
    rng: random.Random | None = None,
    density: str = "natural",
) -> str:
    """A text that opens as ``conversation`` asks and carries ``secret``."""
    rng = rng or random.SystemRandom()
    chosen = DENSITIES[density]
    asked = [{"role": "system", "content": OPENING_PROMPT}, *conversation]
    # What a model server has cached changes its numbers slightly. Writer and
    # reader therefore both start from a freshly loaded model, and every text
    # is read back before it is returned. A text that fails, or that stops
    # making progress, is thrown away and begun again with a new opening.
    problem = "The finished text did not decode back to the secret."
    for _ in range(_WRITE_TRIES):
        drafts = []
        for _ in range(chosen.drafts):
            opening = _opening(backend, asked, chosen.opening_words)
            try:
                drafts.append(_code(secret, keys, backend, opening, on_progress, rng, chosen))
            except _Stuck as stuck:
                problem = str(stuck)
        # How long a text gets is partly luck (which words the bits happen to
        # pick), so the shortest of a few drafts is a good deal shorter.
        for text in sorted(drafts, key=lambda draft: len(draft.split())):
            if _reads_back(text, secret, keys, backend):
                return text
    raise EmbedError(f"{problem} Please try again, or pick a larger model.")


def _reads_back(text: str, secret: str, keys: Keys, backend: Backend) -> bool:
    try:
        return read_dense(text, keys, backend) == secret
    except DenseError:
        return False


class _Stuck(Exception):
    """This attempt at a text is not getting anywhere."""


def _code(
    secret: str,
    keys: Keys,
    backend: Backend,
    text: str,
    on_progress: ProgressCallback | None,
    rng: random.Random,
    density: Density,
) -> str:
    notify = on_progress or (lambda _progress: None)
    payload = _sealed(secret, keys, text, density)
    wanted = len(payload)
    # The point to home in on: the payload, then random digits so that the
    # words after the payload are still sampled the way the model would.
    digits = wanted + _PADDING_BITS
    point = (int(payload, 2) << _PADDING_BITS) | rng.getrandbits(_PADDING_BITS)

    interval = _Interval()
    done_at: int | None = None
    gained_at, most = 0, 0
    backend.reset()
    for step in range(_TOKENS_PER_BIT * wanted + _TAIL_TOKENS):
        found = _lookup(backend, text, first=step == 0)
        choices = [(o, share) for o, share in zip(found, shares(found, density)) if share]
        if not choices:
            raise _Stuck("The model stopped before the whole message was in the text.")
        total = sum(count for _, count in choices)
        if done_at is not None and density.quick_ending:
            # The message is in, so these words are free: head for the full stop.
            at = _closing(choices)
        else:
            # Which share of the interval holds the point?
            offset = point * interval.scale - (interval.low << digits)
            slot = offset * total // (interval.width << digits)
            at = _holding(slot, [count for _, count in choices])
        option, count = choices[at]
        interval = interval.narrowed(sum(c for _, c in choices[:at]), count, total)
        text += option.token
        agreed = min(len(interval.agreed_bits(wanted)), wanted)
        notify(Progress(text, agreed, wanted))
        if agreed == wanted:
            done_at = step if done_at is None else done_at
            ended = not density.finish_sentence or _SENTENCE_END.search(text.rstrip())
            if ended or step - done_at >= _TAIL_TOKENS:
                return text.rstrip()
        elif agreed > most:
            gained_at, most = step, agreed
        elif step - gained_at >= _STALL_TOKENS:
            break
    raise _Stuck("The model kept writing without leaving room for the message.")


def _sealed(secret: str, keys: Keys, opening: str, density: Density) -> str:
    """The encrypted secret as a string of 0s and 1s."""
    if density.lean:
        return LeanBox(keys.dense).seal(pack_bits(secret), opening)
    payload = DenseBox(keys.dense).seal(pack(secret), opening, density.salt_len)
    return "".join(f"{byte:08b}" for byte in payload)


def _holding(slot: int, counts: Sequence[int]) -> int:
    """Which share holds the given slot, counting the shares from the first."""
    before = 0
    for at, count in enumerate(counts):
        if slot < before + count:
            return at
        before += count
    return len(counts) - 1


def _closing(choices: Sequence[tuple[Option, int]]) -> int:
    """The token that ends the sentence if one is on offer, else the likeliest."""
    for at, (option, _) in enumerate(choices):
        if _SENTENCE_END.search(option.token.rstrip()):
            return at
    return max(range(len(choices)), key=lambda at: (choices[at][0].probability, choices[at][0].token))


def read_dense(text: str, keys: Keys, backend: Backend) -> str:
    """Recover the secret from a compact-format text by replaying the model."""
    box, lean = DenseBox(keys.dense), LeanBox(keys.dense)
    too_short = True
    # Where the coded part starts depends on the density, so each opening
    # length gets its own pass over the text. The usual one goes first.
    for words in sorted({density.opening_words for density in DENSITIES.values()}, reverse=True):
        try:
            opening, steps = _replay(text, backend, words)
        except DenseError:
            continue
        too_short = False
        for density in DENSITIES.values():
            if density.opening_words != words:
                continue
            for reading in _readings(steps, density):
                bits = _bits(steps, reading)
                # The older sealing first: its texts were written before this
                # reader existed, so nothing else guards them against the rare
                # tag that fits by accident.
                secret = None
                if density.salt_len:
                    secret = _open(box, _whole_bytes(bits), opening, density.salt_len)
                if secret is None and density.lean:
                    secret = _open_lean(lean, bits, opening)
                if secret is not None:
                    return secret
    if too_short:
        raise DenseError("The text is too short to hold a message.")
    raise DenseError(
        "No hidden message found. The password or model may be wrong, "
        "or the text was changed."
    )


def _replay(text: str, backend: Backend, words: int) -> tuple[str, list[tuple[list[Option], int]]]:
    """Follow the text token by token: what the model offered, and what was written."""
    written, rest = split_opening(text, words)
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
    return opening, steps


def _readings(steps: Sequence[tuple[list[Option], int]], density: Density):
    """Ways to read the shares: as seen first, then with borderline ones moved.

    A token that was written must have had a share, so a written token we see
    below the lowest level is moved up without counting as a doubt.
    """
    base = [shares(found, density) for found, _ in steps]
    doubts = []
    for at, (found, chosen) in enumerate(steps):
        top, word_top = _top(found, density)
        for i, option in enumerate(found):
            other = _other_share(option, density, top, word_top)
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


def _bits(steps: Sequence[tuple[list[Option], int]], reading: Sequence[Sequence[int]]) -> str:
    """The bits the text settles under this reading, as a string of 0s and 1s."""
    interval = _Interval()
    for (_, chosen), row in zip(steps, reading):
        if row[chosen] == 0:
            break
        interval = interval.narrowed(sum(row[:chosen]), row[chosen], sum(row))
    return "".join(map(str, interval.agreed_bits(_MAX_BITS)))


def _whole_bytes(bits: str) -> bytes:
    whole = len(bits) - len(bits) % 8
    return int(bits[:whole], 2).to_bytes(whole // 8, "big") if whole else b""


def _open_lean(box: LeanBox, bits: str, opening: str) -> str | None:
    for plaintext in box.open(bits, opening):
        try:
            return unpack_bits(plaintext)
        except ValueError:
            continue
    return None


def _open(box: DenseBox, blob: bytes, opening: str, salt_len: int) -> str | None:
    for length in range(salt_len + DENSE_TAG_LEN + 1, len(blob) + 1):
        plaintext = box.open(blob[:length], opening, salt_len)
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


def _opening(backend: Backend, messages: Sequence[Message], words: int) -> str:
    asked = list(messages)
    for _ in range(_OPENING_TRIES):
        sample = flatten(backend.start(asked, _OPENING_PIECES)).strip()
        ends = complete_word_ends(sample)
        if len(ends) >= words:
            return sample[: ends[words - 1]]
        if sample:
            asked = [*messages, {"role": "assistant", "content": sample}, _INSIST]
    raise EmbedError(
        "The model only gave a very short reply, too short to start a text. "
        "Say more about what to write (for example: write about a trip to Turkey), "
        "or pick a larger model."
    )
