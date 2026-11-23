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
import math
import random
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .backend import Backend, Choice, Message
from .codec import RevealError
from .crypto import (
    DENSE_SALT_LEN,
    DENSE_TAG_LEN,
    LEAN_TAG_BITS,
    MAX_SECRET_BYTES,
    DenseBox,
    Keys,
    LeanBox,
)
from .embed import EmbedError, Progress, ProgressCallback
from .textcode import pack, pack_bits, unpack, unpack_bits
from .wordcode import pack_words, unpack_words
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
    tag_bits: int = LEAN_TAG_BITS
    # How many readings of a text the reader may try; a shorter tag allows fewer.
    readings: int | None = None
    # The text must be the model's from its opening to its last word: nothing
    # may follow it. This keeps a text of another density, which the model
    # would not have written this way, from being taken for one of this.
    whole_text: bool = False
    # The secret is packed by the word (wordcode) instead of by the letter.
    by_word: bool = False
    # Only letters, digits and everyday punctuation may carry bits: no
    # brackets, emojis or markup, and no capital in the middle of a word.
    plain: bool = False
    # What the model is told, when that is not the usual article-like prose.
    opening_prompt: str | None = None
    continuation: tuple[Message, ...] | None = None


# People text each other every day, and a text may say almost anything: the
# most ordinary cover there is for a few sentences that carry something.
CHAT_OPENING = (
    "Write a casual text message to a friend about what the user asks, two or three short "
    "sentences, the way people really text. Plain text only: no emojis, no hashtags, no "
    "preamble, no quotes, no placeholders. Start with the message itself."
)
CHAT_CONTINUE: tuple[Message, ...] = (
    {
        "role": "system",
        "content": "You are texting a friend. Continue the message from where it stops, casually "
        "and simply, in the same tone and language, with the everyday details real plans have. "
        "Plain text only, no emojis. Never comment on the text or on these instructions.",
    },
    {"role": "user", "content": "Write the text message."},
)

# The reader tries each of these, so the writer is free to choose.
DENSITIES = {
    # Follows the model's own preferences: reads best.
    "natural": Density(((0.5, 16), (0.25, 8), (0.12, 4), (0.06, 2), (0.03, 1))),
    # Every plausible word is about equally likely: more bits per word.
    "tight": Density(((0.3, 2), (0.01, 1))),
    # A casual message of a few sentences. Every token within reach of the
    # likeliest one counts the same: where the model is sure, little else gets
    # through; where it has no favourite (what to bring, when, where) all its
    # ideas do. The secret is packed by the word and sealed with a short tag,
    # and of several drafts the most ordinary one is kept.
    "short": Density(
        ((0.07, 1),),
        piece_level=0.5,
        relative=True,
        floor=0.015,
        quick_ending=True,
        drafts=6,
        opening_words=5,
        salt_len=0,
        lean=True,
        tag_bits=20,
        readings=32,
        whole_text=True,
        by_word=True,
        plain=True,
        opening_prompt=CHAT_OPENING,
        continuation=CHAT_CONTINUE,
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
# What the app and the command line use unless told otherwise.
DEFAULT_DENSITY = "short"
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
# After this long, a slow model writes no further drafts.
_DRAFT_SECONDS = 15.0
# A text of at most this many words is first read as a short-density text.
_BRIEF_WORDS = 80
_SUBJECT_LETTERS = 3
# Words of a request that are about the writing, or about nothing much.
_NOT_A_SUBJECT = frozenset(
    [
        "about", "all", "texted", "reply", "back", "bring", "round", "also", "and", "any", "are", "been", "but", "can", "casual",
        "could", "for", "friend", "friendly", "friends", "from", "get", "good", "got",
        "had", "has", "have", "her", "here", "his", "how", "into", "its", "just", "let",
        "like", "little", "make", "many", "message", "more", "most", "much", "new", "nice",
        "not", "note", "one", "our", "out", "over", "please", "really", "say", "see",
        "short", "should", "some", "such", "tell", "text", "than", "that", "the", "their",
        "them", "then", "there", "they", "thing", "things", "this", "two", "very", "want",
        "was", "were", "what", "when", "who", "why", "will", "with", "would", "write",
        "yes", "you", "your",
    ]
)
_ASSISTANT_STARTS = frozenset(["sure", "certainly", "here", "okay", "alright"])
_SENTENCE_STOP = re.compile(r"[.!?]")
_SENTENCE_WORDS = 3
_PLAIN = re.compile(r"(?:[^\W_]|[\s.,!?'’:;-])*")
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


def _is_plain(token: str, written: str) -> bool:
    """Letters, digits and everyday punctuation only, used the everyday way."""
    mid_word = is_word_char(written[-1:])
    odd_piece = mid_word and is_word_char(token[0]) and any(char.isupper() for char in token)
    # An apostrophe at the end of a token starts a quotation or leaves a word hanging ("don' t").
    hanging = token.rstrip()[-1:] in ("'", "’")
    # A run of one- and two-word sentences ("You? Have fun. Where to?") reads as noise.
    sentence = _SENTENCE_STOP.split(written)[-1]
    clipped = token.lstrip()[:1] in (".", "!", "?") and len(split_words(sentence)) < _SENTENCE_WORDS
    return not (odd_piece or hanging or clipped) and _PLAIN.fullmatch(token) is not None


def _continuation(density: Density) -> tuple[Message, ...]:
    return density.continuation or CONTINUE


def options(
    choices: Sequence[Choice], written: str, first: bool = False, plain: bool = False
) -> list[Option]:
    """Which tokens may come next, in a fixed order.

    Deterministic, because sender and reader must get the same answer. No
    token may read as the start of another once spacing is flattened;
    otherwise the reader could not tell which one was written. The first coded
    token may not extend the last opening word, or the reader would cut the
    opening in a different place. With ``plain``, tokens with brackets, emojis
    or markup in them are left out, as is a capital in the middle of a word.
    """
    seen = flatten(written)
    mid_word = is_word_char(written[-1:])
    keyed = sorted(
        (flatten(written + choice.text), choice.text, choice.probability)
        for choice in choices
        if choice.probability > 0 and choice.text.strip() and "\ufffd" not in choice.text
        and not (first and is_word_char(choice.text[0]))
        and not (plain and not _is_plain(choice.text, written))
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
    clock: Callable[[], float] = time.monotonic,
    about: str | None = None,
) -> str:
    """A text that opens as ``conversation`` asks and carries ``secret``.

    ``about`` names what a short text should open on, when that is not simply
    every word of the requests (a reply steered to a new subject, say).
    """
    rng = rng or random.SystemRandom()
    chosen = DENSITIES[density]
    asked: list[Message] = [
        {"role": "system", "content": chosen.opening_prompt or OPENING_PROMPT},
        *conversation,
    ]
    # What a model server has cached changes its numbers slightly. Writer and
    # reader therefore both start from a freshly loaded model, and every text
    # is read back before it is returned. A text that fails, or that stops
    # making progress, is thrown away and begun again with a new opening.
    problem = "The finished text did not decode back to the secret."
    # With only a few free words at the start, they had better say what the text is about.
    said: Sequence[Message] = conversation if about is None else [{"role": "user", "content": about}]
    subject = _subject(said) if chosen.plain else frozenset()
    # Sealed without random bytes, the same opening would give the same text
    # again: a text asked for after an earlier one must start differently.
    earlier = [flatten(m["content"]) for m in conversation if m["role"] == "assistant"]
    earlier = earlier if chosen.plain else []
    for _ in range(_WRITE_TRIES):
        drafts: list[tuple[float, str]] = []
        started = clock()
        for _ in range(chosen.drafts):
            if drafts and clock() - started > _DRAFT_SECONDS:
                break  # A slow model gets fewer drafts rather than a long wait.
            opening = _opening(
                backend, asked, chosen.opening_words, chosen.plain, subject, earlier
            )
            try:
                drafts.append(_draft(secret, keys, backend, opening, on_progress, rng, chosen))
            except _Stuck as stuck:
                problem = str(stuck)
        # Which words the bits happen to pick is luck. Of several drafts the
        # one the model finds likeliest is kept: few words, and ordinary ones.
        for _, text in sorted(drafts):
            if _reads_back(text, secret, keys, backend):
                return text
    raise EmbedError(f"{problem} Please try again, or pick a larger model.")


def _subject(conversation: Sequence[Message]) -> frozenset[str]:
    """The words of the user's requests that say what the text should be about."""
    said = " ".join(message["content"] for message in conversation if message["role"] == "user")
    return frozenset(
        word for word in split_words(said) if len(word) >= _SUBJECT_LETTERS and word not in _NOT_A_SUBJECT
    )


def _mentions(opening: str, subject: frozenset[str]) -> bool:
    """Does the opening use a word of the subject, or a longer or shorter form of one?"""
    return any(
        word.startswith(key) or key.startswith(word)
        for word in split_words(opening)
        if len(word) >= _SUBJECT_LETTERS
        for key in subject
    )


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
    """The opening ``text`` continued so that it carries ``secret``."""
    return _draft(secret, keys, backend, text, on_progress, rng, density)[1]


def _draft(
    secret: str,
    keys: Keys,
    backend: Backend,
    text: str,
    on_progress: ProgressCallback | None,
    rng: random.Random,
    density: Density,
) -> tuple[float, str]:
    """A text that carries ``secret``, and how surprising the model finds it (in bits)."""
    notify = on_progress or (lambda _progress: None)
    surprise = 0.0
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
        found = _lookup(backend, text, step == 0, _continuation(density), density.plain)
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
        surprise -= math.log2(option.probability)
        agreed = min(len(interval.agreed_bits(wanted)), wanted)
        notify(Progress(text, agreed, wanted))
        if agreed == wanted:
            done_at = step if done_at is None else done_at
            ended = not density.finish_sentence or _SENTENCE_END.search(text.rstrip())
            if ended or step - done_at >= _TAIL_TOKENS:
                return surprise, text.rstrip()
        elif agreed > most:
            gained_at, most = step, agreed
        elif step - gained_at >= _STALL_TOKENS:
            break
    raise _Stuck("The model kept writing without leaving room for the message.")


def _sealed(secret: str, keys: Keys, opening: str, density: Density) -> str:
    """The encrypted secret as a string of 0s and 1s."""
    if density.lean:
        packed = pack_words(secret) if density.by_word else pack_bits(secret)
        return LeanBox(keys.dense, density.tag_bits).seal(packed, opening)
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


@dataclass(frozen=True)
class ReadProgress:
    """How far the reader has got: the text followed so far and the bits it has given up."""

    read: str
    bits: str


def read_dense(
    text: str,
    keys: Keys,
    backend: Backend,
    on_read: Callable[[ReadProgress], None] | None = None,
) -> str:
    """Recover the secret from a compact-format text by replaying the model."""
    box = DenseBox(keys.dense)
    too_short = True
    # Where the coded part starts, and what the model was told, depend on the
    # density, so each such pair gets its own pass over the text.
    for one in _passes(text):
        try:
            opening, steps, to_the_end = _replay(text, backend, *one, _watcher(one, on_read))
        except DenseError:
            continue
        too_short = False
        for density in DENSITIES.values():
            if _pass_of(density) != one or (density.whole_text and not to_the_end):
                continue
            lean = LeanBox(keys.dense, density.tag_bits)
            for reading in _readings(steps, density):
                bits = _bits(steps, reading)
                # The older sealing first: its texts were written before this
                # reader existed, so nothing else guards them against the rare
                # tag that fits by accident.
                secret = None
                if density.salt_len:
                    secret = _open(box, _whole_bytes(bits), opening, density.salt_len)
                if secret is None and density.lean:
                    secret = _open_lean(lean, bits, opening, density.by_word)
                if secret is not None:
                    return secret
    if too_short:
        raise DenseError("The text is too short to hold a message.")
    raise DenseError(
        "No hidden message found. The password or model may be wrong, "
        "or the text was changed."
    )


_Pass = tuple[int, tuple[Message, ...], bool]
_Steps = list[tuple[list["Option"], int]]


def _watcher(
    one: _Pass, on_read: Callable[[ReadProgress], None] | None
) -> Callable[[str, _Steps], None] | None:
    """Reports each step of a pass, read the way the first density of that pass reads it."""
    if on_read is None:
        return None
    density = next(d for d in DENSITIES.values() if _pass_of(d) == one)

    def watch(written: str, steps: _Steps) -> None:
        on_read(ReadProgress(written, _bits(steps, [shares(found, density) for found, _ in steps])))

    return watch


def _pass_of(density: Density) -> _Pass:
    """What a pass over a text depends on: opening words, what the model was told, plain or not."""
    return density.opening_words, _continuation(density), density.plain


def _passes(text: str) -> list[_Pass]:
    """The passes to make over a text.

    Longest opening first, except that a brief text is most likely a short
    one and is read as that first. Every pass is still made if needed.
    """
    passes: list[_Pass] = []
    for density in DENSITIES.values():
        if _pass_of(density) not in passes:
            passes.append(_pass_of(density))
    passes.sort(key=lambda one: -one[0])
    if len(text.split()) <= _BRIEF_WORDS:
        passes.sort(key=lambda one: one != _pass_of(SHORT))
    return passes


def _replay(
    text: str,
    backend: Backend,
    words: int,
    continuation: tuple[Message, ...] = CONTINUE,
    plain: bool = False,
    on_step: Callable[[str, list[tuple[list[Option], int]]], None] | None = None,
) -> tuple[str, list[tuple[list[Option], int]], bool]:
    """Follow the text token by token: what the model offered, and what was written.

    Also says whether that got to the end of the text, or stopped at a place
    where the model could not have written what stands there.
    """
    written, rest = split_opening(text, words)
    opening = written
    steps: list[tuple[list[Option], int]] = []
    backend.reset()
    while rest.strip():
        found = _lookup(backend, written, written == opening, continuation, plain)
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
        if on_step is not None:
            on_step(written, steps)
    return opening, steps, not rest.strip()


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
            if tried >= (density.readings or _MAX_READINGS):
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


def _open_lean(box: LeanBox, bits: str, opening: str, by_word: bool = False) -> str | None:
    unpacked = unpack_words if by_word else unpack_bits
    for plaintext in box.open(bits, opening):
        try:
            return unpacked(plaintext)
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


def _lookup(
    backend: Backend,
    written: str,
    first: bool = False,
    continuation: tuple[Message, ...] = CONTINUE,
    plain: bool = False,
) -> list[Option]:
    # When the model would rather stop, ask again (the same way on both sides)
    # until it shows what else it could write.
    for attempt in range(_LOOKUP_TRIES):
        found = options(backend.next_tokens(continuation, written, attempt), written, first, plain)
        if found:
            return found
    return []


def _opening(
    backend: Backend,
    messages: Sequence[Message],
    words: int,
    plain: bool = False,
    subject: frozenset[str] = frozenset(),
    earlier: Sequence[str] = (),
) -> str:
    """The first ``words`` words of a reply to ``messages``.

    With ``plain``, a start with a placeholder or an emoji in it is asked for
    again; so is one that starts an ``earlier`` text, and with a ``subject``
    one that mentions none of its words. The best start seen is used when the
    model never gives a better one.
    """
    asked = list(messages)
    spare: tuple[tuple[bool, bool, bool], str] | None = None
    for _ in range(_OPENING_TRIES):
        sample = flatten(backend.start(asked, _OPENING_PIECES)).strip()
        ends = complete_word_ends(sample)
        if len(ends) >= words:
            opening = sample[: ends[words - 1]]
            is_plain = not plain or (
                _PLAIN.fullmatch(opening) is not None
                # "Sure! Here is..." is a model answering a request, not a person texting.
                and split_words(opening)[0] not in _ASSISTANT_STARTS
            )
            is_new = not any(text.startswith(opening) for text in earlier)
            on_subject = not subject or _mentions(opening, subject)
            if is_plain and is_new and on_subject:
                return opening
            if spare is None or (is_plain, is_new, on_subject) > spare[0]:
                spare = ((is_plain, is_new, on_subject), opening)
        elif sample:
            asked = [*messages, {"role": "assistant", "content": sample}, _INSIST]
    if spare is not None:
        return spare[1]
    raise EmbedError(
        "The model only gave a very short reply, too short to start a text. "
        "Say more about what to write (for example: write about a trip to Turkey), "
        "or pick a larger model."
    )
