"""Steer a model's writing so that its words spell out the payload bits.

The text grows one block of words at a time. For each block we ask the model
which tokens could come next, and sample among them exactly as an ordinary
sampler would, except that a finished block is kept only when its keyed hash
equals the next payload bits. Answers are remembered, so drawing the same
wording again costs nothing. When a handful of draws finds no fitting wording,
the model has little freedom here, and rather than force an odd word we take
the previous block back and reword that instead.
"""

from __future__ import annotations

import random
import re
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeVar

from .backend import Backend, Message
from .words import (
    BITS_PER_BLOCK,
    WORDS_PER_BLOCK,
    block_bits,
    clean,
    complete_word_ends,
    is_word_char,
    joins_cleanly,
    split_words,
)

_SENTENCE_END = re.compile(r"[.!?][\"')\]]*(?=\s|$)")
_OPENING_TRIES = 5
_END_RETRIES = 3
_MAX_BLOCK_CHARS = 40 * WORDS_PER_BLOCK


@dataclass(frozen=True)
class EmbedConfig:
    temperature: float = 1.0
    min_probability: float = 0.05
    tail_tokens: int = 150
    draws_per_block: int = 12
    lookups_per_block: int = 80
    # When the search keeps falling back, sample a little more freely.
    setbacks_per_warmup: int = 6
    warmup_step: float = 0.15
    max_temperature: float = 1.8


@dataclass(frozen=True)
class Progress:
    text: str
    embedded: int
    total: int


class EmbedError(RuntimeError):
    """The model could not be steered into carrying the whole payload."""


DEFAULT_CONFIG = EmbedConfig()
ProgressCallback = Callable[[Progress], None]
_Option = TypeVar("_Option")
_Weighted = tuple[tuple[str, float], ...]


def embed(
    backend: Backend,
    messages: Sequence[Message],
    payload: Sequence[int],
    mac_key: bytes,
    config: EmbedConfig = DEFAULT_CONFIG,
    on_progress: ProgressCallback | None = None,
    rng: random.Random | None = None,
) -> str:
    """Write a text whose words, after a free opening, carry ``payload``."""
    notify = on_progress or (lambda _progress: None)
    rng = rng or random.SystemRandom()
    model = _Model(backend, messages, config)
    total = len(payload)
    lookup_limit = config.lookups_per_block * -(-total // BITS_PER_BLOCK) + 500

    def search_from(start: str, done: int) -> _BlockSearch:
        needed = tuple(payload[done : done + BITS_PER_BLOCK])
        return _BlockSearch(model, start, needed, mac_key, config.draws_per_block, rng)

    text = model.opening()
    embedded = furthest = setbacks = 0
    finished: list[_BlockSearch] = []
    current = search_from(text, embedded)
    notify(Progress(text, embedded, total))

    while embedded < total:
        if model.lookups > lookup_limit:
            raise EmbedError(
                "The model could not fit the message into the text. "
                "Try again, loosen the instruction, or use a larger model."
            )
        found = current.next_fit()
        if found is not None:
            finished.append(current)
            text, embedded = found, embedded + len(current.needed)
            current = search_from(text, embedded)
            if embedded > furthest:
                furthest, setbacks = embedded, 0
        else:
            setbacks += 1
            if finished:
                # No wording fits here: take the previous block back and reword it.
                current = finished.pop()
                text, embedded = current.base, embedded - len(current.needed)
            else:
                text = model.opening()
                current = search_from(text, embedded)
        model.warm_up(setbacks // config.setbacks_per_warmup)
        notify(Progress(text, embedded, total))

    return _close(model, text, config, rng)


class _Model:
    """The backend, with temperature and a probability floor applied."""

    def __init__(self, backend: Backend, messages: Sequence[Message], config: EmbedConfig) -> None:
        self._backend = backend
        self._messages = messages
        self._config = config
        self._temperature = config.temperature
        self.lookups = 0

    def warm_up(self, steps: int) -> None:
        """Raise the sampling temperature by ``steps`` notches above the configured one."""
        raised = self._config.temperature + steps * self._config.warmup_step
        self._temperature = min(raised, max(self._config.max_temperature, self._config.temperature))

    def opening(self) -> str:
        """The first word is free: it only gets the model past an empty reply."""
        for _ in range(_OPENING_TRIES):
            self.lookups += 1
            sample = clean(self._backend.start(self._messages)).lstrip()
            ends = complete_word_ends(sample)
            if ends:
                return sample[: ends[0]]
        raise EmbedError("The model did not start writing. Check the model and try again.")

    def choices(self, prefix: str, allow_end: bool = False) -> _Weighted:
        """Next tokens worth considering, with probabilities that sum to one."""
        for _ in range(1 if allow_end else _END_RETRIES):
            self.lookups += 1
            weighted = [
                (clean(choice.text), choice.probability ** (1 / self._temperature))
                for choice in self._backend.next_tokens(self._messages, prefix)
                if _usable(choice.text, choice.probability, allow_end)
            ]
            if weighted:
                return _trim(weighted, self._config.min_probability)
        return ()


def _usable(text: str, probability: float, allow_end: bool) -> bool:
    if probability <= 0 or "�" in text:
        return False
    if not text:
        return allow_end
    return not unicodedata.combining(clean(text)[0])


def _trim(weighted: Sequence[tuple[str, float]], floor: float) -> _Weighted:
    """Normalise, then drop tokens too unlikely to read naturally."""
    total = sum(weight for _, weight in weighted)
    kept = [(text, weight) for text, weight in weighted if weight / total >= floor]
    kept = kept or [max(weighted, key=lambda item: item[1])]
    kept_total = sum(weight for _, weight in kept)
    return tuple((text, weight / kept_total) for text, weight in kept)


def _pick(options: Sequence[tuple[float, _Option]], rng: random.Random) -> _Option:
    """Draw one option with probability proportional to its weight."""
    threshold = rng.random() * sum(weight for weight, _ in options)
    for weight, option in options:
        threshold -= weight
        if threshold <= 0:
            return option
    return options[-1][1]


class _Leaf:
    """A complete block of words."""

    def __init__(self, text: str, fits: bool) -> None:
        self.text = text
        self.fits = fits
        self.used = False


class _Node:
    """A partly written block; ``edges`` holds what the model may write next."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.edges: list[tuple[float, _Node | _Leaf]] | None = None


class _BlockSearch:
    """Samples wordings of one block until one carries the needed bits."""

    def __init__(
        self,
        model: _Model,
        base: str,
        needed: tuple[int, ...],
        mac_key: bytes,
        draws: int,
        rng: random.Random,
    ) -> None:
        self.base = base
        self.needed = needed
        self._model = model
        self._mac_key = mac_key
        self._draws = draws
        self._rng = rng
        self._base_words = split_words(base)
        self._root = _Node("")
        self._leaves: dict[str, _Leaf] = {}

    def next_fit(self) -> str | None:
        """The text with a fitting, not yet used block appended, or None."""
        for _ in range(self._draws):
            leaf = self._draw()
            if leaf is not None and leaf.fits and not leaf.used:
                leaf.used = True
                return self.base + leaf.text
        return None

    def _draw(self) -> _Leaf | None:
        """Sample one block, token by token, as the model would write it."""
        node = self._root
        while True:
            if node.edges is None:
                node.edges = self._expand(node)
            if not node.edges:
                return None
            target = _pick(node.edges, self._rng)
            if isinstance(target, _Leaf):
                return target
            node = target

    def _expand(self, node: _Node) -> list[tuple[float, _Node | _Leaf]]:
        edges: list[tuple[float, _Node | _Leaf]] = []
        for token, probability in self._model.choices(self.base + node.text):
            if not node.text and not joins_cleanly(self.base, token):
                continue
            text = node.text + token
            ends = complete_word_ends(text)
            if len(ends) >= WORDS_PER_BLOCK:
                edges.append((probability, self._leaf(text[: ends[WORDS_PER_BLOCK - 1]])))
            elif len(text) <= _MAX_BLOCK_CHARS:
                edges.append((probability, _Node(text)))
        return edges

    def _leaf(self, text: str) -> _Leaf:
        if text not in self._leaves:
            self._leaves[text] = _Leaf(text, self._fits(text))
        return self._leaves[text]

    def _fits(self, block_text: str) -> bool:
        words = split_words(self.base + block_text)
        count = len(self._base_words)
        if words[:count] != self._base_words or len(words) != count + WORDS_PER_BLOCK:
            return False
        bits = block_bits(self._mac_key, words[count:])
        return all(bit == want for bit, want in zip(bits, self.needed))


def _close(model: _Model, text: str, config: EmbedConfig, rng: random.Random) -> str:
    """Let the model finish freely; words after the payload carry nothing."""
    tail = ""
    ended = False
    for _ in range(config.tail_tokens):
        options = [
            (probability, token)
            for token, probability in model.choices(text + tail, allow_end=True)
            if tail or not token or joins_cleanly(text, token)
        ]
        token = _pick(options, rng) if options else ""
        if not token:
            ended = True
            break
        tail += token
    if not ended:
        tail = _trim_to_sentence(tail)
    finished = (text + tail).rstrip()
    return finished + "." if finished and is_word_char(finished[-1]) else finished


def _trim_to_sentence(tail: str) -> str:
    ends = [m.end() for m in _SENTENCE_END.finditer(tail)]
    if ends:
        return tail[: ends[-1]]
    word_ends = complete_word_ends(tail)
    return tail[: word_ends[-1]] if word_ends else ""
