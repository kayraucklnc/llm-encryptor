"""Stand-ins for a language model."""

from __future__ import annotations

import random
import re

from llm_encryptor.backend import Choice

VOCABULARY = ["the", "sea", "was", "calm", "and", "bright", "while", "old", "ships", "moved", "slowly", "past", "green", "hills", "people", "walked", "along", "narrow", "streets", "drinking", "tea", "sharing", "stories", "under", "warm", "evening", "light", "markets", "sold", "fresh", "bread", "olives", "sweet", "figs", "music", "drifted", "from", "open", "windows", "children", "laughed", "near", "quiet", "fountains", "travellers", "rested", "happily"]
_WEIGHTS = (0.4, 0.25, 0.15, 0.1, 0.06)


class FakeBackend:
    """A toy model: the same text so far always gets the same next-token choices.

    ``forced`` maps a word to the only word allowed after it, which gives the
    embedder stretches where it has no freedom at all.
    """

    def __init__(self, seed: int = 0, forced: dict[str, str] | None = None) -> None:
        self._seed = seed
        self._forced = forced or {}
        self.calls = 0

    def start(self, messages):
        self.calls += 1
        return "The sea was"

    def next_tokens(self, messages, prefix):
        self.calls += 1
        chooser = random.Random(f"{self._seed}|{prefix[-30:]}")
        words = chooser.sample(VOCABULARY, len(_WEIGHTS))
        if prefix.endswith("\n\n"):
            return [Choice(word.capitalize(), weight) for word, weight in zip(words, _WEIGHTS)]
        if prefix.endswith("."):
            starts = [Choice(" " + word.capitalize(), weight) for word, weight in zip(words, _WEIGHTS)]
            return [*starts, Choice("", 0.5), Choice("\n\n", 0.1)]
        if not prefix[-1].isalpha():
            return [Choice(" " + word, weight) for word, weight in zip(words, _WEIGHTS)]
        last = re.findall(r"[a-z]+", prefix.lower())[-1]
        if last in self._forced:
            return [Choice(" " + self._forced[last], 1.0)]
        return [
            *(Choice(" " + word, weight) for word, weight in zip(words, _WEIGHTS)),
            Choice(",", 0.08),
            Choice(".", 0.1),
            Choice("s", 0.07),
            Choice("�", 0.05),
            Choice(" rare", 0.001),
        ]


class ScriptedBackend:
    """Answers from a table keyed by the text so far."""

    def __init__(self, script: dict[str, list[Choice]], opening: str = "Start now") -> None:
        self._script = script
        self._opening = opening
        self.prefixes: list[str] = []

    def start(self, messages):
        return self._opening

    def next_tokens(self, messages, prefix):
        self.prefixes.append(prefix)
        return self._script.get(prefix, [Choice("", 1.0)])


def chain(prefix: str, *tokens: str) -> dict[str, list[Choice]]:
    """A script in which the model writes exactly ``tokens`` after ``prefix``."""
    script = {}
    for token in tokens:
        script[prefix] = [Choice(token, 1.0)]
        prefix += token
    return script
