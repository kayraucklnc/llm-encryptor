import importlib
import random
import textwrap

import pytest
from fakes import FakeBackend, ScriptedBackend, chain

from llm_encryptor.backend import Choice
from llm_encryptor.codec import MAX_LEAD_WORDS, RevealError, reveal_with_keys
from llm_encryptor.crypto import MAX_SECRET_BYTES
from llm_encryptor.embed import EmbedConfig, EmbedError, Progress, _Model, embed
from llm_encryptor.hide import (
    hide,
    payload_bit_count,
    system_message,
    validate_secret,
    words_needed,
)
from llm_encryptor.words import BITS_PER_BLOCK, WORDS_PER_BLOCK, block_bits, split_words

ASK = [{"role": "user", "content": "Write about the sea."}]


@pytest.mark.parametrize("secret", ["hi", "Meet me at the old bridge at 9pm.", "şifre: çay 🫖 123"])
def test_hidden_secret_is_recovered(keys, secret):
    cover = hide(secret, keys, FakeBackend(seed=1), ASK)
    assert reveal_with_keys(cover, keys) == secret


def test_wrong_password_recovers_nothing(keys, other_keys):
    cover = hide("top secret", keys, FakeBackend(seed=2), ASK)
    with pytest.raises(RevealError):
        reveal_with_keys(cover, other_keys)


def test_changed_words_never_give_a_wrong_secret(keys):
    cover = hide("top secret", keys, FakeBackend(seed=3), ASK)
    words = cover.split(" ")
    for index in range(4, 64, 3):
        words[index] = f"tampered{index}"
    with pytest.raises(RevealError):
        reveal_with_keys(" ".join(words), keys)


def test_cover_survives_reformatting(keys):
    cover = hide("top secret", keys, FakeBackend(seed=4), ASK)
    mangled = textwrap.fill(cover.upper().replace(",", ";").replace(".", " !"), width=40)
    assert reveal_with_keys(f"  {mangled}\n\n", keys) == "top secret"


def test_text_before_and_after_the_cover_is_tolerated(keys):
    cover = hide("top secret", keys, FakeBackend(seed=5), ASK)
    wrapped = f"Hi Sam, here is that piece I mentioned:\n\n{cover}\n\nTalk soon, K."
    assert reveal_with_keys(wrapped, keys) == "top secret"


def test_too_much_text_before_the_cover_hides_it(keys):
    cover = hide("top secret", keys, FakeBackend(seed=5), ASK)
    with pytest.raises(RevealError):
        reveal_with_keys("filler " * (MAX_LEAD_WORDS + 1) + cover, keys)


def test_each_run_writes_a_different_cover(keys):
    first = hide("top secret", keys, FakeBackend(seed=6), ASK)
    second = hide("top secret", keys, FakeBackend(seed=6), ASK)
    assert first != second


def test_cover_length_matches_estimate(keys):
    secret = "top secret"
    cover = hide(secret, keys, FakeBackend(seed=8), ASK)
    assert words_needed(secret) <= len(split_words(cover)) <= words_needed(secret) + 160


def test_words_needed_counts_whole_blocks():
    blocks = -(-payload_bit_count("hello") // BITS_PER_BLOCK)
    assert words_needed("hello") == blocks * WORDS_PER_BLOCK + 1


def test_cover_ends_with_a_finished_sentence(keys):
    cover = hide("top secret", keys, FakeBackend(seed=9), ASK)
    assert cover.endswith(".")


def test_stretches_without_freedom_are_worked_around(keys):
    forced = {"sea": "was", "tea": "sharing", "the": "sea", "old": "ships", "ships": "moved"}
    cover = hide("a longer secret with many chances", keys, FakeBackend(seed=10, forced=forced), ASK)
    assert reveal_with_keys(cover, keys) == "a longer secret with many chances"


def test_progress_is_reported(keys):
    seen: list[Progress] = []
    hide("hi", keys, FakeBackend(seed=11), ASK, on_progress=seen.append)
    assert seen[0].embedded == 0
    assert seen[-1].embedded == seen[-1].total == payload_bit_count("hi")


def test_system_message_asks_for_enough_words():
    assert str(int(words_needed("hello") * 1.1) + 20) in system_message("hello")["content"]


@pytest.mark.parametrize("secret", ["", "x" * (MAX_SECRET_BYTES + 1)])
def test_bad_secrets_are_refused(keys, secret):
    with pytest.raises(ValueError):
        validate_secret(secret)
    with pytest.raises(ValueError):
        hide(secret, keys, FakeBackend(), ASK)


def test_hide_fails_when_the_cover_does_not_decode(keys, monkeypatch):
    hide_module = importlib.import_module("llm_encryptor.hide")
    monkeypatch.setattr(hide_module, "embed", lambda *args, **kwargs: "not the right text")
    with pytest.raises(EmbedError):
        hide("top secret", keys, FakeBackend(), ASK)


# --- the embedding search, step by step ----------------------------------


def _block(keys, bits, tag):
    """Three made-up words whose block carries ``bits``."""
    for index in range(2000):
        words = [f"{tag}{index}", "and", "so"]
        if block_bits(keys.mac, words) == bits:
            return words
    raise AssertionError("no block found")


def _spoken(words):
    return "".join(f" {word}" for word in words)


def test_a_dead_end_is_backed_out_of(keys):
    """Two wordings fit the first block, but only one of them can be continued."""
    trap, good = _block(keys, (1, 0), "trap"), _block(keys, (1, 0), "good")
    after_trap, after_good = _block(keys, (0, 0), "no"), _block(keys, (1, 1), "yes")
    script = {
        "Start": [Choice(f" {trap[0]}", 0.5), Choice(f" {good[0]}", 0.5)],
        **chain(f"Start {trap[0]}", " and", " so", "."),
        **chain(f"Start {good[0]}", " and", " so", "."),
        **chain("Start" + _spoken(trap), *(f" {word}" for word in after_trap), "."),
        **chain("Start" + _spoken(good), *(f" {word}" for word in after_good), "."),
        **chain("Start" + _spoken(good) + _spoken(after_good), "."),
    }
    visited_trap = False
    for seed in range(8):
        backend = ScriptedBackend(script)
        text = embed(backend, [], (1, 0, 1, 1), keys.mac, rng=random.Random(seed))
        assert text == "Start" + _spoken(good) + _spoken(after_good) + "."
        visited_trap = visited_trap or ("Start" + _spoken(trap)) in backend.prefixes
    assert visited_trap


def test_a_token_that_would_extend_the_last_word_is_not_used(keys):
    block = _block(keys, (0, 1), "w")
    script = {
        "Start": [Choice("ing", 0.9), Choice(f" {block[0]}", 0.1)],
        **chain(f"Start {block[0]}", " and", " so", "."),
    }
    text = embed(ScriptedBackend(script), [], (0, 1), keys.mac)
    assert text == "Start" + _spoken(block) + "."


def test_a_new_opening_is_tried_when_the_first_leads_nowhere(keys):
    block = _block(keys, (1, 1), "w")
    wrong = _block(keys, (0, 0), "x")

    class TwoOpenings(ScriptedBackend):
        openings = iter(["Bad start", "Good start"])

        def start(self, messages):
            return next(self.openings)

    script = {
        **chain("Bad", *(f" {word}" for word in wrong), "."),
        **chain("Good", *(f" {word}" for word in block), "."),
    }
    assert embed(TwoOpenings(script), [], (1, 1), keys.mac) == "Good" + _spoken(block) + "."


def test_a_model_that_only_wants_to_stop_is_an_error(keys):
    with pytest.raises(EmbedError, match="could not fit"):
        embed(ScriptedBackend({}), [], (0, 1), keys.mac, EmbedConfig(lookups_per_block=5))


def test_a_silent_model_is_an_error(keys):
    with pytest.raises(EmbedError, match="did not start"):
        embed(ScriptedBackend({}, opening="..."), [], (0, 1), keys.mac)


def test_an_endless_tail_is_cut_at_a_sentence(keys):
    block = _block(keys, (0, 1), "w")
    done = "Start" + _spoken(block)
    script = {
        **chain("Start", *(f" {word}" for word in block), ","),
        **chain(done, ", more.", " And then", " on", " and", " on", " and"),
    }
    config = EmbedConfig(tail_tokens=5)
    assert embed(ScriptedBackend(script), [], (0, 1), keys.mac, config) == done + ", more."


def test_a_tail_without_a_sentence_end_stops_at_a_word(keys):
    block = _block(keys, (0, 1), "w")
    done = "Start" + _spoken(block)
    script = {
        **chain("Start", *(f" {word}" for word in block), ","),
        **chain(done, " and then", " some mo", "re"),
    }
    config = EmbedConfig(tail_tokens=2)
    assert embed(ScriptedBackend(script), [], (0, 1), keys.mac, config) == done + " and then some."


def test_unlikely_and_unusable_tokens_are_dropped():
    choices = [Choice(" a", 0.9), Choice(" b", 0.08), Choice(" c", 0.01), Choice("�", 0.5),
               Choice("́", 0.5), Choice("", 0.5), Choice(" d", 0.0)]
    model = _Model(ScriptedBackend({"x": choices}), [], EmbedConfig())
    assert [text for text, _ in model.choices("x")] == [" a", " b"]
    assert sum(probability for _, probability in model.choices("x")) == pytest.approx(1.0)
    assert "" in [text for text, _ in model.choices("x", allow_end=True)]


def test_the_likeliest_token_is_kept_even_below_the_floor():
    choices = [Choice(f" w{i}", 0.04) for i in range(20)]
    model = _Model(ScriptedBackend({"x": choices}), [], EmbedConfig(min_probability=0.5))
    assert len(model.choices("x")) == 1


def test_warming_up_flattens_the_choices():
    choices = [Choice(" a", 0.9), Choice(" b", 0.1)]
    model = _Model(ScriptedBackend({"x": choices}), [], EmbedConfig())
    cool = dict(model.choices("x"))[" b"]
    model.warm_up(4)
    warm = dict(model.choices("x"))[" b"]
    model.warm_up(1000)
    hottest = dict(model.choices("x"))[" b"]
    assert cool < warm < hottest < 0.5
    model.warm_up(0)
    assert dict(model.choices("x"))[" b"] == pytest.approx(cool)
