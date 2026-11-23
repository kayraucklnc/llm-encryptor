import asyncio
import itertools

import pytest
from fakes import FakeBackend
from rich.console import Console

from llm_encryptor import tui
from llm_encryptor.crypto import derive_keys
from llm_encryptor.dense import ReadProgress, read_dense, write_dense
from llm_encryptor.talk import DEFAULT_LENGTH, DEFAULT_TOPIC, LENGTHS, request
from llm_encryptor.tui import (
    Chat,
    EncryptorApp,
    LengthDial,
    Steps,
    StepsPanel,
    Turn,
    steps_view,
)


def shown(steps: Steps) -> str:
    console = Console(width=200, record=True, force_terminal=False)
    console.print(steps_view(steps))
    return console.export_text()


@pytest.fixture(autouse=True)
def no_clipboard(monkeypatch):
    monkeypatch.setattr(tui, "copy_text", lambda text: True)
    monkeypatch.setattr(tui, "read_clipboard", lambda: "pasted text")


def app():
    return EncryptorApp("toy", "none", lambda model, host: FakeBackend(seed=1))


async def fill(pilot, where, text):
    await pilot.click(where)
    await pilot.press(*text)


async def until(pilot, ready):
    for _ in range(400):
        await pilot.pause(0.02)
        if ready():
            return
    raise AssertionError("the app never got there")


def status(running) -> str:
    return str(running.screen.query_one("#status").render())


# --- what is asked of the model ---------------------------------------------


def test_the_first_visible_text_is_about_the_topic():
    assert request("  dinner   on Friday ") == [{"role": "user", "content": "dinner on Friday"}]
    assert request("") == [{"role": "user", "content": DEFAULT_TOPIC}]


def test_later_visible_texts_answer_the_one_before():
    asked = request("", "Hey,\n dinner on Friday?")[0]["content"]
    assert '"Hey, dinner on Friday?"' in asked and "reply" in asked and "round to" not in asked
    assert "Bring the reply round to: football." in request("football", "Hey!")[0]["content"]


def test_the_dial_goes_from_longest_to_shortest_and_starts_on_short():
    assert [density for density, _, _ in LENGTHS] == ["natural", "tight", "short", "max"]
    assert LENGTHS[DEFAULT_LENGTH][0] == "short"


def test_a_steer_decides_what_a_short_text_opens_on(keys):
    class TwoStarts(FakeBackend):
        asked = 0

        def start(self, messages, pieces=8):
            TwoStarts.asked += 1
            return ("Hey how is it going today" if TwoStarts.asked % 2 else "The match was calm and bright") + " now."

    asked = request("the football match", "Hey how is it going")
    text = write_dense("hi", keys, TwoStarts(seed=3), asked, density="short", about="the football match")
    assert text.startswith("The match was calm and")


# --- reading, step by step --------------------------------------------------


def test_the_reader_reports_each_word_it_follows(keys):
    backend = FakeBackend(seed=5)
    text = write_dense("top secret", keys, backend, request("the sea"), density="short")
    seen: list[ReadProgress] = []
    assert read_dense(text, keys, backend, seen.append) == "top secret"
    assert seen[-1].read == text and len(seen) > 5
    assert all(later.read.startswith(earlier.read) for earlier, later in itertools.pairwise(seen))
    assert len(seen[-1].bits) >= 20 and set(seen[-1].bits) <= {"0", "1"}


# --- the steps panel ----------------------------------------------------------


def test_the_stages_show_a_message_being_hidden():
    idle = shown(Steps())
    assert idle.index("secret") < idle.index("locked") < idle.index("visible") and "· · ·" in idle
    midway = shown(Steps(secret="meet at noon", text="Hey, how", done_bits=30, all_bits=60, draft=2))
    assert "meet at noon" in midway and "30/60" in midway and "×2" in midway and "Hey, how▌" in midway
    assert midway.count("▰") == midway.count("▱") > 0
    done = shown(Steps(secret="s", text="Hey, how are you.", done_bits=60, all_bits=60, draft=1, finished=True))
    assert "60/60" in done and "▱" not in done and "▌" not in done


def test_the_stages_show_a_message_being_read_the_other_way_up():
    midway = shown(Steps(hiding=False, text="Hey, how are you.", read="Hey, how", bits="0110"))
    assert midway.index("visible") < midway.index("locked") < midway.index("secret")
    assert "Hey, how are you." in midway and "▱▰▰▱  4" in midway and "· · ·" in midway
    done = shown(Steps(hiding=False, text="Hey.", read="Hey.", bits="01", secret="noon", finished=True))
    assert "noon" in done and "· · ·" not in done
    assert "✗ wrong password" in shown(Steps(hiding=False, text="Hey.", problem="wrong password"))


# --- the app ------------------------------------------------------------------


def test_a_message_is_hidden_and_revealed_in_the_app(keys):
    async def scene():
        running = app()
        async with running.run_test(size=(120, 44)) as pilot:
            assert "Chat" in str(running.screen.query_one("#chat").label)
            await pilot.click("#hide")
            await pilot.click("#go")
            assert "secret and password needed" in status(running)
            await fill(pilot, "#secret", "meet at noon")
            await fill(pilot, "#password", "correct horse battery staple")
            await fill(pilot, "#about", "the sea")
            await pilot.press("enter")
            await until(pilot, lambda: "✓" in status(running))
            steps = running.screen.query_one(StepsPanel).steps
            assert steps.finished and steps.done_bits == steps.all_bits == 60
            assert read_dense(steps.text, keys, FakeBackend(seed=1)) == "meet at noon"
            await pilot.press("escape")
            await pilot.click("#reveal")
            running.screen.query_one("#cover").text = steps.text
            await fill(pilot, "#password", "correct horse battery staple")
            await pilot.click("#go")
            await until(pilot, lambda: "✓" in status(running))
            assert "✓ meet at noon" in status(running)
            assert running.screen.query_one(StepsPanel).steps.secret == "meet at noon"
            # A wrong password is said plainly.
            await pilot.click("#password")
            await pilot.press("x", "enter")
            await until(pilot, lambda: "No hidden message found" in status(running))

    asyncio.run(scene())


def test_the_dial_moves_with_the_arrow_keys():
    async def scene():
        running = app()
        async with running.run_test(size=(120, 44)) as pilot:
            await pilot.click("#hide")
            dial = running.screen.query_one(LengthDial)
            dial.focus()
            assert dial.density == "short"
            await pilot.press("left", "left", "left", "left")
            assert dial.density == "natural"
            await pilot.press("right", "right", "right", "right", "right")
            assert dial.density == "max" and "garbled" in str(dial.render())

    asyncio.run(scene())


def test_two_people_chat_and_each_sees_both_conversations():
    async def scene():
        ada, bo = app(), app()
        async with ada.run_test(size=(120, 44)) as a, bo.run_test(size=(120, 44)) as b:
            await a.click("#chat")
            await b.click("#chat")
            await until(a, lambda: ada.screen.line._server is not None)
            port = ada.screen.line._server.sockets[0].getsockname()[1]
            await a.click("#go")
            assert "password needed" in status(ada)
            await fill(a, "#name", "Ada")
            await fill(a, "#password", "pw")
            await a.click("#go")
            assert "waiting for your friend" in status(ada)
            await fill(b, "#name", "Bo")
            await fill(b, "#password", "pw")
            await fill(b, "#friend", f"127.0.0.1:{port}")
            await b.click("#go")
            await until(a, lambda: isinstance(ada.screen, Chat) and isinstance(bo.screen, Chat))
            await b.pause(0.1)
            assert "● Bo" in status(ada) and "● Ada" in status(bo)
            assert ada.screen.query_one("#about").placeholder == "dinner on Friday"

            turns = lambda running: list(running.screen.query(Turn))
            await fill(a, "#secret", "meet at noon")
            await fill(a, "#about", "the sea")
            await a.press("enter")
            await until(b, lambda: turns(bo) and turns(bo)[-1].secret == "meet at noon")
            await a.pause(0.1)
            sent, got = turns(ada)[0], turns(bo)[0]
            assert sent.who == "you" and got.who == "Ada" and sent.cover == got.cover != "…"
            assert "meet at noon" not in got.cover and bo.screen.last_from_friend == got.cover
            assert "replies to their last text" in bo.screen.query_one("#about").placeholder

            await fill(b, "#secret", "ok, bring the keys")
            await b.press("enter")
            await until(a, lambda: len(turns(ada)) == 2 and turns(ada)[-1].secret == "ok, bring the keys")
            assert [turn.who for turn in turns(ada)] == ["you", "Bo"]
            assert ada.screen.query_one(StepsPanel).steps.secret == "ok, bring the keys"

            await b.press("escape", "escape")
            await until(a, lambda: "Bo left" in status(ada))
            await fill(a, "#secret", "hello?")
            await a.press("enter")
            assert "not connected" in status(ada)

    asyncio.run(scene())


def test_a_text_hidden_with_another_password_shows_as_unreadable():
    async def scene():
        running = app()
        async with running.run_test(size=(120, 44)) as pilot:
            other = write_dense("hi", derive_keys("theirs"), FakeBackend(seed=1), request("the sea"), density="short")
            chat = Chat(tui.Line("Me", print), derive_keys("mine"), [other])
            chat.line.friend = "Bo"
            await running.push_screen(chat)
            await until(pilot, lambda: any("unreadable" in turn.secret for turn in chat.query(Turn)))

    asyncio.run(scene())
