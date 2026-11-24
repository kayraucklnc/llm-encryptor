import asyncio

import pytest

from llm_encryptor import peer
from llm_encryptor.crypto import derive_keys
from llm_encryptor.peer import MAX_TEXT_CHARS, Line, LineError, make_id, parse_id
from llm_encryptor.relay import Channel, relay_url


def test_an_id_is_an_address_that_is_easy_to_read_out():
    code = make_id("192.168.1.20", 47474)
    assert len(code) == 11 and code[5] == "-" and code == code.lower()
    assert parse_id(code) == ("192.168.1.20", 47474)
    assert parse_id(f"  {code.upper().replace('-', ' ')} ") == ("192.168.1.20", 47474)
    assert parse_id("localhost:5000") == ("localhost", 5000)


@pytest.mark.parametrize("bad", ["", "hello", "abcde-fghi", "host:port", "host:0", "!!!!!-!!!!!"])
def test_what_is_not_an_id_is_refused(bad):
    with pytest.raises(LineError):
        parse_id(bad)


def test_two_people_connect_and_text_both_ways():
    async def scene():
        got_a, got_b = [], []
        ada, bo = Line("Ada", got_a.append, model="qwen2.5:0.5b"), Line("Bo", got_b.append)
        port = parse_id(await ada.open(host="127.0.0.1"))[1]
        await bo.call(f"127.0.0.1:{port}")
        await asyncio.sleep(0.05)
        assert ada.connected and bo.connected and ada.friend == "Bo" and bo.friend == "Ada"
        assert bo.friend_model == "qwen2.5:0.5b" and ada.friend_model is None
        await ada.send("Hey, dinner on Friday?")
        await bo.send("Sounds great!")
        await asyncio.sleep(0.05)
        assert got_b == ["Hey, dinner on Friday?"] and got_a == ["Sounds great!"]
        await bo.close()
        await asyncio.sleep(0.05)
        assert not ada.connected
        with pytest.raises(LineError):
            await ada.send("anyone there?")
        await ada.close()

    asyncio.run(scene())


def test_nothing_the_other_side_sends_is_trusted():
    async def scene():
        got = []
        ada = Line("Ada", got.append)
        port = parse_id(await ada.open(host="127.0.0.1"))[1]
        _, writer = await asyncio.open_connection("127.0.0.1", port)
        for junk in (b"not json\n", b"[1, 2]\n", b'{"type": "text", "text": 5}\n', b'{"type": "text", "text": ""}\n',
                     b'{"type": "hello", "name": "' + b"x" * 500 + b'"}\n',
                     b'{"type": "text", "text": "' + b"y" * (MAX_TEXT_CHARS + 1) + b'"}\n',
                     b'{"type": "text", "text": "a real one"}\n'):  # fmt: skip
            writer.write(junk)
        await writer.drain()
        await asyncio.sleep(0.1)
        assert got == ["a real one"] and len(ada.friend) == 40
        # A second caller is turned away while the first is connected.
        reader2, _ = await asyncio.open_connection("127.0.0.1", port)
        assert await reader2.read() == b""
        writer.close()
        await ada.close()

    asyncio.run(scene())


def test_nobody_home_is_said_plainly():
    async def scene():
        with pytest.raises(LineError, match="Nobody answers"):
            await Line("Ada", print).call("127.0.0.1:9")

    asyncio.run(scene())


# --- when the two cannot reach each other directly --------------------------

async def out_of_reach(line: Line) -> str:
    """Wait like someone behind a phone's hotspot: the ID points nowhere that answers."""
    my_id = await line.open(host="127.0.0.1")
    line._server.close()
    return my_id


async def until(condition, seconds: float = 5.0) -> None:
    for _ in range(int(seconds / 0.02)):
        if condition():
            return
        await asyncio.sleep(0.02)
    raise AssertionError("never happened")


def test_two_people_on_different_networks_meet_through_the_relay(relay, keys):
    async def scene():
        got_a, got_b, said = [], [], []
        ada = Line("Ada", got_a.append, model="qwen2.5:0.5b", relay=relay.url)
        bo = Line("Bo", got_b.append, relay=relay.url)
        ada_id = await out_of_reach(ada)
        await ada.wait(keys)
        await bo.call(ada_id, keys, said.append)
        assert "relay" in said[0]
        await until(lambda: ada.connected and ada.friend == "Bo")
        assert bo.relayed and ada.relayed and bo.friend == "Ada" and bo.friend_model == "qwen2.5:0.5b"

        await ada.send("Hey, dinner on Friday? Çok güzel olur.")
        await bo.send("Sounds great!")
        await until(lambda: got_a and got_b)
        assert got_b == ["Hey, dinner on Friday? Çok güzel olur."] and got_a == ["Sounds great!"]
        # The relay sees sealed posts only: no text, no names, no model.
        seen = " ".join(post["message"] for post in relay.posts)
        assert not any(word in seen for word in ("Friday", "Sounds", "Ada", "Bo", "qwen", "hello"))

        await bo.close()
        await until(lambda: not ada.connected)
        await ada.close()

    asyncio.run(scene())


def test_the_relays_clock_is_the_one_that_counts(keys):
    from fake_relay import FakeRelay

    async def scene(relay):
        ada, bo = Line("Ada", print, relay=relay.url), Line("Bo", print, relay=relay.url)
        ada_id = await out_of_reach(ada)
        await ada.wait(keys)
        await bo.call(ada_id, keys)
        assert bo.connected
        await bo.close()
        await ada.close()

    relay = FakeRelay(clock_offset=-10**6)
    try:
        asyncio.run(scene(relay))
    finally:
        relay.close()


def test_a_caller_who_comes_first_is_answered_once_the_other_waits(relay, keys):
    async def scene():
        ada, bo = Line("Ada", print, relay=relay.url), Line("Bo", print, relay=relay.url)
        ada_id = await out_of_reach(ada)
        calling = asyncio.create_task(bo.call(ada_id, keys))
        await until(lambda: relay.posts)  # Bo knocked before Ada typed the password.
        await ada.wait(keys)
        await asyncio.wait_for(calling, 5)
        assert bo.connected and bo.friend == "Ada"
        await until(lambda: ada.connected)
        await bo.close()
        await ada.close()

    asyncio.run(scene())


def test_another_password_never_meets(relay, keys, monkeypatch):
    monkeypatch.setattr(peer, "_KNOCK_SECONDS", 0.2)
    monkeypatch.setattr(peer, "_CALL_SECONDS", 0.5)

    async def scene():
        ada, bo = Line("Ada", print, relay=relay.url), Line("Bo", print, relay=relay.url)
        ada_id = await out_of_reach(ada)
        await ada.wait(keys)
        with pytest.raises(LineError, match="same password"):
            await bo.call(ada_id, derive_keys("not the password"))
        assert not ada.connected and not bo.connected
        await ada.close()

    asyncio.run(scene())


def test_old_foreign_and_forged_posts_are_ignored(relay, keys):
    async def scene():
        ada = Line("Ada", print, relay=relay.url)
        bo, cy = Line("Bo", print, relay=relay.url), Line("Cy", print, relay=relay.url)
        ada_id = await out_of_reach(ada)
        channel = Channel.between(keys, ada._place)
        # A hello from an earlier chat in the same place, junk, and a forgery.
        relay.add(channel.topic, channel.seal({"type": "hello", "name": "Ghost", "s": "old"}), when=relay.now() - 3600)
        for junk in ("not sealed", channel.seal({"type": "hello", "name": "Nobody"})[:-4] + "AAAA",
                     Channel.between(derive_keys("other"), ada._place).seal({"type": "hello", "name": "X", "s": "x"}),
                     channel.seal({"type": "hello", "name": "No session"}), channel.seal(["not", "a", "dict"])):  # fmt: skip
            relay.add(channel.topic, junk)
        await ada.wait(keys)
        await asyncio.sleep(0.3)
        assert not ada.connected
        await bo.call(ada_id, keys)
        await until(lambda: ada.friend == "Bo")
        # A second caller gets no answer while the first is there.
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(cy.call(ada_id, keys), 1.0)
        assert ada.friend == "Bo" and not cy.connected
        await bo.close()
        await until(lambda: not ada.connected)
        await ada.close()
        await cy.close()

    asyncio.run(scene())


def test_a_relay_that_cannot_be_reached_is_said_plainly(keys, monkeypatch):
    monkeypatch.setattr(peer, "_REACH_SECONDS", 0.3)

    async def scene():
        ada = Line("Ada", print, relay="http://127.0.0.1:9")
        await out_of_reach(ada)
        with pytest.raises(LineError, match="relay could not be reached"):
            await ada.wait(keys)
        await ada.close()

    asyncio.run(scene())


def test_an_address_that_answers_but_is_not_the_app_is_not_a_friend(monkeypatch):
    monkeypatch.setattr(peer, "_DIRECT_SECONDS", 0.2)

    async def scene():
        silent = await asyncio.start_server(lambda reader, writer: None, "127.0.0.1", 0)
        port = silent.sockets[0].getsockname()[1]
        line = Line("Ada", print)
        with pytest.raises(LineError, match="Nobody answers"):
            await line.call(f"127.0.0.1:{port}")
        assert not line.connected
        silent.close()

    asyncio.run(scene())


def test_where_two_meet_depends_on_the_password_and_the_id(keys, other_keys):
    here = Channel.between(keys, "172.20.10.2:5000")
    assert here == Channel.between(keys, "172.20.10.2:5000")
    assert here.topic != Channel.between(other_keys, "172.20.10.2:5000").topic
    assert here.topic != Channel.between(keys, "172.20.10.2:5001").topic
    assert len(here.topic) == 24 and here.topic.isalnum()
    sealed = here.seal({"type": "text", "text": "hi"})
    assert here.unseal(sealed) == {"type": "text", "text": "hi"}
    assert Channel.between(other_keys, "172.20.10.2:5000").unseal(sealed) is None


def test_the_relay_can_be_chosen_or_turned_off(monkeypatch):
    monkeypatch.delenv("LLM_ENCRYPTOR_RELAY")
    assert relay_url() == "https://ntfy.sh"
    monkeypatch.setenv("LLM_ENCRYPTOR_RELAY", "https://ntfy.example.org/ ")
    assert relay_url() == "https://ntfy.example.org"
    assert relay_url("off") is None and Line("Ada", print).relay == "https://ntfy.example.org"
