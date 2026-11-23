import asyncio

import pytest

from llm_encryptor.peer import MAX_TEXT_CHARS, Line, LineError, make_id, parse_id


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
