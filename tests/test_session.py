import pytest
from fakes import FakeBackend

from llm_encryptor.codec import reveal_with_keys
from llm_encryptor.session import Session


def test_every_revision_still_carries_the_secret(keys):
    session = Session("the eagle lands at dawn", keys, FakeBackend(seed=1))
    first = session.instruct("Write about Turkey.")
    second = first.instruct("Make it more friendly.")
    assert first.cover != second.cover
    assert reveal_with_keys(first.cover, keys) == "the eagle lands at dawn"
    assert reveal_with_keys(second.cover, keys) == "the eagle lands at dawn"


def test_the_model_sees_the_earlier_version_when_revising(keys):
    seen = []

    class Spy(FakeBackend):
        def next_tokens(self, messages, prefix):
            seen.append(list(messages))
            return super().next_tokens(messages, prefix)

    first = Session("hi", keys, Spy(seed=2)).instruct("Write about Turkey.")
    first.instruct("Make it more friendly.")
    roles = [message["role"] for message in seen[-1]]
    assert roles == ["system", "user", "assistant", "user"]
    assert seen[-1][2]["content"] == first.cover
    assert seen[-1][3]["content"] == "Make it more friendly."


def test_sessions_are_not_changed_in_place(keys):
    session = Session("hi", keys, FakeBackend(seed=3))
    session.instruct("Write about Turkey.")
    assert session.cover is None
    assert session.conversation == ()


def test_secret_can_be_replaced(keys):
    session = Session("hi", keys, FakeBackend(seed=4)).with_secret("new secret")
    assert reveal_with_keys(session.instruct("Write.").cover, keys) == "new secret"
    with pytest.raises(ValueError):
        session.with_secret("")


def test_restart_forgets_the_conversation(keys):
    session = Session("hi", keys, FakeBackend(seed=5)).instruct("Write.").restarted()
    assert session.conversation == ()
    assert session.cover is None
    assert session.secret == "hi"
