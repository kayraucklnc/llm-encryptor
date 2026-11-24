import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from llm_encryptor.crypto import derive_keys


@pytest.fixture(autouse=True)
def no_public_relay(monkeypatch):
    """Tests never reach ntfy.sh; the ones about the relay start their own."""
    monkeypatch.setenv("LLM_ENCRYPTOR_RELAY", "off")


@pytest.fixture
def relay():
    from fake_relay import FakeRelay

    running = FakeRelay()
    yield running
    running.close()


@pytest.fixture(scope="session")
def keys():
    return derive_keys("correct horse battery staple")


@pytest.fixture(scope="session")
def other_keys():
    return derive_keys("a different password")
