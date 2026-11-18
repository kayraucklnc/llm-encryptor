import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from llm_encryptor.crypto import derive_keys


@pytest.fixture(scope="session")
def keys():
    return derive_keys("correct horse battery staple")


@pytest.fixture(scope="session")
def other_keys():
    return derive_keys("a different password")
