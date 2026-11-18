"""Hide an encrypted message inside ordinary-looking LLM text."""

from .codec import RevealError, reveal
from .crypto import Keys, derive_keys
from .dense import read_dense, write_dense
from .embed import EmbedConfig, EmbedError
from .hide import hide

__all__ = ["EmbedConfig", "EmbedError", "Keys", "RevealError", "derive_keys", "hide", "read_dense", "reveal", "write_dense"]
__version__ = "0.1.0"
