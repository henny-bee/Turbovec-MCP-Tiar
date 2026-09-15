"""Shared fixtures for the suite.

Every test file used to carry its own copy of the same temp-path and
SentenceTransformer fixtures; they live here once instead. ``make_encoder``
wraps a single-text encoder so it also accepts a batch, because the engine
embeds document chunks and radar candidates in one call.
"""

import numpy as np
import pytest
from unittest.mock import patch


def _deterministic_vector(text: str) -> np.ndarray:
    """A stable pseudo-random vector derived from the text itself."""
    np.random.seed(abs(hash(text)) % (2**32))
    return np.random.rand(384).astype(np.float32)


@pytest.fixture
def make_encoder():
    """Turns a ``str -> vector`` function into a batch-aware ``encode``."""

    def factory(encode_one):
        def encode(text, **kwargs):
            if isinstance(text, (list, tuple)):
                return np.asarray([encode_one(item) for item in text], dtype=np.float32)
            return encode_one(text)

        return encode

    return factory


@pytest.fixture
def mock_sentence_transformer(make_encoder):
    """Patches the embedding model with a deterministic, batch-aware stub."""
    with patch("embeddings.minilm.SentenceTransformer") as MockST:
        instance = MockST.return_value
        instance.encode.side_effect = make_encoder(_deterministic_vector)
        yield instance
