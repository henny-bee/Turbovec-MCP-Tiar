"""Embedding provider contract."""

from __future__ import annotations

from typing import List, Protocol, Sequence


class EmbeddingProvider(Protocol):
    """Anything that can turn text into fixed-width vectors."""

    @property
    def dimension(self) -> int:
        """Returns the dimension of generated embeddings."""
        ...

    @property
    def model(self):
        """Returns the underlying model object."""
        ...

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        """Generates embeddings for a list of texts."""
        ...
