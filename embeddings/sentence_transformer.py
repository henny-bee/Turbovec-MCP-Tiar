"""Shared implementation for every sentence-transformers backed provider."""

from __future__ import annotations

import logging
from typing import List, Sequence

import numpy as np

from memory.errors import EMBEDDING_FAILED, SearchError

logger = logging.getLogger(__name__)

__all__ = ["SentenceTransformerProvider"]


class SentenceTransformerProvider:
    """Loads a sentence-transformers model and exposes the provider contract.

    Subclasses pass the transformer class explicitly so that the symbol stays
    resolvable (and patchable) in the subclass' own module.
    """

    def __init__(self, model_name: str, transformer_cls, dimension: int) -> None:
        self.model_name = model_name
        self._dimension = dimension
        logger.info(f"Initializing SentenceTransformer: {model_name}")
        try:
            self._model = transformer_cls(model_name)
        except Exception as exc:
            raise SearchError(
                code=EMBEDDING_FAILED,
                message=f"Failed to load embedding model {model_name}: {exc}",
                subsystem="EMBEDDING",
                retry_safe=True,
            )

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model(self):
        return self._model

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        try:
            embeddings = self._model.encode(list(texts))
            if isinstance(embeddings, np.ndarray):
                return embeddings.tolist()
            return embeddings
        except Exception as exc:
            logger.error(f"Embedding failed using {self.model_name}: {exc}")
            raise SearchError(
                code=EMBEDDING_FAILED,
                message=f"Embedding generation failed: {exc}",
                subsystem="EMBEDDING",
                retry_safe=True,
            )
