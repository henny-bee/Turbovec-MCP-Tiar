"""Caching, batching facade in front of an :class:`EmbeddingProvider`.

Two things dominate embedding cost in this system: re-encoding the same query
text on every search, and encoding long documents one chunk at a time. The
service fixes both with a bounded LRU cache and true batch encoding.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from typing import List, Sequence

import numpy as np

from memory.errors import EMBEDDING_FAILED, SearchError

logger = logging.getLogger(__name__)

__all__ = ["EmbeddingService"]

# Texts longer than this are not worth keeping in memory; they are also the
# least likely to be requested twice.
MAX_CACHEABLE_CHARS = 4096


class EmbeddingService:
    def __init__(
        self,
        provider,
        cache_size: int = 512,
        batch_size: int = 64,
    ) -> None:
        self.provider = provider
        self.cache_size = max(0, cache_size)
        self.batch_size = max(1, batch_size)
        self._cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    @property
    def model(self):
        """The raw model; kept public for callers that need direct access."""
        return self.provider.model

    @property
    def dimension(self) -> int:
        return self.provider.dimension

    # -- encoding ----------------------------------------------------------
    def encode(self, text: str) -> np.ndarray:
        """Encodes a single string, serving repeats from the cache."""
        cached = self._cache_get(text)
        if cached is not None:
            return cached
        vector = self._encode_raw(text)
        self._cache_put(text, vector)
        return vector

    def encode_many(self, texts: Sequence[str]) -> List[np.ndarray]:
        """Encodes many strings, batching every cache miss into one call."""
        if not texts:
            return []

        results: List[np.ndarray] = [None] * len(texts)  # type: ignore[list-item]
        missing_positions: List[int] = []
        for position, text in enumerate(texts):
            cached = self._cache_get(text)
            if cached is None:
                missing_positions.append(position)
            else:
                results[position] = cached

        for start in range(0, len(missing_positions), self.batch_size):
            batch_positions = missing_positions[start : start + self.batch_size]
            batch_texts = [texts[position] for position in batch_positions]
            vectors = self._encode_raw_batch(batch_texts)
            for position, text, vector in zip(batch_positions, batch_texts, vectors):
                results[position] = vector
                self._cache_put(text, vector)

        return results

    def encode_matrix(self, texts: Sequence[str]) -> np.ndarray:
        """Encodes many strings into a single ``(n, dim)`` float32 matrix."""
        vectors = self.encode_many(texts)
        if not vectors:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return np.asarray(vectors, dtype=np.float32)

    # -- internals ---------------------------------------------------------
    def _encode_raw(self, text: str) -> np.ndarray:
        try:
            return np.asarray(self.model.encode(text), dtype=np.float32)
        except SearchError:
            raise
        except Exception as exc:
            raise SearchError(
                code=EMBEDDING_FAILED,
                message=f"Embedding model failure: {exc}",
                subsystem="EMBEDDING",
                retry_safe=True,
            )

    def _encode_raw_batch(self, texts: Sequence[str]) -> List[np.ndarray]:
        try:
            encoded = self.model.encode(list(texts))
        except SearchError:
            raise
        except Exception as exc:
            raise SearchError(
                code=EMBEDDING_FAILED,
                message=f"Embedding model failure: {exc}",
                subsystem="EMBEDDING",
                retry_safe=True,
            )
        matrix = np.asarray(encoded, dtype=np.float32)
        if matrix.ndim == 1:  # provider collapsed a single-item batch
            matrix = matrix.reshape(1, -1)
        return [np.asarray(row, dtype=np.float32) for row in matrix]

    def _cache_get(self, text: str):
        if not self.cache_size or len(text) > MAX_CACHEABLE_CHARS:
            return None
        with self._lock:
            vector = self._cache.get(text)
            if vector is None:
                self.misses += 1
                return None
            self._cache.move_to_end(text)
            self.hits += 1
            return vector

    def _cache_put(self, text: str, vector: np.ndarray) -> None:
        if not self.cache_size or len(text) > MAX_CACHEABLE_CHARS:
            return
        with self._lock:
            self._cache[text] = vector
            self._cache.move_to_end(text)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)

    def clear_cache(self) -> None:
        with self._lock:
            self._cache.clear()

    def cache_stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "size": len(self._cache),
            "capacity": self.cache_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": (self.hits / total) if total else 0.0,
        }
