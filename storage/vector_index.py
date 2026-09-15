"""Thread-safe wrapper around the turbovec index with debounced persistence.

Writing the index to disk after every single mutation dominates the cost of
graph writes. This wrapper keeps the in-memory index authoritative and flushes
it either after ``flush_ops`` mutations or ``flush_interval`` seconds,
whichever comes first, plus explicitly on save and shutdown.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Optional

import numpy as np
import turbovec

logger = logging.getLogger(__name__)

__all__ = ["VectorIndex"]


class VectorIndex:
    def __init__(
        self,
        dimension: int,
        path: str,
        flush_ops: int = 64,
        flush_interval: float = 2.0,
    ) -> None:
        self.dimension = dimension
        self.path = path
        self.flush_ops = max(1, flush_ops)
        self.flush_interval = max(0.0, flush_interval)
        self._lock = threading.RLock()
        self._index = turbovec.IdMapIndex(dimension)
        self._pending = 0
        self._last_flush = time.monotonic()

    # -- introspection -----------------------------------------------------
    @property
    def raw(self):
        """The underlying turbovec index (escape hatch for advanced use)."""
        return self._index

    def __len__(self) -> int:
        with self._lock:
            return len(self._index)

    def __contains__(self, vector_id: int) -> bool:
        with self._lock:
            return vector_id in self._index

    # -- mutation ----------------------------------------------------------
    def add_with_ids(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        with self._lock:
            self._index.add_with_ids(
                np.asarray(vectors, dtype=np.float32),
                np.asarray(ids, dtype=np.uint64),
            )
            self._touch(len(ids) if hasattr(ids, "__len__") else 1)

    def add_one(self, vector, vector_id: int) -> None:
        self.add_with_ids(np.asarray([vector], dtype=np.float32), [vector_id])

    def remove(self, vector_id: int) -> bool:
        with self._lock:
            removed = self._index.remove(vector_id)
            self._touch(1)
            return removed

    def replace(self, vector, vector_id: int) -> None:
        """Atomically swaps the vector stored under ``vector_id``."""
        with self._lock:
            self._index.remove(vector_id)
            self._index.add_with_ids(
                np.asarray([vector], dtype=np.float32),
                np.asarray([vector_id], dtype=np.uint64),
            )
            self._touch(1)

    def reset(self) -> None:
        with self._lock:
            self._index = turbovec.IdMapIndex(self.dimension)
            self._pending = 0
            self._last_flush = time.monotonic()

    # -- query -------------------------------------------------------------
    def search(self, queries: np.ndarray, k: int):
        with self._lock:
            return self._index.search(np.asarray(queries, dtype=np.float32), k=k)

    # -- persistence -------------------------------------------------------
    def load(self) -> bool:
        """Loads the index from ``path``; falls back to an empty index."""
        if not os.path.exists(self.path):
            return False
        try:
            with self._lock:
                self._index = turbovec.IdMapIndex.load(self.path)
                self._pending = 0
                self._last_flush = time.monotonic()
            logger.info(f"Loaded index from {self.path}")
            return True
        except Exception as exc:
            logger.error(f"Failed to load index from {self.path}: {exc}", exc_info=True)
            self.reset()
            return False

    def write(self, path: Optional[str] = None) -> None:
        """Persists the index immediately, raising on failure."""
        target = path or self.path
        with self._lock:
            self._index.write(target)
            self._pending = 0
            self._last_flush = time.monotonic()

    def flush(self, force: bool = False) -> bool:
        """Writes to disk when dirty. Returns True when a write happened."""
        with self._lock:
            if not force and self._pending == 0:
                return False
            try:
                self.write()
                return True
            except Exception as exc:
                logger.error(
                    f"Failed to save index to {self.path}: {exc}", exc_info=True
                )
                return False

    def _touch(self, operations: int) -> None:
        """Records pending mutations and flushes once a threshold is crossed."""
        self._pending += operations
        if self._pending >= self.flush_ops or (
            time.monotonic() - self._last_flush >= self.flush_interval
        ):
            try:
                self.write()
            except Exception as exc:
                logger.error(f"Deferred index flush failed: {exc}", exc_info=True)
