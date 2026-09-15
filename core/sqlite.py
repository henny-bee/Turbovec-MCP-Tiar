"""Thread-safe SQLite connection management with performance pragmas."""

from __future__ import annotations

import logging
import sqlite3
import threading
from contextlib import contextmanager
from typing import Iterable, Iterator, List, Sequence, TypeVar

from core.config import Settings

logger = logging.getLogger(__name__)

T = TypeVar("T")

__all__ = ["ConnectionManager", "chunked"]


def chunked(items: Sequence[T], size: int) -> Iterator[List[T]]:
    """Yields ``items`` in slices of at most ``size`` elements.

    Used to keep ``IN (...)`` clauses below SQLite's bound-variable ceiling so
    queries stay correct no matter how large the graph grows.
    """
    if size <= 0:
        raise ValueError("size must be greater than zero")
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


class ConnectionManager:
    """Hands out one tuned :class:`sqlite3.Connection` per thread.

    SQLite connections are not shareable between threads, so each thread lazily
    creates its own. Every connection gets the same pragma set: WAL journaling
    for concurrent readers, a large page cache and memory-mapped I/O for read
    throughput, and foreign keys for referential integrity.
    """

    def __init__(self, db_file: str, settings: Settings) -> None:
        self.db_file = db_file
        self.settings = settings
        self._local = threading.local()
        self._lock = threading.Lock()
        self._connections: List[sqlite3.Connection] = []

    # -- lifecycle ---------------------------------------------------------
    @property
    def connection(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._connect()
            self._local.conn = conn
            with self._lock:
                self._connections.append(conn)
        return conn

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_file, timeout=self.settings.sqlite_timeout)
        self._apply_pragmas(conn)
        return conn

    def _apply_pragmas(self, conn: sqlite3.Connection) -> None:
        settings = self.settings
        pragmas = (
            "PRAGMA journal_mode=WAL;",
            "PRAGMA synchronous=NORMAL;",
            "PRAGMA foreign_keys=ON;",
            "PRAGMA temp_store=MEMORY;",
            f"PRAGMA busy_timeout={settings.sqlite_busy_timeout_ms};",
            f"PRAGMA cache_size=-{max(settings.sqlite_cache_kb, 0)};",
            f"PRAGMA mmap_size={max(settings.sqlite_mmap_bytes, 0)};",
        )
        for pragma in pragmas:
            try:
                conn.execute(pragma)
            except sqlite3.Error as exc:  # pragma support varies per build
                logger.debug(f"Pragma rejected ({pragma.strip()}): {exc}")

    def close_all(self) -> None:
        """Closes every connection handed out so far (best effort)."""
        with self._lock:
            connections, self._connections = self._connections, []
        for conn in connections:
            try:
                conn.close()
            except Exception:  # owned by another thread, or already closed
                pass
        self._local.conn = None

    # -- helpers -----------------------------------------------------------
    def cursor(self) -> sqlite3.Cursor:
        return self.connection.cursor()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Commits on success, rolls back on any exception."""
        conn = self.connection
        with conn:
            yield conn

    def executemany(self, sql: str, rows: Iterable[Sequence]) -> None:
        with self.transaction() as conn:
            conn.executemany(sql, rows)

    def scalar(self, sql: str, params: Sequence = ()) -> object:
        row = self.connection.execute(sql, params).fetchone()
        return row[0] if row else None
