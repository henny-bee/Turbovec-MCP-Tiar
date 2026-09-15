"""Retrieval telemetry: per-stage latency recording and rolling statistics.

Recording happens on the hot path of every search, so the connection is
created once per thread instead of once per event, and statistics are computed
over a bounded window of the most recent rows rather than the whole table.
"""

from __future__ import annotations

import datetime
import logging
import sqlite3
import threading
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["SearchTelemetry"]

_EMPTY_STATS: Dict[str, Any] = {
    "total_searches": 0,
    "latency_p50": 0.0,
    "latency_p95": 0.0,
    "latency_p99": 0.0,
    "avg_total_latency": 0.0,
    "avg_fts_latency": 0.0,
    "avg_vector_latency": 0.0,
    "avg_graph_latency": 0.0,
    "avg_embedding_latency": 0.0,
    "avg_reranker_latency": 0.0,
    "avg_candidates": 0.0,
    "avg_results": 0.0,
    "error_rate": 0.0,
    "cache_hit_rate": 0.0,
}

_INSERT = """
    INSERT INTO search_metrics (
        timestamp, total_latency, fts_latency, vector_latency,
        graph_latency, embedding_latency, reranker_latency,
        candidate_count, result_count, is_error, error_code, cache_hit
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_CREATE_TABLE = """
    CREATE TABLE IF NOT EXISTS search_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        total_latency REAL NOT NULL,
        fts_latency REAL NOT NULL,
        vector_latency REAL NOT NULL,
        graph_latency REAL NOT NULL,
        embedding_latency REAL NOT NULL,
        reranker_latency REAL NOT NULL,
        candidate_count INTEGER NOT NULL,
        result_count INTEGER NOT NULL,
        is_error INTEGER NOT NULL,
        error_code TEXT,
        cache_hit INTEGER DEFAULT 0
    );
"""


class SearchTelemetry:
    def __init__(
        self, db_file: str, enabled: bool = True, window: int = 10_000
    ) -> None:
        self.db_file = db_file
        self.enabled = enabled
        self.window = max(1, window)
        self._local = threading.local()
        self._init_metrics_table()

    # -- connection --------------------------------------------------------
    def _connection(self) -> Optional[sqlite3.Connection]:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            try:
                conn = sqlite3.connect(self.db_file, timeout=10.0)
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("PRAGMA synchronous=NORMAL;")
            except Exception as exc:
                logger.error(f"Failed to open telemetry database: {exc}")
                return None
            self._local.conn = conn
        return conn

    def _init_metrics_table(self) -> None:
        conn = self._connection()
        if conn is None:
            return
        try:
            with conn:
                conn.execute(_CREATE_TABLE)
        except Exception as exc:
            logger.error(f"Failed to initialize search_metrics table: {exc}")

    # -- writes ------------------------------------------------------------
    def record_search(
        self,
        total_latency: float,
        fts_latency: float,
        vector_latency: float,
        graph_latency: float,
        embedding_latency: float,
        reranker_latency: float,
        candidate_count: int,
        result_count: int,
        is_error: bool = False,
        error_code: str = None,
        cache_hit: bool = False,
    ) -> None:
        """Records one search event. Never raises into the retrieval path."""
        if not self.enabled:
            return
        conn = self._connection()
        if conn is None:
            return

        try:
            with conn:
                conn.execute(
                    _INSERT,
                    (
                        datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        total_latency,
                        fts_latency,
                        vector_latency,
                        graph_latency,
                        embedding_latency,
                        reranker_latency,
                        candidate_count,
                        result_count,
                        1 if is_error else 0,
                        error_code,
                        1 if cache_hit else 0,
                    ),
                )
        except Exception as exc:
            logger.error(f"Failed to record search telemetry: {exc}")

    # -- reads -------------------------------------------------------------
    def get_stats(self) -> Dict[str, Any]:
        """Percentiles and per-subsystem averages over the most recent window."""
        conn = self._connection()
        if conn is None:
            return {"error": f"Telemetry database unavailable: {self.db_file}"}

        try:
            rows: List[tuple] = conn.execute(
                """
                SELECT total_latency, fts_latency, vector_latency, graph_latency,
                       embedding_latency, reranker_latency, candidate_count,
                       result_count, is_error, cache_hit
                FROM search_metrics
                ORDER BY id DESC
                LIMIT ?
                """,
                (self.window,),
            ).fetchall()
        except Exception as exc:
            logger.error(f"Failed to fetch search metrics: {exc}")
            return {"error": str(exc)}

        if not rows:
            return dict(_EMPTY_STATS)

        matrix = np.asarray(rows, dtype=np.float64)
        totals = matrix[:, 0]

        return {
            "total_searches": matrix.shape[0],
            "latency_p50": float(np.percentile(totals, 50)),
            "latency_p95": float(np.percentile(totals, 95)),
            "latency_p99": float(np.percentile(totals, 99)),
            "avg_total_latency": float(totals.mean()),
            "avg_fts_latency": float(matrix[:, 1].mean()),
            "avg_vector_latency": float(matrix[:, 2].mean()),
            "avg_graph_latency": float(matrix[:, 3].mean()),
            "avg_embedding_latency": float(matrix[:, 4].mean()),
            "avg_reranker_latency": float(matrix[:, 5].mean()),
            "avg_candidates": float(matrix[:, 6].mean()),
            "avg_results": float(matrix[:, 7].mean()),
            "error_rate": float(matrix[:, 8].mean()),
            "cache_hit_rate": float(matrix[:, 9].mean()),
            "window": self.window,
        }
