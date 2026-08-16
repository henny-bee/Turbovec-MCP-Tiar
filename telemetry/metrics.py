import time
import sqlite3
import json
import logging
from typing import Dict, Any, List
import numpy as np

logger = logging.getLogger(__name__)


class SearchTelemetry:
    def __init__(self, db_file: str):
        self.db_file = db_file
        self._init_metrics_table()

    def _init_metrics_table(self) -> None:
        """Initializes the search_metrics table in SQLite."""
        try:
            conn = sqlite3.connect(self.db_file)
            with conn:
                conn.execute("""
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
                    """)
            conn.close()
        except Exception as e:
            logger.error(f"Failed to initialize search_metrics table: {e}")

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
        """Records a single search event to the database."""
        import datetime

        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        try:
            conn = sqlite3.connect(self.db_file)
            with conn:
                conn.execute(
                    """
                    INSERT INTO search_metrics (
                        timestamp, total_latency, fts_latency, vector_latency,
                        graph_latency, embedding_latency, reranker_latency,
                        candidate_count, result_count, is_error, error_code, cache_hit
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        timestamp,
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
            conn.close()
        except Exception as e:
            logger.error(f"Failed to record search telemetry: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Calculates rolling statistics, percentiles, and sub-system latency analysis."""
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT total_latency, fts_latency, vector_latency, graph_latency,
                       embedding_latency, reranker_latency, candidate_count, result_count,
                       is_error, cache_hit
                FROM search_metrics
                """)
            rows = cursor.fetchall()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to fetch search metrics: {e}")
            return {"error": str(e)}

        if not rows:
            return {
                "total_searches": 0,
                "latency_p50": 0.0,
                "latency_p95": 0.0,
                "latency_p99": 0.0,
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

        total_latencies = [r[0] for r in rows]
        fts_latencies = [r[1] for r in rows]
        vector_latencies = [r[2] for r in rows]
        graph_latencies = [r[3] for r in rows]
        embedding_latencies = [r[4] for r in rows]
        reranker_latencies = [r[5] for r in rows]
        candidate_counts = [r[6] for r in rows]
        result_counts = [r[7] for r in rows]
        errors = [r[8] for r in rows]
        cache_hits = [r[9] for r in rows]

        n = len(rows)

        return {
            "total_searches": n,
            "latency_p50": float(np.percentile(total_latencies, 50)),
            "latency_p95": float(np.percentile(total_latencies, 95)),
            "latency_p99": float(np.percentile(total_latencies, 99)),
            "avg_total_latency": float(np.mean(total_latencies)),
            "avg_fts_latency": float(np.mean(fts_latencies)),
            "avg_vector_latency": float(np.mean(vector_latencies)),
            "avg_graph_latency": float(np.mean(graph_latencies)),
            "avg_embedding_latency": float(np.mean(embedding_latencies)),
            "avg_reranker_latency": float(np.mean(reranker_latencies)),
            "avg_candidates": float(np.mean(candidate_counts)),
            "avg_results": float(np.mean(result_counts)),
            "error_rate": float(sum(errors) / n),
            "cache_hit_rate": float(sum(cache_hits) / n),
        }
