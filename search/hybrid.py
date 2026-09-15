"""The hybrid vector + lexical retrieval pipeline.

Stages, in order: vector channel, lexical (FTS5) channel, reciprocal rank
fusion, node hydration, optional cross-encoder reranking. Every stage is timed
and every failure is reported with a typed :class:`SearchError` rather than
silently degrading, so callers can tell "no results" from "retrieval broke".
"""

from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from core.ids import to_signed_64, to_unsigned_64
from core.sqlite import chunked
from core.text import sanitize_fts_query
from graph.rows import fetch_nodes, fetch_observations
from memory.errors import (
    MEMORY_LAYER_DEGRADED,
    RERANKER_FAILED,
    SQLITE_TRANSACTION_FAILED,
    SQLITE_UNAVAILABLE,
    VECTOR_INDEX_UNAVAILABLE,
    SearchError,
)
from search.fusion import reciprocal_rank_fusion, top_n

logger = logging.getLogger(__name__)

__all__ = ["HybridSearchMixin"]


@dataclass
class _Timings:
    """Per-stage latencies plus counters, mirrored straight into telemetry."""

    started: float = field(default_factory=time.perf_counter)
    fts: float = 0.0
    vector: float = 0.0
    embedding: float = 0.0
    reranker: float = 0.0
    graph: float = 0.0
    candidates: int = 0
    results: int = 0

    @property
    def total(self) -> float:
        return time.perf_counter() - self.started


class HybridSearchMixin:
    """Retrieval half of :class:`~core.database.VectorDB`."""

    def search_hybrid(
        self,
        query: str,
        project_id: str = "default",
        limit: int = 10,
        channel_weights: dict = None,
    ) -> List[dict]:
        """Runs both channels, fuses them with RRF and optionally reranks."""
        timings = _Timings()
        weights = channel_weights or {}
        vector_weight = weights.get("vector", 1.0)
        lexical_weight = weights.get("lexical", 1.0)

        if not self._has_any_node(timings):
            return []

        vector_ids = (
            self._vector_channel(query, project_id, limit, timings)
            if vector_weight > 0.0 and query.strip()
            else []
        )
        lexical_ids = (
            self._lexical_channel(query, project_id, timings)
            if lexical_weight > 0.0 and query.strip()
            else []
        )

        scores = reciprocal_rank_fusion(
            ((vector_ids, vector_weight), (lexical_ids, lexical_weight)),
            k=self.settings.rrf_k,
        )
        if not scores:
            self._record_search(timings)
            return []

        candidate_limit = (
            limit * self.settings.rerank_candidate_multiplier
            if self.reranker.enabled
            else limit
        )
        ranked = top_n(scores, candidate_limit)
        timings.candidates = len(ranked)

        results = self._hydrate(ranked, timings)
        results = self._rerank(query, results, limit, timings)

        timings.results = len(results)
        self._record_search(timings)
        return results

    # -- stages ------------------------------------------------------------
    def _has_any_node(self, timings: _Timings) -> bool:
        """Fail-loud guard: an unreachable database must not look like no hits."""
        try:
            row = self.conn.execute("SELECT 1 FROM nodes LIMIT 1").fetchone()
        except sqlite3.Error as exc:
            self._record_search(timings, error_code=SQLITE_UNAVAILABLE)
            raise SearchError(
                code=SQLITE_UNAVAILABLE,
                message=f"SQLite Database Unavailable: {exc}",
                subsystem="SQLITE",
                retry_safe=True,
            )
        return row is not None

    def _vector_channel(
        self, query: str, project_id: str, limit: int, timings: _Timings
    ) -> List[str]:
        """Embeds the query and resolves ANN hits back to active node ids."""
        try:
            started = time.perf_counter()
            query_vector = self.embeddings.encode(query)
            timings.embedding = time.perf_counter() - started

            started = time.perf_counter()
            node_ids = self._search_index(query_vector, project_id, limit)
            timings.vector = time.perf_counter() - started
            return node_ids
        except SearchError as error:
            self._record_search(timings, error_code=error.code)
            raise
        except Exception as exc:
            logger.error(f"Vector search failed unexpectedly: {exc}")
            self._record_search(timings, error_code=MEMORY_LAYER_DEGRADED)
            raise SearchError(
                code=MEMORY_LAYER_DEGRADED,
                message=f"Vector index search degraded: {exc}",
                subsystem="VECTOR_INDEX",
                retry_safe=True,
            )

    def _search_index(
        self, query_vector: np.ndarray, project_id: str, limit: int
    ) -> List[str]:
        cursor = self.conn.cursor()
        total = cursor.execute("SELECT COUNT(*) FROM vector_id_mapping").fetchone()[0]
        if not total:
            return []

        settings = self.settings
        search_k = min(
            total,
            max(settings.vector_min_candidates, limit * settings.vector_oversample),
        )
        try:
            _distances, ids = self.index.search(
                np.asarray([query_vector], dtype=np.float32), k=search_k
            )
        except Exception as exc:
            raise SearchError(
                code=VECTOR_INDEX_UNAVAILABLE,
                message=f"Turbovec index search failed: {exc}",
                subsystem="VECTOR_INDEX",
                retry_safe=True,
            )

        hit_ids = [int(vector_id) for vector_id in ids[0]]
        if not hit_ids:
            return []

        # Resolve vector ids to node ids, dropping other projects and archives.
        node_by_vector: Dict[int, str] = {}
        max_vars = settings.sqlite_max_variables
        for batch in chunked([to_signed_64(v) for v in hit_ids], max_vars - 1):
            placeholders = ",".join("?" for _ in batch)
            cursor.execute(
                f"""
                SELECT n.id, m.vector_id
                FROM nodes n
                JOIN vector_id_mapping m ON n.id = m.node_id
                WHERE n.project_id = ? AND m.vector_id IN ({placeholders})
                  AND n.status != 'ARCHIVED'
                """,
                (project_id, *batch),
            )
            for node_id, vector_id in cursor.fetchall():
                node_by_vector[to_unsigned_64(vector_id)] = node_id

        # Preserve the ANN ranking.
        return [
            node_by_vector[vector_id]
            for vector_id in hit_ids
            if vector_id in node_by_vector
        ]

    def _lexical_channel(
        self, query: str, project_id: str, timings: _Timings
    ) -> List[str]:
        """Runs FTS5 MATCH and filters archived entities, preserving rank."""
        started = time.perf_counter()
        try:
            safe_query = sanitize_fts_query(query)
            if not safe_query:
                return []

            cursor = self.conn.cursor()
            cursor.execute(
                """
                SELECT entity_id FROM entities_fts
                WHERE project_id = ? AND entities_fts MATCH ?
                ORDER BY rank
                """,
                (project_id, safe_query),
            )
            ranked_ids = [row[0] for row in cursor.fetchall()]
            if not ranked_ids:
                return []

            active: set = set()
            for batch in chunked(ranked_ids, self.settings.sqlite_max_variables):
                placeholders = ",".join("?" for _ in batch)
                cursor.execute(
                    f"SELECT id FROM nodes WHERE status != 'ARCHIVED' AND id IN ({placeholders})",
                    batch,
                )
                active.update(row[0] for row in cursor.fetchall())
            return [node_id for node_id in ranked_ids if node_id in active]
        except sqlite3.OperationalError as exc:
            logger.warning(f"FTS search channel failed for query {query!r}: {exc}")
            return []
        finally:
            timings.fts = time.perf_counter() - started

    def _hydrate(self, ranked: Sequence, timings: _Timings) -> List[dict]:
        """Loads full node records (and their observations) for the surviving candidates."""
        started = time.perf_counter()
        node_ids = [node_id for node_id, _ in ranked]
        max_vars = self.settings.sqlite_max_variables

        self._bump_retrieval_counts(node_ids, max_vars)

        try:
            conn = self.conn
            nodes_by_id = fetch_nodes(conn, node_ids, max_vars, with_status=True)
            observations = fetch_observations(conn, list(nodes_by_id), max_vars)
        except Exception as exc:
            logger.error(f"Failed to fetch node details: {exc}", exc_info=True)
            self._record_search(timings, error_code=SQLITE_TRANSACTION_FAILED)
            raise SearchError(
                code=SQLITE_TRANSACTION_FAILED,
                message=f"Database retrieval failure during hydrate: {exc}",
                subsystem="SQLITE",
                retry_safe=True,
            )
        timings.graph = time.perf_counter() - started

        results = []
        for node_id, score in ranked:
            node = nodes_by_id.get(node_id)
            if node is None:
                continue
            node = dict(node)
            node["observations"] = observations.get(node_id, [])
            node["rrf_score"] = score
            results.append(node)
        return results

    def _bump_retrieval_counts(self, node_ids: Sequence[str], max_vars: int) -> None:
        """Best-effort salience bookkeeping; never fails a search."""
        try:
            with self.transaction() as conn:
                for batch in chunked(list(node_ids), max_vars):
                    placeholders = ",".join("?" for _ in batch)
                    conn.execute(
                        f"UPDATE nodes SET retrieval_count = retrieval_count + 1 "
                        f"WHERE id IN ({placeholders})",
                        batch,
                    )
        except Exception as exc:
            logger.error(f"Failed to increment retrieval count: {exc}", exc_info=True)

    def _rerank(
        self, query: str, results: List[dict], limit: int, timings: _Timings
    ) -> List[dict]:
        if not self.reranker.enabled:
            return results[:limit]

        started = time.perf_counter()
        try:
            reranked = self.reranker.rerank(query, results, limit)
        except Exception as exc:
            self._record_search(timings, error_code=RERANKER_FAILED)
            raise SearchError(
                code=RERANKER_FAILED,
                message=f"Reranking stage failed: {exc}",
                subsystem="RERANKER",
                retry_safe=True,
            )
        timings.reranker = time.perf_counter() - started
        return reranked

    # -- telemetry ---------------------------------------------------------
    def _record_search(
        self, timings: _Timings, error_code: Optional[str] = None
    ) -> None:
        self.telemetry.record_search(
            total_latency=timings.total,
            fts_latency=timings.fts,
            vector_latency=timings.vector,
            graph_latency=timings.graph,
            embedding_latency=timings.embedding,
            reranker_latency=timings.reranker,
            candidate_count=timings.candidates,
            result_count=timings.results,
            is_error=error_code is not None,
            error_code=error_code,
        )
