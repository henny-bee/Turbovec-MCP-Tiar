"""``VectorDB`` — the façade that composes every memory subsystem.

The class itself holds only wiring and shared state; all behaviour lives in
focused mixins that can be read, tested and changed in isolation:

============================  ===================================================
:class:`StorageLifecycleMixin`  metadata/index load, save, migrate
:class:`KnowledgeMixin`         chunked document ingestion and vector search
:class:`GraphRepositoryMixin`   node/edge/observation CRUD
:class:`GraphTraversalMixin`    holograms, neighbours, connectivity
:class:`ExtractionMixin`        text → entities/relations/observations
:class:`SessionMixin`           session chaining and breakthroughs
:class:`SnapshotMixin`          point-in-time graph state
:class:`HybridSearchMixin`      vector + lexical retrieval with RRF
:class:`SemanticRadarMixin`     relationship discovery
:class:`BackgroundDiscoveryMixin`  the discovery daemon
============================  ===================================================
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from typing import Optional

from core.config import Settings
from core.ids import generate_vector_id, to_signed_64, to_unsigned_64
from core.schema import apply_schema
from core.sqlite import ConnectionManager
from core.text import build_embedding_text, extract_description, sanitize_fts_query
from embeddings.factory import build_provider
from embeddings.service import EmbeddingService
from extraction.extractor import ExtractionMixin
from graph.analytics import GraphAnalytics
from graph.repository import GraphRepositoryMixin
from graph.traversal import GraphTraversalMixin
from librarian.librarian import LibrarianService
from memory.knowledge import KnowledgeMixin
from memory.lifecycle import archive_node, list_orphans, prune_stale, restore_node
from memory.ontology import OntologyManager
from memory.provenance import BottleManager
from memory.sessions import SessionMixin
from memory.snapshots import SnapshotMixin
from memory.temporal import (
    diff_knowledge_state,
    get_temporal_neighbors,
    query_timeline,
)
from reranking.cross_encoder import CrossEncoderReranker
from search.background import BackgroundDiscoveryMixin
from search.hybrid import HybridSearchMixin
from search.radar import SemanticRadarMixin
from storage.lifecycle import StorageLifecycleMixin
from storage.vector_index import VectorIndex
from telemetry.metrics import SearchTelemetry

logger = logging.getLogger(__name__)

__all__ = ["VectorDB"]


class VectorDB(
    StorageLifecycleMixin,
    KnowledgeMixin,
    GraphRepositoryMixin,
    GraphTraversalMixin,
    ExtractionMixin,
    SessionMixin,
    SnapshotMixin,
    HybridSearchMixin,
    SemanticRadarMixin,
    BackgroundDiscoveryMixin,
):
    """Embedded hybrid graph + vector + lexical memory."""

    def __init__(
        self,
        dimension: int = 384,
        metadata_file: str = "metadata.json",
        index_file: str = "index.tvim",
        sqlite_db_file: str = "memory.db",
        settings: Optional[Settings] = None,
    ) -> None:
        if settings is None:
            settings = Settings.from_env(
                dimension=dimension,
                metadata_file=metadata_file,
                index_file=index_file,
                sqlite_db_file=sqlite_db_file,
            )
        self.settings = settings
        self.dimension = settings.dimension
        self.metadata_file = settings.metadata_file
        self.index_file = settings.index_file
        self.sqlite_db_file = settings.resolved_sqlite_path()

        # -- embeddings ----------------------------------------------------
        self.ontology_manager = OntologyManager(settings.ontology_file)
        self.embedding_provider = build_provider(settings)
        self.embeddings = EmbeddingService(
            self.embedding_provider,
            cache_size=settings.embedding_cache_size,
            batch_size=settings.embedding_batch_size,
        )
        self.model = self.embedding_provider.model

        # -- storage -------------------------------------------------------
        self.index = VectorIndex(
            self.dimension,
            self.index_file,
            flush_ops=settings.index_flush_ops,
            flush_interval=settings.index_flush_interval,
        )
        self.document_store: dict = {}
        self.next_id = 0

        self._connections = ConnectionManager(self.sqlite_db_file, settings)
        self._init_sqlite_db()
        self.load_storage()

        # -- services ------------------------------------------------------
        self.telemetry = SearchTelemetry(
            self.sqlite_db_file,
            enabled=settings.telemetry_enabled,
            window=settings.telemetry_window,
        )
        self.bottle_manager = BottleManager(self)
        self.analytics = GraphAnalytics(self)
        self.librarian = LibrarianService(
            self,
            similarity_threshold=settings.librarian_similarity_threshold,
            eps=settings.librarian_eps,
            min_samples=settings.librarian_min_samples,
        )
        self.reranker = CrossEncoderReranker(
            model_name=settings.reranker_model, enabled=settings.reranker_enabled
        )

        # -- background discovery ------------------------------------------
        self.bg_thread: Optional[threading.Thread] = None
        self.bg_stop_event = threading.Event()
        self.bg_interval = settings.bg_interval
        self.bg_enabled = settings.bg_enabled
        self.bg_similarity_threshold = settings.bg_similarity_threshold
        if self.bg_enabled:
            self.start_background_discovery()

    # -- connections -------------------------------------------------------
    @property
    def conn(self) -> sqlite3.Connection:
        """The calling thread's SQLite connection."""
        return self._connections.connection

    def transaction(self):
        """Context manager committing on success and rolling back on error."""
        return self._connections.transaction()

    def _init_sqlite_db(self) -> None:
        """Creates tables, indexes and migrations for the configured database."""
        apply_schema(self.conn)

    def close(self) -> None:
        """Stops background work and flushes everything to disk."""
        self.stop_background_discovery()
        try:
            self.save_storage()
        finally:
            self._connections.close_all()

    # -- shared helpers ----------------------------------------------------
    def generate_vector_id(self, node_id: str, conn: sqlite3.Connection) -> int:
        """Stable uint64 vector id for a node id (collision-resolving)."""
        return generate_vector_id(node_id, conn)

    def _to_signed_64(self, value: int) -> int:
        return to_signed_64(value)

    def _to_unsigned_64(self, value: int) -> int:
        return to_unsigned_64(value)

    def _get_description(self, properties) -> str:
        return extract_description(properties)

    def _sanitize_query(self, query: str) -> str:
        return sanitize_fts_query(query)

    def build_embedding_text(
        self, name: str, node_type: str, description: str, observations
    ) -> str:
        """Single source of truth for embedding generation text."""
        return build_embedding_text(
            name,
            node_type,
            description,
            observations,
            max_observations=self.settings.embedding_max_observations,
            max_chars_per_observation=self.settings.embedding_max_chars_per_observation,
        )

    # -- delegated subsystems ---------------------------------------------
    def diff_knowledge_state(self, from_timestamp: str, to_timestamp: str) -> dict:
        """Computes a deterministic delta between two points in time."""
        return diff_knowledge_state(self, from_timestamp, to_timestamp)

    def query_timeline(
        self,
        entity: str = None,
        entity_type: str = None,
        start: str = None,
        end: str = None,
        order: str = "ASC",
        limit: int = 50,
        offset: int = 0,
    ) -> list:
        """Chronological event feed with filtering and pagination."""
        return query_timeline(
            self,
            entity=entity,
            entity_type=entity_type,
            start=start,
            end=end,
            order=order,
            limit=limit,
            offset=offset,
        )

    def get_temporal_neighbors(
        self, node_id: str, direction: str = "both", depth: int = 1
    ) -> list:
        """Explores neighbours that are chronologically before/after an anchor."""
        return get_temporal_neighbors(self, node_id, direction=direction, depth=depth)

    def archive_entity(self, node_id: str, reason: str = "Unused/superseded") -> dict:
        """Archives an entity and its historical records."""
        return archive_node(self, node_id, reason=reason)

    def restore_entity(self, node_id: str) -> dict:
        """Restores an archived entity to ACTIVE status."""
        return restore_node(self, node_id)

    def list_orphans(self) -> list:
        """Lists entities with no edges at all."""
        return list_orphans(self)

    def prune_stale(self, max_age_days: int, dry_run: bool = False) -> list:
        """Prunes stale/archived memories older than ``max_age_days``."""
        return prune_stale(self, max_age_days, dry_run=dry_run)

    def create_bottle(
        self,
        message: str,
        priority: str = "medium",
        expires_at: str = None,
        author_session_id: str = None,
    ) -> dict:
        """Leaves a note for a future session."""
        return self.bottle_manager.create_bottle(
            message, priority, expires_at, author_session_id
        )

    def get_bottles(self, include_acknowledged: bool = False) -> list:
        """Lists unexpired bottle notes."""
        return self.bottle_manager.list_bottles(include_acknowledged)

    def acknowledge_bottle(self, bottle_id: str) -> bool:
        """Marks a bottle note as read."""
        return self.bottle_manager.acknowledge_bottle(bottle_id)

    def search_stats(self) -> dict:
        """Rolling retrieval latency percentiles and error rates."""
        stats = self.telemetry.get_stats()
        if isinstance(stats, dict):
            stats.setdefault("embedding_cache", self.embeddings.cache_stats())
        return stats

    def analyze_graph(self) -> dict:
        """Components, communities, degree centrality and PageRank."""
        return self.analytics.analyze_graph()

    def run_librarian_cycle(self) -> dict:
        """One autonomous clustering, dedup and synthesis cycle."""
        return self.librarian.run_cycle()
