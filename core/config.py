"""Centralised, environment-driven configuration.

Every tunable of the memory engine lives here so that behaviour can be changed
without touching call sites. ``Settings.from_env`` is the single entry point.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Any

__all__ = ["Settings", "DEFAULT_PROJECT_ID"]

DEFAULT_PROJECT_ID = "default"


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env_str(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env_str(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    return _env_str(name, "true" if default else "false").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


@dataclass(frozen=True)
class Settings:
    """Immutable runtime configuration for :class:`~core.database.VectorDB`."""

    # --- storage -----------------------------------------------------------
    dimension: int = 384
    metadata_file: str = "metadata.json"
    index_file: str = "index.tvim"
    sqlite_db_file: str = "memory.db"

    # --- sqlite tuning -----------------------------------------------------
    sqlite_timeout: float = 30.0
    sqlite_busy_timeout_ms: int = 10_000
    sqlite_cache_kb: int = 64_000  # negative pragma value => KiB of page cache
    sqlite_mmap_bytes: int = 268_435_456  # 256 MiB
    sqlite_max_variables: int = 900  # stay below SQLITE_MAX_VARIABLE_NUMBER (999)

    # --- chunking ----------------------------------------------------------
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # --- embeddings --------------------------------------------------------
    embedding_provider: str = "minilm"
    embedding_model: str = ""
    embedding_cache_size: int = 512
    embedding_batch_size: int = 64
    embedding_max_observations: int = 20
    embedding_max_chars_per_observation: int = 500

    # --- vector index persistence -----------------------------------------
    index_flush_ops: int = 64  # write to disk after N mutations ...
    index_flush_interval: float = 2.0  # ... or after N seconds, whichever first

    # --- retrieval ---------------------------------------------------------
    rrf_k: float = 60.0
    vector_oversample: int = 5
    vector_min_candidates: int = 100
    rerank_candidate_multiplier: int = 3
    reranker_enabled: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- semantic radar / background discovery -----------------------------
    bg_enabled: bool = False
    bg_interval: int = 300
    bg_similarity_threshold: float = 0.7
    radar_max_nodes: int = 5_000
    radar_block_size: int = 512

    # --- librarian ---------------------------------------------------------
    librarian_similarity_threshold: float = 0.75
    librarian_eps: float = 0.4
    librarian_min_samples: int = 2
    librarian_duplicate_threshold: float = 0.95

    # --- telemetry ---------------------------------------------------------
    telemetry_enabled: bool = True
    telemetry_window: int = 10_000  # rows considered by get_stats()

    # --- misc --------------------------------------------------------------
    ontology_file: str = "ontology.json"
    default_project_id: str = DEFAULT_PROJECT_ID

    @classmethod
    def from_env(cls, **overrides: Any) -> "Settings":
        """Builds settings from environment variables, then applies overrides."""
        settings = cls(
            dimension=_env_int("EMBEDDING_DIMENSION", cls.dimension),
            metadata_file=_env_str("METADATA_FILE", cls.metadata_file),
            index_file=_env_str("INDEX_FILE", cls.index_file),
            sqlite_db_file=_env_str("SQLITE_DB_FILE", cls.sqlite_db_file),
            sqlite_timeout=_env_float("SQLITE_TIMEOUT", cls.sqlite_timeout),
            sqlite_busy_timeout_ms=_env_int(
                "SQLITE_BUSY_TIMEOUT_MS", cls.sqlite_busy_timeout_ms
            ),
            sqlite_cache_kb=_env_int("SQLITE_CACHE_KB", cls.sqlite_cache_kb),
            sqlite_mmap_bytes=_env_int("SQLITE_MMAP_BYTES", cls.sqlite_mmap_bytes),
            chunk_size=_env_int("CHUNK_SIZE", cls.chunk_size),
            chunk_overlap=_env_int("CHUNK_OVERLAP", cls.chunk_overlap),
            embedding_provider=_env_str("EMBEDDING_PROVIDER", cls.embedding_provider),
            embedding_model=_env_str("EMBEDDING_MODEL", cls.embedding_model),
            embedding_cache_size=_env_int(
                "EMBEDDING_CACHE_SIZE", cls.embedding_cache_size
            ),
            embedding_batch_size=_env_int(
                "EMBEDDING_BATCH_SIZE", cls.embedding_batch_size
            ),
            index_flush_ops=_env_int("INDEX_FLUSH_OPS", cls.index_flush_ops),
            index_flush_interval=_env_float(
                "INDEX_FLUSH_INTERVAL", cls.index_flush_interval
            ),
            rrf_k=_env_float("RRF_K", cls.rrf_k),
            vector_oversample=_env_int("VECTOR_OVERSAMPLE", cls.vector_oversample),
            reranker_enabled=_env_bool("RERANKER_ENABLED", cls.reranker_enabled),
            reranker_model=_env_str("RERANKER_MODEL", cls.reranker_model),
            bg_enabled=_env_bool("BACKGROUND_DISCOVERY", cls.bg_enabled),
            bg_interval=_env_int("BG_DISCOVERY_INTERVAL", cls.bg_interval),
            bg_similarity_threshold=_env_float(
                "BG_DISCOVERY_THRESHOLD", cls.bg_similarity_threshold
            ),
            radar_max_nodes=_env_int("RADAR_MAX_NODES", cls.radar_max_nodes),
            telemetry_enabled=_env_bool("TELEMETRY_ENABLED", cls.telemetry_enabled),
            telemetry_window=_env_int("TELEMETRY_WINDOW", cls.telemetry_window),
            ontology_file=_env_str("ONTOLOGY_FILE", cls.ontology_file),
            default_project_id=_env_str("DEFAULT_PROJECT_ID", cls.default_project_id),
        )
        return replace(settings, **overrides) if overrides else settings

    def resolved_sqlite_path(self) -> str:
        """Places the default database next to the metadata file when co-located."""
        if self.sqlite_db_file == "memory.db":
            directory = os.path.dirname(self.metadata_file)
            if directory:
                return os.path.join(directory, "memory.db")
        return self.sqlite_db_file
