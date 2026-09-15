"""Single source of truth for the SQLite schema, indexes and migrations."""

from __future__ import annotations

import logging
import sqlite3
from typing import Sequence

logger = logging.getLogger(__name__)

__all__ = ["apply_schema", "TABLES", "INDEXES", "MIGRATIONS"]


TABLES: Sequence[str] = (
    """
    CREATE TABLE IF NOT EXISTS nodes (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        node_type TEXT NOT NULL,
        project_id TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        occurred_at TEXT,
        certainty TEXT NOT NULL DEFAULT 'confirmed',
        salience_score REAL NOT NULL DEFAULT 1.0,
        retrieval_count INTEGER NOT NULL DEFAULT 0,
        status TEXT NOT NULL DEFAULT 'ACTIVE',
        properties TEXT
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS edges (
        id TEXT PRIMARY KEY,
        from_node_id TEXT NOT NULL,
        to_node_id TEXT NOT NULL,
        relationship_type TEXT NOT NULL,
        confidence REAL NOT NULL DEFAULT 1.0,
        weight REAL NOT NULL DEFAULT 1.0,
        created_at TEXT NOT NULL,
        properties TEXT,
        FOREIGN KEY (from_node_id) REFERENCES nodes(id) ON DELETE CASCADE,
        FOREIGN KEY (to_node_id) REFERENCES nodes(id) ON DELETE CASCADE
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS observations (
        id TEXT PRIMARY KEY,
        entity_id TEXT NOT NULL,
        content TEXT NOT NULL,
        certainty TEXT NOT NULL DEFAULT 'confirmed',
        created_at TEXT NOT NULL,
        occurred_at TEXT,
        FOREIGN KEY (entity_id) REFERENCES nodes(id) ON DELETE CASCADE
    );
    """,
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS entities_fts USING fts5(
        entity_id UNINDEXED,
        project_id UNINDEXED,
        name,
        node_type,
        description,
        observations,
        tokenize='porter unicode61'
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS vector_id_mapping (
        vector_id INTEGER PRIMARY KEY,
        node_id TEXT UNIQUE NOT NULL,
        FOREIGN KEY (node_id) REFERENCES nodes(id) ON DELETE CASCADE
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS bottles (
        id TEXT PRIMARY KEY,
        message TEXT NOT NULL,
        created_at TEXT NOT NULL,
        author_session_id TEXT,
        priority TEXT NOT NULL,
        expires_at TEXT,
        acknowledged INTEGER NOT NULL DEFAULT 0
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS librarian_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        duration REAL NOT NULL,
        clusters_detected INTEGER NOT NULL,
        synthesized_concepts INTEGER NOT NULL,
        duplicates_detected INTEGER NOT NULL,
        edges_created INTEGER NOT NULL
    );
    """,
    """
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
    """,
)

# Indexes backing every hot lookup path. Names are stable so re-running is a
# no-op on existing databases.
INDEXES: Sequence[str] = (
    "CREATE INDEX IF NOT EXISTS idx_nodes_project_type ON nodes(project_id, node_type);",
    "CREATE INDEX IF NOT EXISTS idx_nodes_created_at ON nodes(created_at);",
    "CREATE INDEX IF NOT EXISTS idx_nodes_name ON nodes(name);",
    "CREATE INDEX IF NOT EXISTS idx_nodes_status ON nodes(status);",
    "CREATE INDEX IF NOT EXISTS idx_edges_from ON edges(from_node_id);",
    "CREATE INDEX IF NOT EXISTS idx_edges_to ON edges(to_node_id);",
    "CREATE INDEX IF NOT EXISTS idx_edges_type ON edges(relationship_type);",
    "CREATE INDEX IF NOT EXISTS idx_observations_entity ON observations(entity_id);",
    "CREATE INDEX IF NOT EXISTS idx_observations_created_at ON observations(created_at);",
    "CREATE INDEX IF NOT EXISTS idx_observations_entity_created ON observations(entity_id, created_at);",
    "CREATE INDEX IF NOT EXISTS idx_bottles_acknowledged ON bottles(acknowledged, created_at);",
)

# Idempotent statements for databases created by older versions. Each is
# attempted once and its "duplicate column" failure is expected and ignored.
MIGRATIONS: Sequence[str] = (
    "ALTER TABLE nodes ADD COLUMN status TEXT NOT NULL DEFAULT 'ACTIVE';",
)


def apply_schema(conn: sqlite3.Connection) -> None:
    """Creates every table, index and pending migration in one transaction."""
    with conn:
        for statement in TABLES:
            conn.execute(statement)
        for statement in MIGRATIONS:
            try:
                conn.execute(statement)
                logger.info(f"Applied schema migration: {statement.strip()}")
            except sqlite3.OperationalError:
                pass  # already applied
        for statement in INDEXES:
            conn.execute(statement)
