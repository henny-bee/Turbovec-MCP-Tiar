"""Startup and shutdown of the on-disk state: metadata, index and migrations."""

from __future__ import annotations

import datetime
import json
import logging
from typing import List

import numpy as np

from core.ids import to_signed_64
from storage.document_store import load_documents, save_documents

logger = logging.getLogger(__name__)

__all__ = ["StorageLifecycleMixin"]


class StorageLifecycleMixin:
    """Persistence lifecycle half of :class:`~core.database.VectorDB`."""

    def save_storage(self) -> None:
        """Flushes the document metadata and the vector index to disk."""
        save_documents(self.metadata_file, self.document_store, self.next_id)
        self.index.flush(force=True)

    def load_storage(self) -> None:
        """Restores metadata and index, rebuilding or migrating when needed."""
        self.document_store, self.next_id = load_documents(self.metadata_file)

        loaded_index = self.index.load()
        if loaded_index and len(self.index) != len(self.document_store):
            logger.warning("Index and metadata contain different numbers of entries")

        if not loaded_index and self.document_store:
            self._rebuild_index()

        self._migrate_documents_to_sqlite()

    # -- internals ---------------------------------------------------------
    def _rebuild_index(self) -> None:
        """Re-embeds the document store when the index file is unusable."""
        logger.info("Rebuilding index from document store...")
        try:
            ids: List[int] = []
            texts: List[str] = []
            for doc_id, document in self.document_store.items():
                ids.append(doc_id)
                texts.append(document["content"])

            if ids:
                self.index.add_with_ids(
                    self.embeddings.encode_matrix(texts),
                    np.asarray(ids, dtype=np.uint64),
                )
            self.save_storage()
            logger.info("Finished rebuilding index.")
        except Exception:
            self.index.reset()
            logger.error("Failed to rebuild index from metadata", exc_info=True)
            raise

    def _migrate_documents_to_sqlite(self) -> None:
        """Imports a legacy metadata-only store into the graph tables once."""
        if not self.document_store:
            return
        cursor = self.conn.cursor()
        if cursor.execute("SELECT 1 FROM nodes LIMIT 1").fetchone():
            return

        logger.info("Migrating existing document store to SQLite...")
        created_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        node_rows = []
        fts_rows = []
        mapping_rows = []

        for doc_id, document in self.document_store.items():
            node_id = str(doc_id)
            title = document.get("title", "Untitled")
            content = document.get("content", "")
            node_rows.append(
                (
                    node_id,
                    title,
                    "document",
                    "default",
                    created_at,
                    created_at,
                    "confirmed",
                    json.dumps(
                        {
                            "content": content,
                            "chunk_index": document.get("chunk_index", 0),
                        }
                    ),
                )
            )
            fts_rows.append((node_id, "default", title, "document", content, ""))
            mapping_rows.append((to_signed_64(int(doc_id)), node_id))

        try:
            with self.transaction() as conn:
                conn.executemany(
                    """
                    INSERT INTO nodes (
                        id, name, node_type, project_id, created_at, updated_at,
                        certainty, properties
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    node_rows,
                )
                conn.executemany(
                    """
                    INSERT INTO entities_fts (
                        entity_id, project_id, name, node_type, description, observations
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    fts_rows,
                )
                conn.executemany(
                    "INSERT OR IGNORE INTO vector_id_mapping (vector_id, node_id) VALUES (?, ?)",
                    mapping_rows,
                )
            logger.info("Migration complete.")
        except Exception as exc:
            logger.error(f"Migration failed: {exc}", exc_info=True)
