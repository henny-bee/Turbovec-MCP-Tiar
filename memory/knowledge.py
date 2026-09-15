"""Document-level ingestion and retrieval (the chunked "knowledge" layer)."""

from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Dict, Iterable, List, Sequence

import numpy as np

from core.ids import to_signed_64
from core.sqlite import chunked
from core.text import chunk_text
from graph.repository import utc_now

logger = logging.getLogger(__name__)

__all__ = ["KnowledgeMixin"]


class KnowledgeMixin:
    """Ingestion half of :class:`~core.database.VectorDB`."""

    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200):
        """Splits text into overlapping windows (see :func:`core.text.chunk_text`)."""
        return chunk_text(text, chunk_size=chunk_size, overlap=overlap)

    def add_knowledge(self, title: str, content: str) -> str:
        """Chunks, embeds and indexes a document, then extracts its graph."""
        if not title.strip():
            return "Error: Title must not be empty."
        if not content.strip():
            return "Error: Content must not be empty."

        chunks = self.chunk_text(
            content,
            chunk_size=self.settings.chunk_size,
            overlap=self.settings.chunk_overlap,
        )
        if not chunks:
            return "Error: No chunks generated from content."

        try:
            vectors = self.embeddings.encode_matrix(chunks)
        except Exception as exc:
            logger.error(f"Failed to encode '{title}': {exc}", exc_info=True)
            return f"Error: Failed to process '{title}'."

        try:
            documents = self._store_chunks(title, chunks, vectors)
        except Exception as exc:
            logger.error(f"Insertion failed during add_knowledge: {exc}", exc_info=True)
            return f"Error: Failed to save knowledge: {exc}"

        for document in documents:
            self.document_store[document["id"]] = document
        self.save_storage()

        self._ingest_extracted_graph(content)
        return (
            f"Successfully added '{title}' ({len(chunks)} chunks) into Turbovec memory."
        )

    def delete_knowledge(self, title_or_id: str) -> str:
        """Removes every chunk matching a document title or chunk id."""
        to_delete = [
            doc_id
            for doc_id, document in self.document_store.items()
            if str(document["id"]) == title_or_id or document["title"] == title_or_id
        ]

        deleted = []
        for doc_id in to_delete:
            if self.index.remove(doc_id):
                del self.document_store[doc_id]
                deleted.append(str(doc_id))

        if not deleted:
            logger.info(f"No document found matching '{title_or_id}' for deletion")
            return f"No document found matching '{title_or_id}'."

        try:
            with self.transaction() as conn:
                for batch in chunked(deleted, self.settings.sqlite_max_variables):
                    placeholders = ",".join("?" for _ in batch)
                    conn.execute(
                        f"DELETE FROM nodes WHERE id IN ({placeholders})", batch
                    )
                    conn.execute(
                        f"DELETE FROM entities_fts WHERE entity_id IN ({placeholders})",
                        batch,
                    )
        except Exception as exc:
            logger.error(f"Failed to delete chunks from SQLite: {exc}", exc_info=True)

        self.save_storage()
        logger.info(f"Deleted {len(deleted)} chunks matching '{title_or_id}'")
        return f"Successfully deleted {len(deleted)} chunks matching '{title_or_id}'."

    def clear_memory(self) -> str:
        """Wipes every artefact and recreates an empty database."""
        self.document_store = {}
        self.next_id = 0
        self.index.reset()
        self.embeddings.clear_cache()
        self._connections.close_all()

        # The -wal/-shm sidecars must go too, or WAL journalling resurrects
        # committed rows the next time the database is opened.
        paths = (
            self.metadata_file,
            self.index_file,
            self.sqlite_db_file,
            f"{self.sqlite_db_file}-wal",
            f"{self.sqlite_db_file}-shm",
        )
        for path in paths:
            if not os.path.exists(path):
                continue
            try:
                os.remove(path)
                logger.info(f"Deleted file: {path}")
            except Exception as exc:
                logger.error(f"Failed to delete file {path}: {exc}", exc_info=True)

        self._init_sqlite_db()
        return "Memory cleared successfully."

    def optimize_index(self) -> str:
        """Compacts on-disk state by rewriting metadata and index."""
        logger.info("Starting index optimization (garbage collection)...")
        self.save_storage()
        return (
            f"Optimization complete. Current size: {len(self.document_store)} chunks."
        )

    def search_knowledge(self, query: str, top_k: int = 3) -> str:
        """Pure vector search over the document chunks, rendered for chat."""
        if not query.strip():
            return "Error: Search query must not be empty."
        if top_k <= 0:
            return "Error: top_k must be greater than zero."
        if not self.document_store:
            return (
                "Database is still empty. Please add knowledge first using the "
                "add_knowledge tool."
            )

        try:
            query_vector = self.embeddings.encode(query)
            search_k = min(len(self.index), top_k + 100)
            distances, ids = self.index.search(
                np.asarray([query_vector], dtype=np.float32), k=search_k
            )
        except Exception as exc:
            logger.error(f"Search failed for query '{query}': {exc}", exc_info=True)
            return f"Error during search: {exc}"

        results = []
        for position, doc_id in enumerate(ids[0]):
            document = self.document_store.get(int(doc_id))
            if document is None:
                continue
            results.append(
                f"📄 **Source**: {document['title']} (ID: {document['id']})\n"
                f"🎯 **Score**: {distances[0][position]:.4f}\n"
                f"📝 **Snippet**:\n{document['content']}"
            )
            if len(results) >= top_k:
                break

        if not results:
            return "No relevant information found."
        return "Semantic Search Results:\n\n" + "\n\n---\n\n".join(results)

    # -- internals ---------------------------------------------------------
    def _store_chunks(
        self, title: str, chunks: Sequence[str], vectors: np.ndarray
    ) -> List[dict]:
        """Writes chunks to SQLite and the vector index in one transaction."""
        documents: List[dict] = []
        vector_ids: List[int] = []
        created_at = utc_now()

        with self.transaction() as conn:
            for index, chunk in enumerate(chunks):
                node_id = str(uuid.uuid4())
                doc_id = self.generate_vector_id(node_id, conn)
                vector_ids.append(doc_id)

                conn.execute(
                    """
                    INSERT INTO nodes (
                        id, name, node_type, project_id, created_at, updated_at,
                        certainty, properties
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        node_id,
                        title,
                        "document",
                        "default",
                        created_at,
                        created_at,
                        "confirmed",
                        _chunk_properties(chunk, index),
                    ),
                )
                conn.execute(
                    """
                    INSERT INTO entities_fts (
                        entity_id, project_id, name, node_type, description, observations
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (node_id, "default", title, "document", chunk, ""),
                )
                conn.execute(
                    "INSERT INTO vector_id_mapping (vector_id, node_id) VALUES (?, ?)",
                    (to_signed_64(doc_id), node_id),
                )
                documents.append(
                    {
                        "id": doc_id,
                        "title": title,
                        "chunk_index": index,
                        "content": chunk,
                    }
                )

            # Inside the transaction: a vector-index failure rolls the rows back.
            self.index.add_with_ids(vectors, np.asarray(vector_ids, dtype=np.uint64))

        return documents

    def _ingest_extracted_graph(self, content: str) -> None:
        """Materialises extracted entities, relations and observations."""
        try:
            extracted = self.extract_entities_and_relations(content)
            entities = extracted.get("entities", [])
            relationships = extracted.get("relationships", [])
            observations = extracted.get("observations", [])

            wanted = {entity["name"] for entity in entities}
            wanted.update(rel["from"] for rel in relationships)
            wanted.update(rel["to"] for rel in relationships)
            wanted.update(obs["entity_name"] for obs in observations)
            ids_by_name = self._node_ids_by_name(wanted)

            for entity in entities:
                if entity["name"] in ids_by_name:
                    continue
                node_id = str(uuid.uuid4())
                self.create_node(
                    node_id=node_id,
                    name=entity["name"],
                    node_type=entity["type"],
                    properties=entity["properties"],
                )
                ids_by_name[entity["name"]] = node_id

            self._create_extracted_edges(relationships, ids_by_name)

            for observation in observations:
                entity_id = ids_by_name.get(observation["entity_name"])
                if entity_id:
                    self.create_observation(
                        obs_id=str(uuid.uuid4()),
                        entity_id=entity_id,
                        content=observation["content"],
                    )
        except Exception as exc:
            logger.error(
                f"Extraction pipeline failed during add_knowledge: {exc}", exc_info=True
            )

    def _node_ids_by_name(self, names: Iterable[str]) -> Dict[str, str]:
        """One batched lookup instead of a query per extracted name."""
        unique = [name for name in dict.fromkeys(names) if name]
        found: Dict[str, str] = {}
        if not unique:
            return found

        cursor = self.conn.cursor()
        for batch in chunked(unique, self.settings.sqlite_max_variables):
            placeholders = ",".join("?" for _ in batch)
            cursor.execute(
                f"SELECT name, id FROM nodes WHERE name IN ({placeholders})", batch
            )
            for name, node_id in cursor.fetchall():
                found.setdefault(name, node_id)
        return found

    def _create_extracted_edges(
        self, relationships: Sequence[dict], ids_by_name: Dict[str, str]
    ) -> None:
        pending = []
        for relationship in relationships:
            from_id = ids_by_name.get(relationship["from"])
            to_id = ids_by_name.get(relationship["to"])
            if not from_id or not to_id:
                continue
            edge_id = f"edge-{from_id}-{to_id}-{relationship['type']}"
            pending.append((edge_id, from_id, to_id, relationship))

        if not pending:
            return

        existing: set = set()
        cursor = self.conn.cursor()
        edge_ids = [edge_id for edge_id, _, _, _ in pending]
        for batch in chunked(edge_ids, self.settings.sqlite_max_variables):
            placeholders = ",".join("?" for _ in batch)
            cursor.execute(f"SELECT id FROM edges WHERE id IN ({placeholders})", batch)
            existing.update(row[0] for row in cursor.fetchall())

        for edge_id, from_id, to_id, relationship in pending:
            if edge_id in existing:
                continue
            self.create_edge(
                edge_id=edge_id,
                from_node_id=from_id,
                to_node_id=to_id,
                relationship_type=relationship["type"],
                properties=relationship.get("properties"),
            )
            existing.add(edge_id)


def _chunk_properties(chunk: str, index: int) -> str:
    return json.dumps({"content": chunk, "chunk_index": index})
