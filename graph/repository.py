"""Transactional CRUD for nodes, edges and observations.

Every write keeps the three representations of an entity consistent: the
relational row, its FTS5 document and its vector. Writes run inside a single
SQLite transaction so a failure anywhere rolls the row back, and the vector
index is compensated explicitly because it lives outside SQLite.
"""

from __future__ import annotations

import datetime
import json
import logging
import sqlite3
from typing import Optional

from core.ids import to_signed_64, to_unsigned_64
from core.text import build_embedding_text, extract_description

logger = logging.getLogger(__name__)

__all__ = ["GraphRepositoryMixin", "utc_now"]


def utc_now() -> str:
    """Current UTC time as an ISO-8601 string (the project-wide timestamp)."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


class GraphRepositoryMixin:
    """CRUD half of :class:`~core.database.VectorDB`."""

    # -- nodes -------------------------------------------------------------
    def create_node(
        self,
        node_id: str,
        name: str,
        node_type: str,
        project_id: str = "default",
        properties: dict = None,
        certainty: str = "confirmed",
        salience_score: float = 1.0,
        occurred_at: str = None,
    ) -> dict:
        """Creates a node and registers its FTS and vector representations."""
        created_at = utc_now()
        properties_json = json.dumps(properties or {})
        vector_id: Optional[int] = None

        try:
            with self.transaction() as conn:
                conn.execute(
                    """
                    INSERT INTO nodes (
                        id, name, node_type, project_id, created_at, updated_at,
                        occurred_at, certainty, salience_score, retrieval_count, properties
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        node_id,
                        name,
                        node_type,
                        project_id,
                        created_at,
                        created_at,
                        occurred_at,
                        certainty,
                        salience_score,
                        0,
                        properties_json,
                    ),
                )

                description = extract_description(properties)
                conn.execute(
                    """
                    INSERT INTO entities_fts (
                        entity_id, project_id, name, node_type, description, observations
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (node_id, project_id, name, node_type, description, ""),
                )

                vector_id = self.generate_vector_id(node_id, conn)
                conn.execute(
                    "INSERT INTO vector_id_mapping (vector_id, node_id) VALUES (?, ?)",
                    (to_signed_64(vector_id), node_id),
                )

                vector = self.embeddings.encode(
                    self._embedding_text(name, node_type, description, [])
                )
                self.index.add_one(vector, vector_id)
        except Exception as exc:
            # SQLite rolled itself back; undo the out-of-transaction vector write.
            if vector_id is not None:
                try:
                    self.index.remove(vector_id)
                except Exception:
                    pass
            logger.error(f"Failed to create node {node_id}: {exc}", exc_info=True)
            raise

        return {
            "id": node_id,
            "name": name,
            "node_type": node_type,
            "project_id": project_id,
            "created_at": created_at,
            "updated_at": created_at,
            "occurred_at": occurred_at,
            "certainty": certainty,
            "salience_score": salience_score,
            "properties": properties or {},
        }

    def update_node(
        self,
        node_id: str,
        name: str = None,
        node_type: str = None,
        properties: dict = None,
        certainty: str = None,
        salience_score: float = None,
        occurred_at: str = None,
    ) -> dict:
        """Applies a partial update and re-synchronises FTS and the vector."""
        row = self.conn.execute(
            """
            SELECT name, node_type, project_id, created_at, occurred_at,
                   certainty, salience_score, properties
            FROM nodes WHERE id = ?
            """,
            (node_id,),
        ).fetchone()
        if not row:
            raise ValueError(f"Node {node_id} not found")

        current_properties = json.loads(row[7]) if row[7] else {}
        new_name = name if name is not None else row[0]
        new_node_type = node_type if node_type is not None else row[1]
        project_id, created_at = row[2], row[3]
        new_occurred_at = occurred_at if occurred_at is not None else row[4]
        new_certainty = certainty if certainty is not None else row[5]
        new_salience = salience_score if salience_score is not None else row[6]

        new_properties = dict(current_properties)
        if properties is not None:
            new_properties.update(properties)

        updated_at = utc_now()
        try:
            with self.transaction() as conn:
                conn.execute(
                    """
                    UPDATE nodes SET
                        name = ?, node_type = ?, updated_at = ?, occurred_at = ?,
                        certainty = ?, salience_score = ?, properties = ?
                    WHERE id = ?
                    """,
                    (
                        new_name,
                        new_node_type,
                        updated_at,
                        new_occurred_at,
                        new_certainty,
                        new_salience,
                        json.dumps(new_properties),
                        node_id,
                    ),
                )
                self._sync_node_fts_and_embedding(node_id, conn)
        except Exception as exc:
            logger.error(f"Failed to update node {node_id}: {exc}", exc_info=True)
            raise

        return {
            "id": node_id,
            "name": new_name,
            "node_type": new_node_type,
            "project_id": project_id,
            "created_at": created_at,
            "updated_at": updated_at,
            "occurred_at": new_occurred_at,
            "certainty": new_certainty,
            "salience_score": new_salience,
            "properties": new_properties,
        }

    def delete_node(self, node_id: str) -> bool:
        """Deletes a node; edges, observations and vectors cascade with it."""
        row = self.conn.execute(
            "SELECT vector_id FROM vector_id_mapping WHERE node_id = ?", (node_id,)
        ).fetchone()
        if not row:
            return False
        vector_id = to_unsigned_64(row[0])

        try:
            with self.transaction() as conn:
                conn.execute("DELETE FROM nodes WHERE id = ?", (node_id,))
                conn.execute("DELETE FROM entities_fts WHERE entity_id = ?", (node_id,))
                self.index.remove(vector_id)
        except Exception as exc:
            logger.error(f"Failed to delete node {node_id}: {exc}", exc_info=True)
            raise
        return True

    # -- edges -------------------------------------------------------------
    def create_edge(
        self,
        edge_id: str,
        from_node_id: str,
        to_node_id: str,
        relationship_type: str,
        confidence: float = 1.0,
        weight: float = 1.0,
        properties: dict = None,
    ) -> dict:
        """Creates a directed, typed, weighted relationship."""
        self.ontology_manager.validate_relation_type(relationship_type)
        created_at = utc_now()

        try:
            with self.transaction() as conn:
                conn.execute(
                    """
                    INSERT INTO edges (
                        id, from_node_id, to_node_id, relationship_type,
                        confidence, weight, created_at, properties
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        edge_id,
                        from_node_id,
                        to_node_id,
                        relationship_type,
                        confidence,
                        weight,
                        created_at,
                        json.dumps(properties or {}),
                    ),
                )
        except Exception as exc:
            logger.error(f"Failed to create edge {edge_id}: {exc}", exc_info=True)
            raise

        return {
            "id": edge_id,
            "from_node_id": from_node_id,
            "to_node_id": to_node_id,
            "relationship_type": relationship_type,
            "confidence": confidence,
            "weight": weight,
            "created_at": created_at,
            "properties": properties or {},
        }

    def update_edge(
        self,
        edge_id: str,
        confidence: float = None,
        weight: float = None,
        properties: dict = None,
    ) -> dict:
        """Updates an edge's weights and properties."""
        row = self.conn.execute(
            """
            SELECT from_node_id, to_node_id, relationship_type, confidence,
                   weight, created_at, properties
            FROM edges WHERE id = ?
            """,
            (edge_id,),
        ).fetchone()
        if not row:
            raise ValueError(f"Edge {edge_id} not found")

        new_confidence = confidence if confidence is not None else row[3]
        new_weight = weight if weight is not None else row[4]
        new_properties = dict(json.loads(row[6]) if row[6] else {})
        if properties is not None:
            new_properties.update(properties)

        try:
            with self.transaction() as conn:
                conn.execute(
                    "UPDATE edges SET confidence = ?, weight = ?, properties = ? WHERE id = ?",
                    (new_confidence, new_weight, json.dumps(new_properties), edge_id),
                )
        except Exception as exc:
            logger.error(f"Failed to update edge {edge_id}: {exc}", exc_info=True)
            raise

        return {
            "id": edge_id,
            "from_node_id": row[0],
            "to_node_id": row[1],
            "relationship_type": row[2],
            "confidence": new_confidence,
            "weight": new_weight,
            "created_at": row[5],
            "properties": new_properties,
        }

    def delete_edge(self, edge_id: str) -> bool:
        """Deletes an edge, returning False when it did not exist."""
        try:
            with self.transaction() as conn:
                cursor = conn.execute("DELETE FROM edges WHERE id = ?", (edge_id,))
                if cursor.rowcount == 0:
                    return False
        except Exception as exc:
            logger.error(f"Failed to delete edge {edge_id}: {exc}", exc_info=True)
            raise
        return True

    # -- observations ------------------------------------------------------
    def create_observation(
        self,
        obs_id: str,
        entity_id: str,
        content: str,
        certainty: str = "confirmed",
        occurred_at: str = None,
    ) -> dict:
        """Attaches an observation and refreshes the parent entity's indices."""
        created_at = utc_now()

        try:
            with self.transaction() as conn:
                conn.execute(
                    """
                    INSERT INTO observations (
                        id, entity_id, content, certainty, created_at, occurred_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (obs_id, entity_id, content, certainty, created_at, occurred_at),
                )
                self._sync_node_fts_and_embedding(entity_id, conn)
        except Exception as exc:
            logger.error(f"Failed to create observation {obs_id}: {exc}", exc_info=True)
            raise

        return {
            "id": obs_id,
            "entity_id": entity_id,
            "content": content,
            "certainty": certainty,
            "created_at": created_at,
            "occurred_at": occurred_at,
        }

    def update_observation(
        self,
        obs_id: str,
        content: str = None,
        certainty: str = None,
        occurred_at: str = None,
    ) -> dict:
        """Updates an observation and refreshes the parent entity's indices."""
        row = self.conn.execute(
            """
            SELECT entity_id, content, certainty, created_at, occurred_at
            FROM observations WHERE id = ?
            """,
            (obs_id,),
        ).fetchone()
        if not row:
            raise ValueError(f"Observation {obs_id} not found")

        entity_id = row[0]
        new_content = content if content is not None else row[1]
        new_certainty = certainty if certainty is not None else row[2]
        new_occurred_at = occurred_at if occurred_at is not None else row[4]

        try:
            with self.transaction() as conn:
                conn.execute(
                    "UPDATE observations SET content = ?, certainty = ?, occurred_at = ? WHERE id = ?",
                    (new_content, new_certainty, new_occurred_at, obs_id),
                )
                self._sync_node_fts_and_embedding(entity_id, conn)
        except Exception as exc:
            logger.error(f"Failed to update observation {obs_id}: {exc}", exc_info=True)
            raise

        return {
            "id": obs_id,
            "entity_id": entity_id,
            "content": new_content,
            "certainty": new_certainty,
            "created_at": row[3],
            "occurred_at": new_occurred_at,
        }

    def delete_observation(self, obs_id: str) -> bool:
        """Deletes an observation and refreshes the parent entity's indices."""
        row = self.conn.execute(
            "SELECT entity_id FROM observations WHERE id = ?", (obs_id,)
        ).fetchone()
        if not row:
            return False
        entity_id = row[0]

        try:
            with self.transaction() as conn:
                conn.execute("DELETE FROM observations WHERE id = ?", (obs_id,))
                self._sync_node_fts_and_embedding(entity_id, conn)
        except Exception as exc:
            logger.error(f"Failed to delete observation {obs_id}: {exc}", exc_info=True)
            raise
        return True

    # -- consistency -------------------------------------------------------
    def _embedding_text(self, name, node_type, description, observations) -> str:
        settings = self.settings
        return build_embedding_text(
            name,
            node_type,
            description,
            observations,
            max_observations=settings.embedding_max_observations,
            max_chars_per_observation=settings.embedding_max_chars_per_observation,
        )

    def _sync_node_fts_and_embedding(
        self, node_id: str, conn: sqlite3.Connection
    ) -> None:
        """Rebuilds the FTS row and vector of a node from its current state."""
        node_row = conn.execute(
            "SELECT name, node_type, properties FROM nodes WHERE id = ?", (node_id,)
        ).fetchone()
        if not node_row:
            raise ValueError(f"Node {node_id} not found during sync")
        name, node_type, properties_json = node_row
        properties = json.loads(properties_json) if properties_json else {}

        observations = [
            row[0]
            for row in conn.execute(
                "SELECT content FROM observations WHERE entity_id = ? "
                "ORDER BY created_at DESC, rowid DESC",
                (node_id,),
            ).fetchall()
        ]

        description = extract_description(properties)
        conn.execute(
            """
            UPDATE entities_fts SET
                name = ?, node_type = ?, description = ?, observations = ?
            WHERE entity_id = ?
            """,
            (name, node_type, description, " ".join(observations), node_id),
        )

        mapping_row = conn.execute(
            "SELECT vector_id FROM vector_id_mapping WHERE node_id = ?", (node_id,)
        ).fetchone()
        if not mapping_row:
            raise ValueError(f"Vector mapping not found for node {node_id}")

        vector = self.embeddings.encode(
            self._embedding_text(name, node_type, description, observations)
        )
        self.index.replace(vector, to_unsigned_64(mapping_row[0]))
