"""Semantic radar: finds semantically close entities that the graph never linked.

The naive form of this scan is doubly quadratic — an embedding per node per
scan, and a breadth-first path search per candidate pair. Here the embeddings
are produced in one batch, similarity is a blocked matrix product, and
reachability comes from a single union-find that is kept up to date as edges
are created.
"""

from __future__ import annotations

import json
import logging
from typing import List, Optional, Sequence, Tuple

import numpy as np

from core.text import extract_description
from graph.rows import fetch_observations

logger = logging.getLogger(__name__)

__all__ = ["SemanticRadarMixin"]

# Node-type pairs that imply a specific relationship, checked in order.
_TYPE_RELATIONS = (
    ("Decision", "DECIDED_IN"),
    ("Session", "MENTIONED_IN"),
    ("Person", "CREATED_BY"),
)


class SemanticRadarMixin:
    """Relationship-discovery half of :class:`~core.database.VectorDB`."""

    def semantic_radar(
        self,
        similarity_threshold: float = 0.7,
        project_id: str = "default",
        auto_create: bool = False,
    ) -> List[dict]:
        """Reports (and optionally creates) missing links between similar nodes."""
        nodes = self._radar_nodes(project_id)
        if len(nodes) < 2:
            return []

        unit_vectors = self._radar_vectors(nodes)
        components = self.connectivity(node["id"] for node in nodes)

        suggestions: List[dict] = []
        for left_index, right_index, similarity in self._similar_pairs(
            unit_vectors, similarity_threshold
        ):
            node_a, node_b = nodes[left_index], nodes[right_index]
            if components.connected(node_a["id"], node_b["id"]):
                continue

            relationship = self._infer_relationship_type(
                node_a["node_type"],
                node_b["node_type"],
                node_a["project_id"],
                node_b["project_id"],
            )
            edge_details = None
            created_edge = False
            if auto_create:
                edge_details = self._create_discovered_edge(
                    node_a["id"], node_b["id"], relationship
                )
                created_edge = edge_details is not None
                if created_edge:
                    components.union(node_a["id"], node_b["id"])

            suggestions.append(
                {
                    "node_1": node_a,
                    "node_2": node_b,
                    "similarity": similarity,
                    "suggested_relationship": relationship,
                    "reasoning": (
                        f"High semantic similarity ({similarity:.2f}) but no graph path "
                        f"— potential bridge"
                    ),
                    "created_edge": created_edge,
                    "edge_details": edge_details,
                }
            )

        suggestions.sort(key=lambda item: item["similarity"], reverse=True)
        return suggestions

    # -- inputs ------------------------------------------------------------
    def _radar_nodes(self, project_id: str) -> List[dict]:
        """Loads the scannable nodes of a project, bounded by ``radar_max_nodes``."""
        limit = self.settings.radar_max_nodes
        rows = self.conn.execute(
            """
            SELECT id, name, node_type, properties FROM nodes
            WHERE project_id = ?
            ORDER BY created_at ASC, id ASC
            LIMIT ?
            """,
            (project_id, limit),
        ).fetchall()
        if len(rows) == limit:
            logger.warning(
                f"Semantic radar truncated to the {limit} oldest nodes of "
                f"project '{project_id}'; raise RADAR_MAX_NODES to widen the scan."
            )
        return [
            {
                "id": row[0],
                "name": row[1],
                "node_type": row[2],
                "project_id": project_id,
                "_properties": row[3],
            }
            for row in rows
        ]

    def _radar_vectors(self, nodes: Sequence[dict]) -> np.ndarray:
        """Embeds every node in one batch and returns L2-normalised rows."""
        node_ids = [node["id"] for node in nodes]
        observations = fetch_observations(
            self.conn, node_ids, self.settings.sqlite_max_variables, order="DESC"
        )

        texts = []
        for node in nodes:
            properties = json.loads(node["_properties"]) if node["_properties"] else {}
            texts.append(
                self._embedding_text(
                    node["name"],
                    node["node_type"],
                    extract_description(properties),
                    [obs["content"] for obs in observations.get(node["id"], [])],
                )
            )
            node.pop("_properties", None)

        matrix = self.embeddings.encode_matrix(texts)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # zero vectors stay zero, similarity stays 0
        return matrix / norms

    def _similar_pairs(
        self, unit_vectors: np.ndarray, threshold: float
    ) -> List[Tuple[int, int, float]]:
        """Yields ``(i, j, similarity)`` for i < j above ``threshold``.

        Similarity is computed one row-block at a time so peak memory stays at
        ``block_size * n`` floats rather than ``n * n``.
        """
        count = unit_vectors.shape[0]
        block = max(1, self.settings.radar_block_size)
        pairs: List[Tuple[int, int, float]] = []

        for start in range(0, count, block):
            stop = min(start + block, count)
            similarities = unit_vectors[start:stop] @ unit_vectors.T
            for offset in range(stop - start):
                left = start + offset
                row = similarities[offset]
                for right in np.nonzero(row[left + 1 :] >= threshold)[0]:
                    right_index = left + 1 + int(right)
                    pairs.append((left, right_index, float(row[right_index])))
        return pairs

    # -- outputs -----------------------------------------------------------
    def _create_discovered_edge(
        self, from_id: str, to_id: str, relationship: str
    ) -> Optional[dict]:
        edge_id = f"edge-{from_id}-{to_id}-{relationship}"
        try:
            exists = self.conn.execute(
                "SELECT 1 FROM edges WHERE id = ? LIMIT 1", (edge_id,)
            ).fetchone()
            if exists:
                return None
            return self.create_edge(
                edge_id=edge_id,
                from_node_id=from_id,
                to_node_id=to_id,
                relationship_type=relationship,
                properties={
                    "reason": "Automatic relationship inference from semantic_radar"
                },
            )
        except Exception as exc:
            logger.error(f"Failed to auto-create edge in semantic_radar: {exc}")
            return None

    def _infer_relationship_type(
        self,
        source_type: str,
        candidate_type: str,
        source_project: str,
        candidate_project: str,
    ) -> str:
        """Heuristic relationship type inference from node types."""
        if source_project and candidate_project and source_project != candidate_project:
            return "BRIDGES_TO"

        types = frozenset({source_type, candidate_type})

        if types in ({"Concept"}, {"Breakthrough"}) or (
            "Concept" in types and "Analogy" in types
        ):
            return "ANALOGOUS_TO"

        if "Tool" in types and "Procedure" in types:
            return "ENABLES"

        for node_type, relationship in _TYPE_RELATIONS:
            if node_type in types:
                return relationship

        return "RELATED_TO"
