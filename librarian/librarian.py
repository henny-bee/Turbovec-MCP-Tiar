"""The Librarian: autonomous clustering, deduplication and concept synthesis.

One cycle scans active entities, embeds them in a single batch, clusters them
with DBSCAN on cosine distance, reports near-duplicates, runs relationship
discovery, and promotes dense clusters into higher-order concept nodes that
carry provenance back to their sources.
"""

from __future__ import annotations

import datetime
import json
import logging
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Set

import numpy as np

from core.text import extract_description
from graph.rows import fetch_observations
from librarian.clustering import DBSCANClustering
from memory.errors import MEMORY_LAYER_DEGRADED, SearchError

logger = logging.getLogger(__name__)

__all__ = ["LibrarianService"]

# Node types that are journal entries rather than knowledge worth clustering.
_EXCLUDED_TYPES = ("session", "breakthrough")

_SKIPPED_RESULT = {
    "status": "SKIPPED",
    "clusters_detected": 0,
    "synthesized_concepts": 0,
    "duplicates_detected": 0,
    "edges_created": 0,
}


@dataclass
class _Candidate:
    """One clusterable entity with the text that represents it."""

    node_id: str
    name: str
    node_type: str
    text: str


class LibrarianService:
    def __init__(
        self,
        db,
        similarity_threshold: float = 0.75,
        eps: float = 0.4,
        min_samples: int = 2,
    ) -> None:
        self.db = db
        self.similarity_threshold = similarity_threshold
        self.eps = eps
        self.min_samples = min_samples

    # -- public API --------------------------------------------------------
    def run_cycle(self) -> Dict[str, Any]:
        """Executes one full reorganisation cycle and logs its statistics."""
        logger.info("Librarian service cycle started.")
        started = datetime.datetime.now()

        try:
            candidates = self._load_candidates()
            if len(candidates) < self.min_samples:
                logger.info(
                    "Not enough active nodes for a full Librarian clustering cycle."
                )
                return dict(
                    _SKIPPED_RESULT,
                    reason=f"Fewer than {self.min_samples} active nodes.",
                )

            unit_vectors = self._unit_vectors(candidates)
            clusters = self._cluster(unit_vectors)
            duplicates = self._find_duplicates(candidates, unit_vectors, clusters)
            edges_created = self._discover_relationships()
            synthesized = self._synthesize_concepts(candidates, clusters)

            duration = (datetime.datetime.now() - started).total_seconds()
            logger.info(
                f"Librarian cycle completed in {duration:.4f}s. "
                f"Synthesized {len(synthesized)} concepts."
            )
            self._log_run(
                duration,
                len(clusters),
                len(synthesized),
                len(duplicates),
                edges_created,
            )

            return {
                "status": "COMPLETED",
                "duration_seconds": duration,
                "clusters_detected": len(clusters),
                "synthesized_concepts": len(synthesized),
                "duplicates_detected": len(duplicates),
                "edges_created": edges_created,
                "synthesized_nodes": synthesized,
                "duplicates": duplicates,
            }
        except Exception as exc:
            logger.error(f"Critical error during Librarian cycle: {exc}", exc_info=True)
            raise SearchError(
                code=MEMORY_LAYER_DEGRADED,
                message=f"Librarian cycle failed: {exc}",
                subsystem="LIBRARIAN",
                retry_safe=True,
            )

    # -- stages ------------------------------------------------------------
    def _load_candidates(self) -> List[_Candidate]:
        """Loads active entities and their observations in two queries."""
        placeholders = ",".join("?" for _ in _EXCLUDED_TYPES)
        rows = self.db.conn.execute(
            f"""
            SELECT id, name, node_type, properties FROM nodes
            WHERE status = 'ACTIVE' AND node_type NOT IN ({placeholders})
            """,
            _EXCLUDED_TYPES,
        ).fetchall()
        if not rows:
            return []

        observations = fetch_observations(
            self.db.conn,
            [row[0] for row in rows],
            self.db.settings.sqlite_max_variables,
            order="DESC",
        )

        candidates = []
        for node_id, name, node_type, properties_json in rows:
            properties = json.loads(properties_json) if properties_json else {}
            joined = " ".join(item["content"] for item in observations.get(node_id, []))
            description = extract_description(properties)
            candidates.append(
                _Candidate(
                    node_id=node_id,
                    name=name,
                    node_type=node_type,
                    text=f"{name} {node_type} {description} {joined}".strip(),
                )
            )
        return candidates

    def _unit_vectors(self, candidates: Sequence[_Candidate]) -> np.ndarray:
        """Embeds every candidate in one batch and L2-normalises the rows.

        Normalising first makes Euclidean distance monotone in cosine distance,
        which is what DBSCAN's ``eps`` is calibrated against here.
        """
        matrix = self.db.embeddings.encode_matrix([c.text for c in candidates])
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        return matrix / norms

    def _cluster(self, unit_vectors: np.ndarray) -> Dict[int, List[int]]:
        labels = DBSCANClustering(eps=self.eps, min_samples=self.min_samples).fit(
            unit_vectors
        )
        clusters: Dict[int, List[int]] = {}
        for index, label in enumerate(labels):
            if label != -1:
                clusters.setdefault(label, []).append(index)
        return clusters

    def _find_duplicates(
        self,
        candidates: Sequence[_Candidate],
        unit_vectors: np.ndarray,
        clusters: Dict[int, List[int]],
    ) -> List[dict]:
        """Reports intra-cluster pairs that are almost certainly the same thing."""
        threshold = self.db.settings.librarian_duplicate_threshold
        duplicates = []
        for indices in clusters.values():
            block = unit_vectors[indices]
            similarities = block @ block.T
            for left in range(len(indices)):
                for right in range(left + 1, len(indices)):
                    similarity = float(similarities[left, right])
                    if similarity <= threshold:
                        continue
                    duplicates.append(
                        {
                            "node_1": {
                                "id": candidates[indices[left]].node_id,
                                "name": candidates[indices[left]].name,
                            },
                            "node_2": {
                                "id": candidates[indices[right]].node_id,
                                "name": candidates[indices[right]].name,
                            },
                            "similarity": similarity,
                        }
                    )
        return duplicates

    def _discover_relationships(self) -> int:
        try:
            results = self.db.semantic_radar(
                similarity_threshold=self.similarity_threshold, auto_create=True
            )
            return sum(1 for item in results if item.get("created_edge"))
        except Exception as exc:
            logger.error(
                f"Librarian relationship discovery via semantic_radar failed: {exc}"
            )
            return 0

    def _synthesize_concepts(
        self, candidates: Sequence[_Candidate], clusters: Dict[int, List[int]]
    ) -> List[dict]:
        """Promotes each new dense cluster into a provenance-carrying concept."""
        existing_sources = self._synthesized_source_sets()
        synthesized = []

        for label, indices in clusters.items():
            member_types = {candidates[index].node_type for index in indices}
            if member_types == {"concept"}:
                # Guards against the librarian endlessly synthesizing concepts
                # about its own concepts.
                logger.debug(
                    f"Skipping cluster {label} because all members are concepts."
                )
                continue

            source_ids = sorted(candidates[index].node_id for index in indices)
            if frozenset(source_ids) in existing_sources:
                logger.debug(f"Cluster {label} already has synthesized concept.")
                continue

            member_names = [candidates[index].name for index in indices]
            concept_id = f"concept-{uuid.uuid4()}"
            concept_name = f"Synthesized Concept Group {label + 1}"
            properties = {
                "description": (
                    "A synthesized higher-order concept grouping similar facts: "
                    f"{', '.join(member_names)}."
                ),
                "provenance": {
                    "source_type": "librarian",
                    "source_ids": source_ids,
                    "algorithm": "dbscan",
                    "created_at": datetime.datetime.now(
                        datetime.timezone.utc
                    ).isoformat(),
                    "confidence": 0.85,
                },
            }

            try:
                self.db.create_node(
                    node_id=concept_id,
                    name=concept_name,
                    node_type="concept",
                    properties=properties,
                )
                for source_id in source_ids:
                    self.db.create_edge(
                        edge_id=f"edge-{concept_id}-{source_id}-implements",
                        from_node_id=concept_id,
                        to_node_id=source_id,
                        relationship_type="implements",
                        confidence=0.85,
                        weight=1.0,
                        properties={
                            "reason": "Automatic conceptual grouping from librarian DBSCAN cycle"
                        },
                    )
            except Exception as exc:
                logger.error(
                    f"Failed to create synthesized concept {concept_id}: {exc}"
                )
                continue

            existing_sources.add(frozenset(source_ids))
            synthesized.append(
                {"id": concept_id, "name": concept_name, "members": member_names}
            )
        return synthesized

    def _synthesized_source_sets(self) -> Set[frozenset]:
        """Loads every existing librarian provenance once, not once per cluster."""
        sources: Set[frozenset] = set()
        rows = self.db.conn.execute(
            "SELECT properties FROM nodes WHERE node_type = 'concept'"
        ).fetchall()
        for (properties_json,) in rows:
            properties = json.loads(properties_json) if properties_json else {}
            provenance = properties.get("provenance") or {}
            if provenance.get("source_type") == "librarian":
                sources.add(frozenset(provenance.get("source_ids", [])))
        return sources

    def _log_run(
        self,
        duration: float,
        clusters: int,
        synthesized: int,
        duplicates: int,
        edges_created: int,
    ) -> None:
        try:
            with self.db.transaction() as conn:
                conn.execute(
                    """
                    INSERT INTO librarian_runs (
                        timestamp, duration, clusters_detected,
                        synthesized_concepts, duplicates_detected, edges_created
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        duration,
                        clusters,
                        synthesized,
                        duplicates,
                        edges_created,
                    ),
                )
        except Exception as exc:
            logger.error(f"Failed to log Librarian run stats: {exc}")
