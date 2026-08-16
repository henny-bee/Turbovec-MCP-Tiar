import uuid
import logging
import datetime
import json
import numpy as np
from typing import Dict, List, Any, Set
from librarian.clustering import DBSCANClustering
from memory.errors import SearchError, MEMORY_LAYER_DEGRADED

logger = logging.getLogger(__name__)

class LibrarianService:
    def __init__(self, db, similarity_threshold: float = 0.75, eps: float = 0.4, min_samples: int = 2):
        self.db = db
        self.similarity_threshold = similarity_threshold
        self.eps = eps
        self.min_samples = min_samples

    def run_cycle(self) -> Dict[str, Any]:
        """
        Executes one Librarian cycle:
        1. Scanning and generating embeddings for all active nodes.
        2. Clustering nodes via DBSCAN.
        3. Identifying potential duplicate entities.
        4. Running semantic radar relationship discovery.
        5. Synthesizing higher-order concepts for dense clusters with structured provenance.
        """
        logger.info("Librarian service cycle started.")
        start_time = datetime.datetime.now()

        try:
            # Step 1: Scan nodes
            cursor = self.db.conn.cursor()
            cursor.execute(
                "SELECT id, name, node_type, properties FROM nodes WHERE status = 'ACTIVE' AND node_type != 'session' AND node_type != 'breakthrough'"
            )
            rows = cursor.fetchall()
            
            if len(rows) < self.min_samples:
                logger.info("Not enough active nodes for a full Librarian clustering cycle.")
                return {
                    "status": "SKIPPED",
                    "reason": f"Fewer than {self.min_samples} active nodes.",
                    "clusters_detected": 0,
                    "synthesized_concepts": 0,
                    "duplicates_detected": 0,
                    "edges_created": 0
                }

            node_ids = []
            node_names = []
            node_types = []
            embeddings_list = []

            for row in rows:
                nid, name, ntype, properties_json = row
                props = json.loads(properties_json) if properties_json else {}
                
                # Fetch observations
                cursor.execute(
                    "SELECT content FROM observations WHERE entity_id = ? ORDER BY created_at DESC",
                    (nid,)
                )
                obs_concat = " ".join([r[0] for r in cursor.fetchall()])
                
                desc = props.get("description", "") if isinstance(props, dict) else str(props)
                emb_text = f"{name} {ntype} {desc} {obs_concat}".strip()
                
                # Use current db model to encode
                vector = self.db.model.encode(emb_text)
                
                node_ids.append(nid)
                node_names.append(name)
                node_types.append(ntype)
                embeddings_list.append(vector)

            embeddings = np.asarray(embeddings_list, dtype=np.float32)

            # Step 2: Semantic Clustering using DBSCAN
            # Normalize embeddings to make Euclidean distance equivalent to cosine distance
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1e-10  # prevent division by zero
            normalized_embeddings = embeddings / norms

            dbscan = DBSCANClustering(eps=self.eps, min_samples=self.min_samples)
            labels = dbscan.fit(normalized_embeddings)

            # Group node indices by cluster
            clusters = {}
            for idx, lbl in enumerate(labels):
                if lbl != -1:
                    clusters.setdefault(lbl, []).append(idx)

            # Step 3: Duplicate Detection
            duplicates = []
            for lbl, indices in clusters.items():
                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        idx_a = indices[i]
                        idx_b = indices[j]
                        v_a = normalized_embeddings[idx_a]
                        v_b = normalized_embeddings[idx_b]
                        sim = float(np.dot(v_a, v_b))
                        if sim > 0.95:
                            duplicates.append({
                                "node_1": {"id": node_ids[idx_a], "name": node_names[idx_a]},
                                "node_2": {"id": node_ids[idx_b], "name": node_names[idx_b]},
                                "similarity": sim
                            })

            # Step 4: Relationship Discovery (Semantic Radar)
            # Run semantic radar on active project
            discovered_edges_count = 0
            try:
                radar_results = self.db.semantic_radar(
                    similarity_threshold=self.similarity_threshold,
                    auto_create=True
                )
                discovered_edges_count = sum(1 for r in radar_results if r.get("created_edge"))
            except Exception as e:
                logger.error(f"Librarian relationship discovery via semantic_radar failed: {e}")

            # Step 5: Concept Synthesis
            synthesized_count = 0
            synthesized_nodes = []

            for lbl, indices in clusters.items():
                # Avoid infinite synthesis loops:
                # 1. Do not cluster if ALL nodes in the cluster are already Synthesized Concepts
                # 2. Check if a synthesized concept already represents this specific combination of nodes
                cluster_node_ids = sorted([node_ids[idx] for idx in indices])
                
                # Check if some nodes are already concepts
                types_in_cluster = [node_types[idx] for idx in indices]
                if all(t == "concept" for t in types_in_cluster):
                    logger.debug(f"Skipping cluster {lbl} because all members are concepts.")
                    continue

                # Query database to check if we already have a concept node with these exact sources in provenance
                source_set = set(cluster_node_ids)
                already_synthesized = False
                
                cursor.execute("SELECT id, properties FROM nodes WHERE node_type = 'concept'")
                concept_rows = cursor.fetchall()
                for c_id, c_props_json in concept_rows:
                    c_props = json.loads(c_props_json) if c_props_json else {}
                    prov = c_props.get("provenance", {})
                    if prov and prov.get("source_type") == "librarian":
                        prov_sources = set(prov.get("source_ids", []))
                        if prov_sources == source_set:
                            already_synthesized = True
                            break

                if already_synthesized:
                    logger.debug(f"Cluster {lbl} already has synthesized concept.")
                    continue

                # Generate a synthesized concept!
                concept_id = f"concept-{uuid.uuid4()}"
                concept_name = f"Synthesized Concept Group {lbl + 1}"
                
                # Generate a smart consolidated title/description based on cluster member names
                member_names = [node_names[idx] for idx in indices]
                description = f"A synthesized higher-order concept grouping similar facts: {', '.join(member_names)}."
                
                provenance = {
                    "source_type": "librarian",
                    "source_ids": cluster_node_ids,
                    "algorithm": "dbscan",
                    "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "confidence": 0.85
                }

                properties = {
                    "description": description,
                    "provenance": provenance
                }

                # Create the synthesized node
                try:
                    self.db.create_node(
                        node_id=concept_id,
                        name=concept_name,
                        node_type="concept",
                        properties=properties
                    )
                    synthesized_count += 1
                    synthesized_nodes.append({
                        "id": concept_id,
                        "name": concept_name,
                        "members": member_names
                    })

                    # Link the synthesized concept to all member nodes with weight/confidence
                    for nid in cluster_node_ids:
                        edge_id = f"edge-{concept_id}-{nid}-implements"
                        self.db.create_edge(
                            edge_id=edge_id,
                            from_node_id=concept_id,
                            to_node_id=nid,
                            relationship_type="implements",
                            confidence=0.85,
                            weight=1.0,
                            properties={"reason": "Automatic conceptual grouping from librarian DBSCAN cycle"}
                        )
                except Exception as e:
                    logger.error(f"Failed to create synthesized concept {concept_id}: {e}")

            duration = (datetime.datetime.now() - start_time).total_seconds()
            logger.info(f"Librarian cycle completed in {duration:.4f}s. Synthesized {synthesized_count} concepts.")

            # Record running log to database
            try:
                with self.db.conn as conn:
                    conn.execute(
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
                        """
                    )
                    conn.execute(
                        """
                        INSERT INTO librarian_runs (
                            timestamp, duration, clusters_detected, synthesized_concepts, duplicates_detected, edges_created
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            datetime.datetime.now(datetime.timezone.utc).isoformat(),
                            duration,
                            len(clusters),
                            synthesized_count,
                            len(duplicates),
                            discovered_edges_count
                        )
                    )
            except Exception as e:
                logger.error(f"Failed to log Librarian run stats: {e}")

            return {
                "status": "COMPLETED",
                "duration_seconds": duration,
                "clusters_detected": len(clusters),
                "synthesized_concepts": synthesized_count,
                "duplicates_detected": len(duplicates),
                "edges_created": discovered_edges_count,
                "synthesized_nodes": synthesized_nodes,
                "duplicates": duplicates
            }

        except Exception as e:
            logger.error(f"Critical error during Librarian cycle: {e}", exc_info=True)
            raise SearchError(
                code=MEMORY_LAYER_DEGRADED,
                message=f"Librarian cycle failed: {e}",
                subsystem="LIBRARIAN",
                retry_safe=True
            )
