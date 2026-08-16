import logging
import json
from typing import Dict, List, Any, Set, Tuple

logger = logging.getLogger(__name__)

class GraphAnalytics:
    def __init__(self, db):
        self.db = db
        self._cached_results = {}
        self._last_cache_time = 0.0

    def get_nodes_and_edges(self) -> Tuple[List[str], List[Tuple[str, str, float]]]:
        """Fetches all node IDs and edge tuples (from_id, to_id, weight)."""
        cursor = self.db.conn.cursor()
        cursor.execute("SELECT id FROM nodes")
        nodes = [row[0] for row in cursor.fetchall()]

        cursor.execute("SELECT from_node_id, to_node_id, weight FROM edges")
        edges = []
        for row in cursor.fetchall():
            edges.append((row[0], row[1], float(row[2]) if row[2] else 1.0))

        return nodes, edges

    def compute_degree_centrality(self, nodes: List[str], edges: List[Tuple[str, str, float]]) -> Dict[str, float]:
        """Calculates normalized degree centrality for all nodes."""
        n = len(nodes)
        if n <= 1:
            return {node: 0.0 for node in nodes}

        degrees = {node: 0.0 for node in nodes}
        for u, v, _ in edges:
            if u in degrees:
                degrees[u] += 1.0
            if v in degrees:
                degrees[v] += 1.0

        # Normalize by n - 1
        denom = float(n - 1)
        return {node: deg / denom for node, deg in degrees.items()}

    def compute_pagerank(self, nodes: List[str], edges: List[Tuple[str, str, float]], d: float = 0.85, max_iter: int = 100, tol: float = 1.0e-6) -> Dict[str, float]:
        """Calculates PageRank score for all nodes with damping factor d."""
        n = len(nodes)
        if n == 0:
            return {}
        if n == 1:
            return {nodes[0]: 1.0}

        # Setup adjacency list (out-edges) and in-edges
        out_degrees = {node: 0 for node in nodes}
        in_edges = {node: [] for node in nodes}

        for u, v, _ in edges:
            if u in out_degrees and v in in_edges:
                out_degrees[u] += 1
                in_edges[v].append(u)

        # Initialize ranks
        pagerank = {node: 1.0 / n for node in nodes}
        
        for iteration in range(max_iter):
            new_pagerank = {}
            # Handle sink nodes (nodes with 0 out-degree)
            sink_sum = sum(pagerank[node] for node in nodes if out_degrees[node] == 0)
            
            err = 0.0
            for node in nodes:
                # PageRank formula: (1 - d)/N + d * (sum(PR(v) / L(v)) + sink_sum / N)
                rank_sum = sum(pagerank[v] / out_degrees[v] for v in in_edges[node])
                rank_sum += sink_sum / n
                
                val = ((1.0 - d) / n) + d * rank_sum
                new_pagerank[node] = val
                err += abs(val - pagerank[node])

            pagerank = new_pagerank
            if err < tol:
                logger.debug(f"PageRank converged in {iteration + 1} iterations.")
                break

        return pagerank

    def compute_connected_components(self, nodes: List[str], edges: List[Tuple[str, str, float]]) -> List[List[str]]:
        """Finds all connected components in the graph."""
        adj = {node: set() for node in nodes}
        for u, v, _ in edges:
            if u in adj and v in adj:
                adj[u].add(v)
                adj[v].add(u)

        visited = set()
        components = []

        for node in nodes:
            if node not in visited:
                component = []
                queue = [node]
                visited.add(node)
                while queue:
                    curr = queue.pop(0)
                    component.append(curr)
                    for neighbor in adj[curr]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
                components.append(component)

        return components

    def compute_communities_lpa(self, nodes: List[str], edges: List[Tuple[str, str, float]], max_iter: int = 20) -> Dict[str, int]:
        """
        Detects communities using Label Propagation Algorithm (LPA).
        Returns a dictionary mapping node_id to community_id (int).
        """
        if not nodes:
            return {}

        # Set unique initial labels
        labels = {node: idx for idx, node in enumerate(nodes)}

        # Build bidirectional adjacency list
        adj = {node: [] for node in nodes}
        for u, v, w in edges:
            if u in adj and v in adj:
                adj[u].append((v, w))
                adj[v].append((u, w))

        import random
        # Seed for determinism or pseudo-random shuffles
        rng = random.Random(42)

        for _ in range(max_iter):
            shuffled_nodes = list(nodes)
            rng.shuffle(shuffled_nodes)

            changed = False
            for node in shuffled_nodes:
                if not adj[node]:
                    continue

                # Count labels among neighbors
                label_weights = {}
                for neighbor, weight in adj[node]:
                    neigh_label = labels[neighbor]
                    label_weights[neigh_label] = label_weights.get(neigh_label, 0.0) + weight

                # Get maximum weight label
                max_val = -1.0
                best_labels = []
                for label, weight in label_weights.items():
                    if weight > max_val:
                        max_val = weight
                        best_labels = [label]
                    elif weight == max_val:
                        best_labels.append(label)

                # Pick one of the best labels
                chosen_label = rng.choice(best_labels)
                if labels[node] != chosen_label:
                    labels[node] = chosen_label
                    changed = True

            if not changed:
                break

        # Re-index communities consecutively from 0
        unique_labels = sorted(list(set(labels.values())))
        label_map = {old_lbl: idx for idx, old_lbl in enumerate(unique_labels)}

        return {node: label_map[lbl] for node, lbl in labels.items()}

    def analyze_graph(self) -> Dict[str, Any]:
        """Runs connected components, LPA community detection, centrality, and PageRank."""
        # Check cache (expire in 5 seconds to prevent hammering during intensive search sessions)
        import time
        now = time.time()
        if self._cached_results and (now - self._last_cache_time) < 5.0:
            return self._cached_results

        nodes, edges = self.get_nodes_and_edges()
        if not nodes:
            return {
                "nodes": 0,
                "edges": 0,
                "communities": 0,
                "components": 0,
                "top_nodes": []
            }

        centrality = self.compute_degree_centrality(nodes, edges)
        pagerank = self.compute_pagerank(nodes, edges)
        components = self.compute_connected_components(nodes, edges)
        communities = self.compute_communities_lpa(nodes, edges)

        # Build node registry
        cursor = self.db.conn.cursor()
        cursor.execute("SELECT id, name, node_type FROM nodes")
        node_details = {row[0]: {"name": row[1], "node_type": row[2]} for row in cursor.fetchall()}

        top_nodes = []
        for node in nodes:
            details = node_details.get(node, {"name": "Unknown", "node_type": "entity"})
            top_nodes.append({
                "id": node,
                "name": details["name"],
                "node_type": details["node_type"],
                "pagerank": pagerank.get(node, 0.0),
                "centrality": centrality.get(node, 0.0),
                "community": communities.get(node, 0)
            })

        # Sort by PageRank
        top_nodes.sort(key=lambda x: x["pagerank"], reverse=True)

        results = {
            "nodes": len(nodes),
            "edges": len(edges),
            "communities": len(set(communities.values())) if communities else 0,
            "components": len(components),
            "top_nodes": top_nodes[:20]  # Return top 20 nodes
        }

        self._cached_results = results
        self._last_cache_time = now
        return results
