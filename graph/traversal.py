"""Breadth-first graph traversal and connectivity helpers.

All traversal here is level-synchronous and batched: one query per hop instead
of one per node, and one query for the observations of the whole frontier.
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Sequence, Set, Tuple

from core.sqlite import chunked
from graph.rows import fetch_edges_between, fetch_nodes, fetch_observations

logger = logging.getLogger(__name__)

__all__ = ["GraphTraversalMixin", "UnionFind"]


class UnionFind:
    """Disjoint-set over node ids, used to answer reachability in ~O(1).

    The semantic radar asks "is there a path between A and B?" for every pair
    of nodes. Running a BFS per pair is quadratic in edges; a single union-find
    built once answers all of them, and stays correct as edges are added.
    """

    def __init__(self, items: Iterable[str] = ()) -> None:
        self._parent: Dict[str, str] = {item: item for item in items}

    def find(self, item: str) -> str:
        self._parent.setdefault(item, item)
        root = item
        while self._parent[root] != root:
            root = self._parent[root]
        # Path compression keeps repeated lookups flat.
        while self._parent[item] != root:
            self._parent[item], item = root, self._parent[item]
        return root

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self._parent[right_root] = left_root

    def connected(self, left: str, right: str) -> bool:
        return self.find(left) == self.find(right)


class GraphTraversalMixin:
    """Read-only neighbourhood queries for :class:`~core.database.VectorDB`."""

    def get_hologram(self, node_id: str, depth: int = 1) -> dict:
        """Returns a node with its observations and its ``depth``-hop subgraph."""
        exists = self.conn.execute(
            "SELECT 1 FROM nodes WHERE id = ? LIMIT 1", (node_id,)
        ).fetchone()
        if not exists:
            raise ValueError(f"Node {node_id} not found")

        visited = self._expand(node_id, depth)
        conn = self.conn
        max_vars = self.settings.sqlite_max_variables

        nodes_by_id = fetch_nodes(conn, list(visited), max_vars)
        observations = fetch_observations(conn, list(nodes_by_id), max_vars)
        for identifier, node in nodes_by_id.items():
            node["observations"] = observations.get(identifier, [])

        return {
            "nodes": list(nodes_by_id.values()),
            "edges": fetch_edges_between(conn, visited, max_vars),
        }

    def get_neighbors(self, node_id: str, depth: int = 1) -> List[dict]:
        """Returns the nodes reachable from ``node_id`` within ``depth`` hops."""
        return self.get_hologram(node_id, depth=depth)["nodes"]

    # -- internals ---------------------------------------------------------
    def _expand(self, node_id: str, depth: int) -> Set[str]:
        """Collects the ids reachable within ``depth`` hops, one query per hop."""
        visited: Set[str] = {node_id}
        frontier: Set[str] = {node_id}
        max_vars = self.settings.sqlite_max_variables
        cursor = self.conn.cursor()

        for _ in range(max(0, depth)):
            if not frontier:
                break
            next_frontier: Set[str] = set()
            for batch in chunked(list(frontier), max_vars // 2):
                placeholders = ",".join("?" for _ in batch)
                cursor.execute(
                    f"""
                    SELECT from_node_id, to_node_id FROM edges
                    WHERE from_node_id IN ({placeholders})
                       OR to_node_id IN ({placeholders})
                    """,
                    batch + batch,
                )
                for from_id, to_id in cursor.fetchall():
                    for candidate in (from_id, to_id):
                        if candidate not in visited:
                            visited.add(candidate)
                            next_frontier.add(candidate)
            frontier = next_frontier
        return visited

    def edge_pairs(self) -> List[Tuple[str, str]]:
        """Returns every ``(from, to)`` pair in the graph."""
        return [
            (row[0], row[1])
            for row in self.conn.execute(
                "SELECT from_node_id, to_node_id FROM edges"
            ).fetchall()
        ]

    def connectivity(self, node_ids: Sequence[str] = ()) -> UnionFind:
        """Builds a union-find over the whole edge set in one pass."""
        components = UnionFind(node_ids)
        for from_id, to_id in self.edge_pairs():
            components.union(from_id, to_id)
        return components

    def _has_path(self, start_id: str, end_id: str) -> bool:
        """True when ``start_id`` and ``end_id`` are in the same component."""
        if start_id == end_id:
            return True

        visited = {start_id}
        queue = [start_id]
        cursor = self.conn.cursor()
        while queue:
            current = queue.pop()
            cursor.execute(
                "SELECT from_node_id, to_node_id FROM edges "
                "WHERE from_node_id = ? OR to_node_id = ?",
                (current, current),
            )
            for from_id, to_id in cursor.fetchall():
                neighbor = to_id if from_id == current else from_id
                if neighbor == end_id:
                    return True
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return False
