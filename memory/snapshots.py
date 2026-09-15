"""Point-in-time reconstruction of the knowledge graph."""

from __future__ import annotations

import logging
from typing import List

from graph.rows import (
    EDGE_COLUMNS,
    NODE_COLUMNS,
    edge_from_row,
    fetch_observations,
    node_from_row,
)

logger = logging.getLogger(__name__)

__all__ = ["SnapshotMixin"]


class SnapshotMixin:
    """Time-travel half of :class:`~core.database.VectorDB`."""

    def get_graph_state_as_of(self, as_of_iso_timestamp: str) -> dict:
        """Returns the graph as it existed at ``as_of_iso_timestamp`` (UTC ISO).

        Nodes, edges and observations created after the cutoff are excluded, and
        edges are kept only when both endpoints existed at that time.
        """
        conn = self.conn
        max_vars = self.settings.sqlite_max_variables

        node_rows = conn.execute(
            f"SELECT {NODE_COLUMNS} FROM nodes WHERE created_at <= ?",
            (as_of_iso_timestamp,),
        ).fetchall()

        nodes: List[dict] = [node_from_row(row) for row in node_rows]
        node_ids = [node["id"] for node in nodes]
        observations = fetch_observations(
            conn, node_ids, max_vars, created_before=as_of_iso_timestamp
        )
        for node in nodes:
            node["observations"] = observations.get(node["id"], [])

        known = set(node_ids)
        edge_rows = conn.execute(
            f"SELECT {EDGE_COLUMNS} FROM edges WHERE created_at <= ?",
            (as_of_iso_timestamp,),
        ).fetchall()
        edges = [
            edge_from_row(row)
            for row in edge_rows
            if row[1] in known and row[2] in known
        ]

        return {"nodes": nodes, "edges": edges}
