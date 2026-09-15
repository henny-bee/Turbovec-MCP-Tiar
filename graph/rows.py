"""Row-to-dict mappers and batched hydration helpers.

Hydrating observations one node at a time is the classic N+1 query; every
retrieval path here fetches them for the whole result set in a single
statement instead.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Dict, Iterable, List, Optional, Sequence

from core.sqlite import chunked

__all__ = [
    "NODE_COLUMNS",
    "NODE_COLUMNS_WITH_STATUS",
    "EDGE_COLUMNS",
    "node_from_row",
    "edge_from_row",
    "observation_from_row",
    "fetch_observations",
    "fetch_nodes",
    "fetch_edges_between",
]

NODE_COLUMNS = (
    "id, name, node_type, project_id, created_at, updated_at, occurred_at, "
    "certainty, salience_score, retrieval_count, properties"
)
NODE_COLUMNS_WITH_STATUS = (
    "id, name, node_type, project_id, created_at, updated_at, occurred_at, "
    "certainty, salience_score, retrieval_count, status, properties"
)
EDGE_COLUMNS = (
    "id, from_node_id, to_node_id, relationship_type, confidence, weight, "
    "created_at, properties"
)
_OBSERVATION_COLUMNS = "entity_id, id, content, certainty, created_at, occurred_at"


def _loads(blob) -> dict:
    return json.loads(blob) if blob else {}


def node_from_row(row: Sequence, with_status: bool = False) -> dict:
    """Maps a row selected with ``NODE_COLUMNS``/``..._WITH_STATUS`` to a dict."""
    node = {
        "id": row[0],
        "name": row[1],
        "node_type": row[2],
        "project_id": row[3],
        "created_at": row[4],
        "updated_at": row[5],
        "occurred_at": row[6],
        "certainty": row[7],
        "salience_score": row[8],
        "retrieval_count": row[9],
    }
    if with_status:
        node["status"] = row[10]
    node["properties"] = _loads(row[11 if with_status else 10])
    return node


def edge_from_row(row: Sequence) -> dict:
    """Maps a row selected with ``EDGE_COLUMNS`` to a dict."""
    return {
        "id": row[0],
        "from_node_id": row[1],
        "to_node_id": row[2],
        "relationship_type": row[3],
        "confidence": row[4],
        "weight": row[5],
        "created_at": row[6],
        "properties": _loads(row[7]),
    }


def observation_from_row(row: Sequence) -> dict:
    """Maps a row selected with ``_OBSERVATION_COLUMNS`` (minus entity_id)."""
    return {
        "id": row[1],
        "content": row[2],
        "certainty": row[3],
        "created_at": row[4],
        "occurred_at": row[5],
    }


def fetch_observations(
    conn: sqlite3.Connection,
    node_ids: Sequence[str],
    max_variables: int = 900,
    order: str = "ASC",
    created_before: Optional[str] = None,
) -> Dict[str, List[dict]]:
    """Loads observations for many nodes at once, grouped by ``entity_id``."""
    grouped: Dict[str, List[dict]] = {node_id: [] for node_id in node_ids}
    if not node_ids:
        return grouped

    direction = "DESC" if order.upper() == "DESC" else "ASC"
    cursor = conn.cursor()
    for batch in chunked(list(node_ids), max_variables - 1):
        placeholders = ",".join("?" for _ in batch)
        sql = (
            f"SELECT {_OBSERVATION_COLUMNS} FROM observations "
            f"WHERE entity_id IN ({placeholders})"
        )
        params: List = list(batch)
        if created_before is not None:
            sql += " AND created_at <= ?"
            params.append(created_before)
        sql += f" ORDER BY created_at {direction}, rowid {direction}"
        cursor.execute(sql, params)
        for row in cursor.fetchall():
            grouped.setdefault(row[0], []).append(observation_from_row(row))
    return grouped


def fetch_nodes(
    conn: sqlite3.Connection,
    node_ids: Sequence[str],
    max_variables: int = 900,
    with_status: bool = False,
) -> Dict[str, dict]:
    """Loads many nodes by id in batched ``IN (...)`` queries."""
    columns = NODE_COLUMNS_WITH_STATUS if with_status else NODE_COLUMNS
    nodes: Dict[str, dict] = {}
    if not node_ids:
        return nodes

    cursor = conn.cursor()
    for batch in chunked(list(node_ids), max_variables):
        placeholders = ",".join("?" for _ in batch)
        cursor.execute(
            f"SELECT {columns} FROM nodes WHERE id IN ({placeholders})", batch
        )
        for row in cursor.fetchall():
            nodes[row[0]] = node_from_row(row, with_status=with_status)
    return nodes


def fetch_edges_between(
    conn: sqlite3.Connection,
    node_ids: Iterable[str],
    max_variables: int = 900,
) -> List[dict]:
    """Loads every edge whose both endpoints are inside ``node_ids``."""
    ids = list(node_ids)
    if not ids:
        return []

    id_set = set(ids)
    edges: List[dict] = []
    seen: set = set()
    cursor = conn.cursor()
    for batch in chunked(ids, max_variables):
        placeholders = ",".join("?" for _ in batch)
        cursor.execute(
            f"SELECT {EDGE_COLUMNS} FROM edges WHERE from_node_id IN ({placeholders})",
            batch,
        )
        for row in cursor.fetchall():
            if row[2] in id_set and row[0] not in seen:
                seen.add(row[0])
                edges.append(edge_from_row(row))
    return edges
