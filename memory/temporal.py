import logging
from typing import List, Dict, Any
from datetime import datetime
from memory.errors import SearchError, TEMPORAL_QUERY_FAILED

logger = logging.getLogger(__name__)

def parse_iso(ts: str) -> datetime:
    """Parses an ISO format timestamp, with fallback parsing to handle variations."""
    if not ts:
        raise ValueError("Empty timestamp")
    try:
        # replace Z with +00:00 for python's fromisoformat
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        return datetime.fromisoformat(ts)
    except Exception as e:
        # try common variations
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d"):
            try:
                return datetime.strptime(ts, fmt)
            except ValueError:
                continue
        raise ValueError(f"Could not parse timestamp '{ts}': {e}")

def diff_knowledge_state(db, from_timestamp: str, to_timestamp: str) -> Dict[str, Any]:
    """
    Computes a deterministic delta between two points in time.
    """
    try:
        from_snap = db.get_graph_state_as_of(from_timestamp)
        to_snap = db.get_graph_state_as_of(to_timestamp)
    except Exception as e:
        logger.error(f"Failed to fetch snapshots for temporal diff: {e}")
        raise SearchError(
            code=TEMPORAL_QUERY_FAILED,
            message=f"Failed to fetch snapshots for temporal diff: {e}",
            subsystem="TEMPORAL",
            retry_safe=True
        )

    from_nodes = {n["id"]: n for n in from_snap["nodes"]}
    to_nodes = {n["id"]: n for n in to_snap["nodes"]}
    from_edges = {e["id"]: e for e in from_snap["edges"]}
    to_edges = {e["id"]: e for e in to_snap["edges"]}

    added = []
    removed = []
    changed = []

    for nid, node in to_nodes.items():
        if nid not in from_nodes:
            added.append(node)
        else:
            old_node = from_nodes[nid]
            if node.get("updated_at") != old_node.get("updated_at") or node.get("properties") != old_node.get("properties"):
                changed.append({
                    "id": nid,
                    "before": old_node,
                    "after": node
                })

    for nid, node in from_nodes.items():
        if nid not in to_nodes:
            removed.append(node)

    edges_added = []
    edges_removed = []

    for eid, edge in to_edges.items():
        if eid not in from_edges:
            edges_added.append(edge)

    for eid, edge in from_edges.items():
        if eid not in to_edges:
            edges_removed.append(edge)

    return {
        "from": from_timestamp,
        "to": to_timestamp,
        "added": added,
        "removed": removed,
        "changed": changed,
        "edges_added": edges_added,
        "edges_removed": edges_removed
    }

def query_timeline(
    db,
    entity: str = None,
    entity_type: str = None,
    start: str = None,
    end: str = None,
    order: str = "ASC",
    limit: int = 50,
    offset: int = 0
) -> List[Dict[str, Any]]:
    """
    Queries timeline of events (nodes, observations) chronologically with strict filtering and pagination.
    """
    try:
        # Build strict dynamic SQL querying both nodes and observations
        conn = db.conn
        cursor = conn.cursor()

        # Let's collect all nodes and observations and merge them chronologically
        query_nodes = """
            SELECT id, name, node_type, created_at, occurred_at, properties
            FROM nodes
            WHERE 1=1
        """
        params_nodes = []

        if entity_type:
            query_nodes += " AND node_type = ?"
            params_nodes.append(entity_type)
        if entity:
            query_nodes += " AND (name LIKE ? OR properties LIKE ?)"
            params_nodes.append(f"%{entity}%")
            params_nodes.append(f"%{entity}%")

        cursor.execute(query_nodes, tuple(params_nodes))
        node_rows = cursor.fetchall()

        events = []
        for r in node_rows:
            # Timestamp to sort on: preferred is occurred_at if present, else created_at
            ts = r[4] if r[4] else r[3]
            events.append({
                "event_type": "entity_creation",
                "id": r[0],
                "name": r[1],
                "type": r[2],
                "timestamp": ts,
                "created_at": r[3],
                "occurred_at": r[4],
                "properties": r[5]
            })

        # Observations timeline
        query_obs = """
            SELECT o.id, o.content, o.created_at, o.occurred_at, n.name, n.node_type
            FROM observations o
            JOIN nodes n ON o.entity_id = n.id
            WHERE 1=1
        """
        params_obs = []

        if entity_type:
            query_obs += " AND n.node_type = ?"
            params_obs.append(entity_type)
        if entity:
            query_obs += " AND (n.name LIKE ? OR o.content LIKE ?)"
            params_obs.append(f"%{entity}%")
            params_obs.append(f"%{entity}%")

        cursor.execute(query_obs, tuple(params_obs))
        obs_rows = cursor.fetchall()

        for r in obs_rows:
            ts = r[3] if r[3] else r[2]
            events.append({
                "event_type": "observation_recorded",
                "id": r[0],
                "content": r[1],
                "entity_name": r[4],
                "entity_type": r[5],
                "timestamp": ts,
                "created_at": r[2],
                "occurred_at": r[3]
            })

        # Filters for time window
        filtered_events = []
        for ev in events:
            ev_ts_str = ev["timestamp"]
            if not ev_ts_str:
                continue
            
            try:
                ev_ts = parse_iso(ev_ts_str)
            except Exception:
                # Fallback to lexical string comparison
                ev_ts = ev_ts_str

            if start:
                try:
                    start_ts = parse_iso(start)
                    if isinstance(ev_ts, datetime):
                        if ev_ts < start_ts:
                            continue
                    else:
                        if ev_ts_str < start:
                            continue
                except Exception:
                    if ev_ts_str < start:
                        continue

            if end:
                try:
                    end_ts = parse_iso(end)
                    if isinstance(ev_ts, datetime):
                        if ev_ts > end_ts:
                            continue
                    else:
                        if ev_ts_str > end:
                            continue
                except Exception:
                    if ev_ts_str > end:
                        continue

            filtered_events.append(ev)

        # Sort
        reverse = (order.upper() == "DESC")
        filtered_events.sort(key=lambda x: x["timestamp"], reverse=reverse)

        # Paginate
        paginated = filtered_events[offset:offset + limit]
        return paginated

    except Exception as e:
        logger.error(f"Timeline query failed: {e}")
        raise SearchError(
            code=TEMPORAL_QUERY_FAILED,
            message=f"Timeline query failed: {e}",
            subsystem="TEMPORAL",
            retry_safe=True
        )

def get_temporal_neighbors(
    db,
    node_id: str,
    direction: str = "both",
    depth: int = 1
) -> List[Dict[str, Any]]:
    """
    Explores connected nodes along temporal edges or nodes that are chronologically related.
    Compares the chronological state (occurred_at/created_at) of neighbors to the anchor node.
    - before: neighbors that happened before the anchor node.
    - after: neighbors that happened after the anchor node.
    - both: any chronological neighbors.
    """
    try:
        # Get anchor node time
        cursor = db.conn.cursor()
        cursor.execute("SELECT name, created_at, occurred_at FROM nodes WHERE id = ?", (node_id,))
        anchor_row = cursor.fetchone()
        if not anchor_row:
            raise SearchError(
                code=TEMPORAL_QUERY_FAILED,
                message=f"Anchor node '{node_id}' not found.",
                subsystem="TEMPORAL",
                retry_safe=False
            )

        anchor_name, anchor_created, anchor_occurred = anchor_row
        anchor_ts_str = anchor_occurred if anchor_occurred else anchor_created
        if not anchor_ts_str:
            # If no timestamp, default to now
            anchor_ts = datetime.utcnow()
        else:
            anchor_ts = parse_iso(anchor_ts_str)

        # Standard traversal with path tracking & cycle prevention
        visited = {node_id}
        queue = [(node_id, [node_id])]  # (current_node, path)
        results = []

        while queue:
            curr_id, path = queue.pop(0)
            if len(path) - 1 >= depth:
                continue

            # Fetch edges connected to curr_id
            cursor.execute(
                """
                SELECT id, from_node_id, to_node_id, relationship_type, created_at, properties
                FROM edges
                WHERE from_node_id = ? OR to_node_id = ?
                """,
                (curr_id, curr_id)
            )
            edges_rows = cursor.fetchall()

            for edge in edges_rows:
                edge_id, from_id, to_id, rel_type, edge_created, edge_properties = edge
                neighbor_id = to_id if from_id == curr_id else from_id

                if neighbor_id in visited:
                    continue

                # Fetch neighbor node time
                cursor.execute(
                    "SELECT id, name, node_type, created_at, occurred_at, properties FROM nodes WHERE id = ?",
                    (neighbor_id,)
                )
                neigh_row = cursor.fetchone()
                if not neigh_row:
                    continue

                nid, nname, ntype, ncreated, noccurred, nproperties = neigh_row
                neigh_ts_str = noccurred if noccurred else ncreated
                
                try:
                    neigh_ts = parse_iso(neigh_ts_str)
                except Exception:
                    neigh_ts = datetime.utcnow()

                # Determine direction match
                is_valid = False
                if direction == "both":
                    is_valid = True
                elif direction == "before" and neigh_ts < anchor_ts:
                    is_valid = True
                elif direction == "after" and neigh_ts > anchor_ts:
                    is_valid = True

                if is_valid:
                    visited.add(neighbor_id)
                    new_path = path + [neighbor_id]
                    queue.append((neighbor_id, new_path))
                    
                    results.append({
                        "node": {
                            "id": nid,
                            "name": nname,
                            "node_type": ntype,
                            "created_at": ncreated,
                            "occurred_at": noccurred,
                            "properties": nproperties
                        },
                        "edge": {
                            "id": edge_id,
                            "from_node_id": from_id,
                            "to_node_id": to_id,
                            "relationship_type": rel_type,
                            "created_at": edge_created,
                        },
                        "path": new_path,
                        "chronological_diff_seconds": (neigh_ts - anchor_ts).total_seconds()
                    })

        return results

    except SearchError:
        raise
    except Exception as e:
        logger.error(f"Temporal neighbor exploration failed: {e}")
        raise SearchError(
            code=TEMPORAL_QUERY_FAILED,
            message=f"Temporal neighbor exploration failed: {e}",
            subsystem="TEMPORAL",
            retry_safe=True
        )
