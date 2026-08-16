import json
import datetime
import logging
from typing import List, Dict, Any
from memory.errors import SearchError, SQLITE_TRANSACTION_FAILED

logger = logging.getLogger(__name__)


def archive_node(db, node_id: str, reason: str = "Unused/superseded") -> Dict[str, Any]:
    """Archives an entity and its historical records."""
    try:
        cursor = db.conn.cursor()
        cursor.execute("SELECT properties FROM nodes WHERE id = ?", (node_id,))
        row = cursor.fetchone()
        if not row:
            raise ValueError(f"Node {node_id} not found")

        properties = json.loads(row[0]) if row[0] else {}
        properties["archived_metadata"] = {
            "archived_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "reason": reason,
        }

        updated_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

        with db.conn as conn:
            conn.execute(
                """
                UPDATE nodes SET
                    status = 'ARCHIVED',
                    updated_at = ?,
                    properties = ?
                WHERE id = ?
                """,
                (updated_at, json.dumps(properties), node_id),
            )
            # Sync FTS
            db._sync_node_fts_and_embedding(node_id, conn)

        return {
            "id": node_id,
            "status": "ARCHIVED",
            "archived_at": properties["archived_metadata"]["archived_at"],
            "reason": reason,
        }
    except Exception as e:
        logger.error(f"Failed to archive node {node_id}: {e}")
        raise SearchError(
            code=SQLITE_TRANSACTION_FAILED,
            message=f"Failed to archive node {node_id}: {e}",
            subsystem="LIFECYCLE",
            retry_safe=True,
        )


def restore_node(db, node_id: str) -> Dict[str, Any]:
    """Restores an archived entity to ACTIVE status."""
    try:
        cursor = db.conn.cursor()
        cursor.execute("SELECT properties FROM nodes WHERE id = ?", (node_id,))
        row = cursor.fetchone()
        if not row:
            raise ValueError(f"Node {node_id} not found")

        properties = json.loads(row[0]) if row[0] else {}
        properties.pop("archived_metadata", None)

        updated_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

        with db.conn as conn:
            conn.execute(
                """
                UPDATE nodes SET
                    status = 'ACTIVE',
                    updated_at = ?,
                    properties = ?
                WHERE id = ?
                """,
                (updated_at, json.dumps(properties), node_id),
            )
            # Sync FTS
            db._sync_node_fts_and_embedding(node_id, conn)

        return {"id": node_id, "status": "ACTIVE"}
    except Exception as e:
        logger.error(f"Failed to restore node {node_id}: {e}")
        raise SearchError(
            code=SQLITE_TRANSACTION_FAILED,
            message=f"Failed to restore node {node_id}: {e}",
            subsystem="LIFECYCLE",
            retry_safe=True,
        )


def list_orphans(db) -> List[Dict[str, Any]]:
    """Identifies and lists orphaned entities (nodes with zero edges)."""
    try:
        cursor = db.conn.cursor()
        cursor.execute("""
            SELECT id, name, node_type, created_at, status
            FROM nodes
            WHERE id NOT IN (SELECT DISTINCT from_node_id FROM edges)
              AND id NOT IN (SELECT DISTINCT to_node_id FROM edges)
            """)
        rows = cursor.fetchall()
        return [
            {
                "id": r[0],
                "name": r[1],
                "node_type": r[2],
                "created_at": r[3],
                "status": r[4],
            }
            for r in rows
        ]
    except Exception as e:
        logger.error(f"Failed to list orphans: {e}")
        raise SearchError(
            code=SQLITE_TRANSACTION_FAILED,
            message=f"Failed to list orphans: {e}",
            subsystem="LIFECYCLE",
            retry_safe=True,
        )


def prune_stale(db, max_age_days: int, dry_run: bool = False) -> List[Dict[str, Any]]:
    """
    Identifies stale/archived memories older than max_age_days and prunes them.
    In dry_run mode, returns them without actual deletion.
    """
    try:
        limit_date = (
            datetime.datetime.now(datetime.timezone.utc)
            - datetime.timedelta(days=max_age_days)
        ).isoformat()

        cursor = db.conn.cursor()
        cursor.execute(
            """
            SELECT id, name, node_type, status, created_at
            FROM nodes
            WHERE (status = 'STALE' OR status = 'ARCHIVED')
              AND created_at <= ?
            """,
            (limit_date,),
        )
        rows = cursor.fetchall()

        candidates = [
            {
                "id": r[0],
                "name": r[1],
                "node_type": r[2],
                "status": r[3],
                "created_at": r[4],
            }
            for r in rows
        ]

        if not dry_run and candidates:
            # Delete candidates
            for cand in candidates:
                db.delete_node(cand["id"])
            logger.info(
                f"Successfully pruned {len(candidates)} stale nodes from database."
            )

        return candidates
    except Exception as e:
        logger.error(f"Failed to prune stale nodes: {e}")
        raise SearchError(
            code=SQLITE_TRANSACTION_FAILED,
            message=f"Failed to prune stale nodes: {e}",
            subsystem="LIFECYCLE",
            retry_safe=True,
        )
