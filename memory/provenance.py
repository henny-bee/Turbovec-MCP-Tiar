import sqlite3
import datetime
import uuid
import logging
from typing import List, Dict, Any
from memory.errors import SearchError, SQLITE_TRANSACTION_FAILED

logger = logging.getLogger(__name__)

class BottleManager:
    def __init__(self, db):
        self.db = db
        self._init_bottles_table()

    def _init_bottles_table(self) -> None:
        """Initializes the bottles table in the SQLite database."""
        try:
            with self.db.conn as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS bottles (
                        id TEXT PRIMARY KEY,
                        message TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        author_session_id TEXT,
                        priority TEXT NOT NULL,
                        expires_at TEXT,
                        acknowledged INTEGER NOT NULL DEFAULT 0
                    );
                    """
                )
        except Exception as e:
            logger.error(f"Failed to initialize bottles table: {e}")

    def create_bottle(self, message: str, priority: str = "medium", expires_at: str = None, author_session_id: str = None) -> Dict[str, Any]:
        """Creates a standardized bottle note in the database."""
        bottle_id = str(uuid.uuid4())
        created_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        
        try:
            with self.db.conn as conn:
                conn.execute(
                    """
                    INSERT INTO bottles (id, message, created_at, author_session_id, priority, expires_at, acknowledged)
                    VALUES (?, ?, ?, ?, ?, ?, 0)
                    """,
                    (bottle_id, message, created_at, author_session_id, priority, expires_at)
                )
            return {
                "id": bottle_id,
                "message": message,
                "created_at": created_at,
                "author_session_id": author_session_id,
                "priority": priority,
                "expires_at": expires_at,
                "acknowledged": False
            }
        except Exception as e:
            logger.error(f"Failed to create bottle: {e}")
            raise SearchError(
                code=SQLITE_TRANSACTION_FAILED,
                message=f"Failed to create bottle note: {e}",
                subsystem="BOTTLES",
                retry_safe=True
            )

    def list_bottles(self, include_acknowledged: bool = False) -> List[Dict[str, Any]]:
        """Lists active and optionally acknowledged bottles."""
        try:
            cursor = self.db.conn.cursor()
            if include_acknowledged:
                cursor.execute("SELECT id, message, created_at, author_session_id, priority, expires_at, acknowledged FROM bottles ORDER BY created_at DESC")
            else:
                cursor.execute("SELECT id, message, created_at, author_session_id, priority, expires_at, acknowledged FROM bottles WHERE acknowledged = 0 ORDER BY created_at DESC")
            
            rows = cursor.fetchall()
            now = datetime.datetime.now(datetime.timezone.utc)
            
            results = []
            for r in rows:
                expires_str = r[5]
                # filter out expired notes
                if expires_str:
                    try:
                        # replace Z with +00:00 for python's fromisoformat
                        iso_ts = expires_str[:-1] + "+00:00" if expires_str.endswith("Z") else expires_str
                        expires_dt = datetime.datetime.fromisoformat(iso_ts)
                        if expires_dt < now:
                            continue
                    except Exception:
                        pass
                
                results.append({
                    "id": r[0],
                    "message": r[1],
                    "created_at": r[2],
                    "author_session_id": r[3],
                    "priority": r[4],
                    "expires_at": r[5],
                    "acknowledged": bool(r[6])
                })
            return results
        except Exception as e:
            logger.error(f"Failed to list bottles: {e}")
            raise SearchError(
                code=SQLITE_TRANSACTION_FAILED,
                message=f"Failed to list bottles: {e}",
                subsystem="BOTTLES",
                retry_safe=True
            )

    def acknowledge_bottle(self, bottle_id: str) -> bool:
        """Marks a bottle note as acknowledged."""
        try:
            with self.db.conn as conn:
                cursor = conn.cursor()
                cursor.execute("UPDATE bottles SET acknowledged = 1 WHERE id = ?", (bottle_id,))
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Failed to acknowledge bottle {bottle_id}: {e}")
            raise SearchError(
                code=SQLITE_TRANSACTION_FAILED,
                message=f"Failed to acknowledge bottle {bottle_id}: {e}",
                subsystem="BOTTLES",
                retry_safe=True
            )
