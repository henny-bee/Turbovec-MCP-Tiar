"""Session lifecycle: chaining work sessions and recording breakthroughs."""

from __future__ import annotations

import json
import logging
import uuid

from graph.repository import utc_now

logger = logging.getLogger(__name__)

__all__ = ["SessionMixin"]


class SessionMixin:
    """Session half of :class:`~core.database.VectorDB`."""

    def start_session(self, session_id: str, name: str, properties: dict = None) -> str:
        """Opens a session node, linked back to the previous one."""
        session_properties = dict(properties) if properties else {}
        session_properties.setdefault("status", "active")

        previous = self.conn.execute(
            "SELECT id FROM nodes WHERE node_type = 'session' "
            "ORDER BY created_at DESC LIMIT 1"
        ).fetchone()

        self.create_node(
            node_id=session_id,
            name=name,
            node_type="session",
            properties=session_properties,
        )

        if previous:
            self.create_edge(
                edge_id=f"edge-{session_id}-{previous[0]}",
                from_node_id=session_id,
                to_node_id=previous[0],
                relationship_type="PRECEDED_BY",
            )
        return session_id

    def end_session(self, session_id: str, summary: str) -> None:
        """Closes a session and stores its summary."""
        row = self.conn.execute(
            "SELECT properties FROM nodes WHERE id = ?", (session_id,)
        ).fetchone()
        if not row:
            raise ValueError(f"Session {session_id} not found")

        properties = json.loads(row[0]) if row[0] else {}
        properties.update(
            {"status": "closed", "summary": summary, "ended_at": utc_now()}
        )
        self.update_node(node_id=session_id, properties=properties)

    def record_breakthrough(self, session_id: str, title: str, content: str) -> str:
        """Records a learning breakthrough linked to a session."""
        breakthrough_id = str(uuid.uuid4())
        self.create_node(
            node_id=breakthrough_id,
            name=title,
            node_type="breakthrough",
            properties={"content": content},
        )
        self.create_edge(
            edge_id=f"edge-{breakthrough_id}-{session_id}",
            from_node_id=breakthrough_id,
            to_node_id=session_id,
            relationship_type="BREAKTHROUGH_IN",
        )
        return breakthrough_id
