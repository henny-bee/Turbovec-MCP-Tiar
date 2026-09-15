"""Session lifecycle MCP tools."""

from __future__ import annotations

import logging

from mcp.server.fastmcp import FastMCP

from tools._shared import failure

logger = logging.getLogger(__name__)

__all__ = ["register"]


def register(mcp: FastMCP, db) -> None:
    @mcp.tool()
    def start_session(
        session_id: str,
        name: str,
        properties: dict = None,
        project_id: str = "default",
    ) -> str:
        """Starts a session, creating a session node and linking to any previous session."""
        logger.info(f"Tool 'start_session' called: {session_id} - {name}")
        try:
            return db.start_session(
                session_id=session_id, name=name, properties=properties
            )
        except Exception as exc:
            return failure(logger, f"Error starting session '{session_id}'", exc)

    @mcp.tool()
    def end_session(
        session_id: str,
        summary: str,
        project_id: str = "default",
    ) -> str:
        """Ends an active session, saving its summary and updating status to closed."""
        logger.info(f"Tool 'end_session' called for: {session_id}")
        try:
            db.end_session(session_id=session_id, summary=summary)
            return f"Successfully ended session '{session_id}'."
        except Exception as exc:
            return failure(logger, f"Error ending session '{session_id}'", exc)

    @mcp.tool()
    def record_breakthrough(
        session_id: str,
        title: str,
        content: str,
        project_id: str = "default",
    ) -> str:
        """Records a learning breakthrough and links it to the active session."""
        logger.info(f"Tool 'record_breakthrough' called for Session: {session_id}")
        try:
            breakthrough_id = db.record_breakthrough(
                session_id=session_id, title=title, content=content
            )
            return (
                f"Successfully recorded breakthrough with ID '{breakthrough_id}' "
                f"linked to session '{session_id}'."
            )
        except Exception as exc:
            return failure(logger, "Error recording breakthrough", exc)
