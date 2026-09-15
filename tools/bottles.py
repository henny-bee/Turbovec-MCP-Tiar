"""Bottle-note MCP tools: messages left for future sessions."""

from __future__ import annotations

import logging

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from tools._shared import resolve_field

logger = logging.getLogger(__name__)

__all__ = ["register"]


def register(mcp: FastMCP, db) -> None:
    @mcp.tool()
    def create_bottle(
        message: str = Field(
            ..., description="The content of the note/message left in the bottle"
        ),
        priority: str = Field(
            "medium", description="Priority tier: low, medium, or high"
        ),
        expires_at: str = Field(
            None, description="Optional ISO 8601 expiration timestamp"
        ),
        author_session_id: str = Field(
            None, description="Optional session ID of the authoring agent"
        ),
    ) -> dict:
        """Creates a standardized bottle note for subsequent sessions or agents to discover."""
        logger.info("Tool 'create_bottle' called")
        return db.create_bottle(
            message=resolve_field(message),
            priority=resolve_field(priority, "medium"),
            expires_at=resolve_field(expires_at),
            author_session_id=resolve_field(author_session_id),
        )

    @mcp.tool()
    def get_bottles(
        include_acknowledged: bool = Field(
            False, description="If True, includes acknowledged bottle notes in list"
        ),
    ) -> list[dict]:
        """Lists active and optionally acknowledged bottle notes left by agents."""
        logger.info("Tool 'get_bottles' called")
        return db.get_bottles(
            include_acknowledged=resolve_field(include_acknowledged, False)
        )

    @mcp.tool()
    def acknowledge_bottle(
        id: str = Field(..., description="The ID of the bottle note to acknowledge"),
    ) -> bool:
        """Marks a bottle note as acknowledged/read."""
        logger.info(f"Tool 'acknowledge_bottle' called for ID: {id}")
        return db.acknowledge_bottle(id)
