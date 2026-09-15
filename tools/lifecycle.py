"""Memory hygiene MCP tools: archive, restore, orphans and pruning."""

from __future__ import annotations

import logging

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from tools._shared import resolve_field

logger = logging.getLogger(__name__)

__all__ = ["register"]


def register(mcp: FastMCP, db) -> None:
    @mcp.tool()
    def archive_entity(
        id: str = Field(..., description="The ID of the entity node to archive"),
        reason: str = Field(
            "Unused/superseded", description="Reason for archiving the entity"
        ),
    ) -> dict:
        """Archives an entity and updates its status to ARCHIVED, excluding it from active retrieval."""
        logger.info(f"Tool 'archive_entity' called for ID: {id}")
        return db.archive_entity(id, reason=resolve_field(reason, "Unused/superseded"))

    @mcp.tool()
    def restore_entity(
        id: str = Field(
            ..., description="The ID of the entity node to restore to ACTIVE status"
        ),
    ) -> dict:
        """Restores an archived entity node back to ACTIVE status."""
        logger.info(f"Tool 'restore_entity' called for ID: {id}")
        return db.restore_entity(id)

    @mcp.tool()
    def list_orphans() -> list[dict]:
        """Identifies and lists orphaned entities (nodes with zero edges)."""
        logger.info("Tool 'list_orphans' called")
        return db.list_orphans()

    @mcp.tool()
    def prune_stale(
        max_age_days: int = Field(
            ...,
            description="Maximum age in days for STALE or ARCHIVED nodes to be pruned",
        ),
        dry_run: bool = Field(
            False,
            description="If True, only lists the candidates without deleting them",
        ),
    ) -> list[dict]:
        """Identifies stale or archived memories older than a specified duration and prunes them from the database."""
        logger.info(
            f"Tool 'prune_stale' called with max_age_days: {max_age_days}, "
            f"dry_run: {dry_run}"
        )
        return db.prune_stale(
            max_age_days=resolve_field(max_age_days),
            dry_run=resolve_field(dry_run, False),
        )
