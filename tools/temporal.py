"""Time-travel MCP tools: snapshots, diffs, timelines and temporal neighbours."""

from __future__ import annotations

import logging

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from tools._shared import error_dict, resolve_field

logger = logging.getLogger(__name__)

__all__ = ["register"]


def register(mcp: FastMCP, db) -> None:
    @mcp.tool()
    def point_in_time_query(
        as_of_iso_timestamp: str,
    ) -> dict:
        """Queries the state of the memory graph as of a specific date and time."""
        logger.info(f"Tool 'point_in_time_query' called for: {as_of_iso_timestamp}")
        try:
            return db.get_graph_state_as_of(as_of_iso_timestamp=as_of_iso_timestamp)
        except Exception as exc:
            return error_dict(logger, "Error in point_in_time_query", exc)

    @mcp.tool()
    def diff_knowledge_state(
        from_timestamp: str = Field(
            ..., description="Starting ISO 8601 timestamp for the diff window"
        ),
        to_timestamp: str = Field(
            ..., description="Ending ISO 8601 timestamp for the diff window"
        ),
    ) -> dict:
        """Computes a deterministic delta between two points in time (added, removed, changed nodes/edges)."""
        logger.info(
            f"Tool 'diff_knowledge_state' called from {from_timestamp} to {to_timestamp}"
        )
        return db.diff_knowledge_state(
            from_timestamp=resolve_field(from_timestamp),
            to_timestamp=resolve_field(to_timestamp),
        )

    @mcp.tool()
    def query_timeline(
        entity: str = Field(
            None, description="Optional entity name/attribute keyword filter"
        ),
        entity_type: str = Field(None, description="Optional node type filter"),
        start: str = Field(
            None, description="Optional starting ISO 8601 timestamp filter"
        ),
        end: str = Field(None, description="Optional ending ISO 8601 timestamp filter"),
        order: str = Field(
            "ASC", description="Chronological sorting order: ASC or DESC"
        ),
        limit: int = Field(
            50, description="Maximum number of timeline events to return"
        ),
        offset: int = Field(0, description="Offset for pagination"),
    ) -> list[dict]:
        """Queries the timeline of events chronologically with strict filtering and pagination."""
        logger.info("Tool 'query_timeline' called")
        return db.query_timeline(
            entity=resolve_field(entity),
            entity_type=resolve_field(entity_type),
            start=resolve_field(start),
            end=resolve_field(end),
            order=resolve_field(order, "ASC"),
            limit=resolve_field(limit, 50),
            offset=resolve_field(offset, 0),
        )

    @mcp.tool()
    def get_temporal_neighbors(
        node_id: str = Field(
            ..., description="The anchor entity ID to start traversal from"
        ),
        direction: str = Field(
            "both",
            description="Temporal relationship direction: before, after, or both",
        ),
        depth: int = Field(1, description="Graph traversal depth to explore"),
    ) -> list[dict]:
        """Explores connected nodes along temporal edges or chronologically related nodes relative to an anchor node."""
        logger.info(f"Tool 'get_temporal_neighbors' called for: {node_id}")
        return db.get_temporal_neighbors(
            node_id=resolve_field(node_id),
            direction=resolve_field(direction, "both"),
            depth=resolve_field(depth, 1),
        )
