"""Hybrid retrieval, semantic radar and background-discovery MCP tools."""

from __future__ import annotations

import logging

from mcp.server.fastmcp import FastMCP

from tools._shared import error_list, failure

logger = logging.getLogger(__name__)

__all__ = ["register"]


def register(mcp: FastMCP, db) -> None:
    @mcp.tool()
    def search_memory(
        query: str,
        limit: int = 10,
        project_id: str = "default",
        channel_weights: dict = None,
    ) -> list[dict]:
        """Runs a hybrid vector-lexical search over the memory graph and returns ranked nodes."""
        logger.info(f"Tool 'search_memory' called with query: '{query}'")
        try:
            return db.search_hybrid(
                query=query,
                project_id=project_id,
                limit=limit,
                channel_weights=channel_weights,
            )
        except Exception as exc:
            return error_list(logger, f"Error searching memory for '{query}'", exc)

    @mcp.tool()
    def semantic_radar(
        similarity_threshold: float = 0.7,
        project_id: str = "default",
        auto_create: bool = False,
    ) -> list[dict]:
        """Scans the graph to detect pairs of entities with high semantic similarity but no path between them, with optional automatic creation of inferred edges."""
        logger.info(
            f"Tool 'semantic_radar' called with threshold: {similarity_threshold}, "
            f"auto_create: {auto_create}"
        )
        try:
            return db.semantic_radar(
                similarity_threshold=similarity_threshold,
                project_id=project_id,
                auto_create=auto_create,
            )
        except Exception as exc:
            return error_list(logger, "Error in semantic_radar", exc)

    @mcp.tool()
    def run_relationship_discovery(
        similarity_threshold: float = 0.7,
        project_id: str = "default",
        auto_create: bool = True,
    ) -> list[dict]:
        """Runs the semantic radar discovery process to find disconnected highly similar nodes and optionally create inferred relationship edges."""
        logger.info(
            f"Tool 'run_relationship_discovery' called with similarity_threshold: "
            f"{similarity_threshold}, auto_create: {auto_create}"
        )
        try:
            return db.semantic_radar(
                similarity_threshold=similarity_threshold,
                project_id=project_id,
                auto_create=auto_create,
            )
        except Exception as exc:
            return error_list(logger, "Error in run_relationship_discovery", exc)

    @mcp.tool()
    def get_background_discovery_status() -> dict:
        """Returns the current status, interval, and similarity threshold of the background relationship discovery worker."""
        logger.info("Tool 'get_background_discovery_status' called")
        return {
            "background_discovery_enabled_at_startup": db.bg_enabled,
            "background_worker_running": (
                db.bg_thread is not None and db.bg_thread.is_alive()
            ),
            "interval_seconds": db.bg_interval,
            "similarity_threshold": db.bg_similarity_threshold,
        }

    @mcp.tool()
    def set_background_discovery(
        enabled: bool,
        interval_seconds: int = 300,
        similarity_threshold: float = 0.7,
    ) -> str:
        """Dynamically starts, stops, or configures the background relationship discovery daemon worker."""
        logger.info(
            f"Tool 'set_background_discovery' called with enabled: {enabled}, "
            f"interval_seconds: {interval_seconds}, "
            f"similarity_threshold: {similarity_threshold}"
        )
        try:
            db.bg_interval = interval_seconds
            db.bg_similarity_threshold = similarity_threshold
            if not enabled:
                db.stop_background_discovery()
                return "Successfully stopped background relationship discovery."

            db.start_background_discovery()
            return (
                f"Successfully started/updated background relationship discovery with "
                f"interval of {interval_seconds}s and similarity threshold "
                f"{similarity_threshold}."
            )
        except Exception as exc:
            return failure(logger, "Error in set_background_discovery", exc)
