"""Observability and maintenance MCP tools."""

from __future__ import annotations

import logging

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

__all__ = ["register"]


def register(mcp: FastMCP, db) -> None:
    @mcp.tool()
    def search_stats() -> dict:
        """Calculates rolling statistics, latencies per subsystem, percentiles, error rates, and cache performance."""
        logger.info("Tool 'search_stats' called")
        return db.search_stats()

    @mcp.tool()
    def analyze_graph() -> dict:
        """Runs connected components, LPA community detection, degree centrality, and PageRank analytics over the graph topology."""
        logger.info("Tool 'analyze_graph' called")
        return db.analyze_graph()

    @mcp.tool()
    def run_librarian_cycle() -> dict:
        """Executes one background librarian cycle to organize, cluster, detect duplicates, and synthesize concepts autonomously."""
        logger.info("Tool 'run_librarian_cycle' called")
        return db.run_librarian_cycle()
