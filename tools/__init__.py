"""MCP tool and prompt registry.

Tools are grouped by domain, one module each, and every module exposes a
``register(mcp, db)`` function. Adding a tool means touching exactly one module
and, if it is a new domain, one line of ``_MODULES``.
"""

from __future__ import annotations

import logging

from mcp.server.fastmcp import FastMCP

from tools import (
    admin,
    bottles,
    graph,
    knowledge,
    lifecycle,
    prompts,
    search,
    sessions,
    temporal,
)
from tools._shared import resolve_field

logger = logging.getLogger(__name__)

__all__ = ["register_tools_and_prompts", "resolve_field"]

_MODULES = (
    knowledge,
    graph,
    search,
    sessions,
    temporal,
    lifecycle,
    bottles,
    admin,
    prompts,
)


def register_tools_and_prompts(mcp: FastMCP, db) -> None:
    """Registers every MCP tool and prompt on ``mcp``, bound to ``db``."""
    logger.info("Registering MCP tools and prompts")
    for module in _MODULES:
        module.register(mcp, db)
    logger.info("Successfully registered all MCP tools and prompts")
