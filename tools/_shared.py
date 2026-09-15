"""Helpers shared by every MCP tool module."""

from __future__ import annotations

import logging
from typing import Any

from pydantic.fields import FieldInfo

__all__ = ["resolve_field", "failure", "error_dict", "error_list"]


def resolve_field(value: Any, default: Any = None) -> Any:
    """Falls back to ``default`` when a Field() default leaks through as a value.

    FastMCP passes ``FieldInfo`` objects through when a parameter is omitted by
    a client that does not fill defaults; this normalises that back.
    """
    return default if isinstance(value, FieldInfo) else value


def failure(logger: logging.Logger, message: str, exc: Exception) -> str:
    """Logs a tool failure and renders the text form MCP clients expect."""
    logger.error(f"{message}: {exc}", exc_info=True)
    return f"Error: {exc}"


def error_dict(logger: logging.Logger, message: str, exc: Exception) -> dict:
    logger.error(f"{message}: {exc}", exc_info=True)
    return {"error": str(exc)}


def error_list(logger: logging.Logger, message: str, exc: Exception) -> list:
    logger.error(f"{message}: {exc}", exc_info=True)
    return [{"error": str(exc)}]
