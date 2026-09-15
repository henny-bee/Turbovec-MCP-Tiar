"""Backwards-compatible entry point for the memory engine.

The implementation now lives in :mod:`core.database` and its mixins; this
module keeps ``from vector_db import VectorDB`` working for existing clients,
editor configurations and integrations.
"""

from __future__ import annotations

from core.config import Settings
from core.database import VectorDB

__all__ = ["VectorDB", "Settings"]
