"""Load/save helpers for the chunk-level document metadata file."""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

__all__ = ["load_documents", "save_documents"]

Documents = Dict[int, dict]


def load_documents(path: str) -> Tuple[Documents, int]:
    """Reads ``path`` and returns ``(documents, next_id)``.

    Accepts both the current ``{"next_id": ..., "documents": {...}}`` layout and
    the legacy list-of-documents format.
    """
    if not os.path.exists(path):
        return {}, 0

    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception as exc:
        logger.error(f"Failed to load metadata from {path}: {exc}", exc_info=True)
        return {}, 0

    if isinstance(data, dict) and "documents" in data:
        documents = {int(key): value for key, value in data["documents"].items()}
        next_id = data.get("next_id", max(documents, default=-1) + 1)
    elif isinstance(data, list):
        documents = {doc["id"]: doc for doc in data if doc and not doc.get("deleted")}
        next_id = max(documents, default=-1) + 1
    else:
        return {}, 0

    logger.info(f"Loaded {len(documents)} documents from {path}")
    return documents, next_id


def save_documents(path: str, documents: Documents, next_id: int) -> None:
    """Writes the document store, logging (but not raising on) failures."""
    try:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"next_id": next_id, "documents": documents}, handle, indent=2)
        logger.info(f"Metadata saved to {path}")
    except Exception as exc:
        logger.error(f"Failed to save metadata to {path}: {exc}", exc_info=True)
