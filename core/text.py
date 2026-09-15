"""Pure text helpers shared by ingestion, indexing and retrieval."""

from __future__ import annotations

import json
from typing import Any, List, Sequence

__all__ = [
    "chunk_text",
    "build_embedding_text",
    "extract_description",
    "sanitize_fts_query",
]


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Splits ``text`` into overlapping windows of ``chunk_size`` characters."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")
    if not text:
        return []

    step = chunk_size - overlap
    chunks: List[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= length:
            break
        start += step
    return chunks


def build_embedding_text(
    name: str,
    node_type: str,
    description: str,
    observations: Sequence[str],
    max_observations: int = 20,
    max_chars_per_observation: int = 500,
) -> str:
    """The single source of truth for what an entity's embedding sees."""
    parts = [name, node_type, description]
    parts.extend(
        obs[:max_chars_per_observation] for obs in observations[:max_observations]
    )
    return " ".join(part for part in parts if part)


def extract_description(properties: Any) -> str:
    """Pulls a human-readable description out of a node's properties blob."""
    if not properties:
        return ""
    if isinstance(properties, str):
        try:
            properties = json.loads(properties)
        except Exception:
            return properties
    if isinstance(properties, dict):
        return properties.get("description", json.dumps(properties))
    return str(properties)


def sanitize_fts_query(query: str) -> str:
    """Rewrites a free-form query into a safe FTS5 MATCH expression.

    Every term is quoted, which neutralises FTS5 operators (NEAR, OR, column
    filters, prefix stars) that would otherwise raise or let a user craft an
    unintended query. Terms remain implicitly ANDed, as before.
    """
    cleaned = query
    for char in ("*", "(", ")", ":", "^"):
        cleaned = cleaned.replace(char, " ")
    terms = [term for term in cleaned.split() if term]
    if not terms:
        return ""
    return " ".join('"{}"'.format(term.replace('"', '""')) for term in terms)
