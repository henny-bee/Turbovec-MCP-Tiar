"""Deterministic mapping between graph node ids and 64-bit vector ids."""

from __future__ import annotations

import hashlib
import sqlite3

UINT64_MASK = 0xFFFFFFFFFFFFFFFF
INT64_OFFSET = 0x10000000000000000
INT64_SIGN_BIT = 0x8000000000000000

__all__ = ["to_signed_64", "to_unsigned_64", "generate_vector_id"]


def to_signed_64(value: int) -> int:
    """Converts an unsigned 64-bit integer into SQLite's signed integer range."""
    value &= UINT64_MASK
    return value - INT64_OFFSET if value >= INT64_SIGN_BIT else value


def to_unsigned_64(value: int) -> int:
    """Converts a signed 64-bit integer read from SQLite back to unsigned."""
    return value & UINT64_MASK


def generate_vector_id(node_id: str, conn: sqlite3.Connection) -> int:
    """Derives a stable uint64 vector id from ``node_id``, resolving collisions.

    Numeric node ids are used verbatim for backwards compatibility; everything
    else takes the first 8 bytes of its SHA-256 digest. On collision the
    candidate is incremented until a free slot is found, so the mapping stays
    injective.
    """
    try:
        candidate = int(node_id)
        if not (0 <= candidate <= UINT64_MASK):
            raise ValueError
    except ValueError:
        digest = hashlib.sha256(node_id.encode("utf-8")).digest()
        candidate = int.from_bytes(digest[:8], byteorder="big") & UINT64_MASK

    cursor = conn.cursor()
    while True:
        cursor.execute(
            "SELECT node_id FROM vector_id_mapping WHERE vector_id = ?",
            (to_signed_64(candidate),),
        )
        row = cursor.fetchone()
        if row is None or row[0] == node_id:
            return candidate
        candidate = (candidate + 1) & UINT64_MASK
