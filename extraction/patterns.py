"""Pre-compiled regexes for the rule-based extraction fallback.

Compiling these once at import time (instead of per call, as before) makes
repeated ingestion measurably cheaper and keeps the rules in one readable
place where new relation verbs can be added without touching the extractor.
"""

from __future__ import annotations

import re
from typing import Tuple

__all__ = [
    "STOP_WORDS",
    "NAMED_ENTITY_RE",
    "SENTENCE_SPLIT_RE",
    "DEFINITION_RE",
    "PREFERENCE_RE",
    "ARTICLE_PREFIX_RE",
    "RELATION_PATTERNS",
]

# Capitalised words that are never useful as standalone entities.
STOP_WORDS = frozenset(
    {
        "I",
        "The",
        "A",
        "An",
        "They",
        "He",
        "She",
        "It",
        "We",
        "You",
        "But",
        "And",
        "Or",
        "If",
        "Then",
        "Else",
        "When",
        "Where",
        "Why",
        "How",
        "This",
        "That",
        "These",
        "Those",
        "My",
        "Your",
        "His",
        "Her",
        "Its",
        "Our",
        "Their",
    }
)

# A proper-noun phrase: one or more capitalised tokens.
_NAME = r"[A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)*"

NAMED_ENTITY_RE = re.compile(r"\b" + _NAME + r"\b")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
ARTICLE_PREFIX_RE = re.compile(r"^(?:a|an|the)\s+", re.IGNORECASE)

DEFINITION_RE = re.compile(
    r"\b(?P<entity>" + _NAME + r")\s+is\s+(?:a|an)?\s*(?P<definition>[^.!?]+)"
)
PREFERENCE_RE = re.compile(
    r"\b(?:I|User)\s+(?:prefer|like|enjoy|love)s?\s+(?P<preference>[^.!?]+)",
    re.IGNORECASE,
)


def _relation(verb_pattern: str) -> "re.Pattern[str]":
    return re.compile(
        r"\b(?P<from_node>"
        + _NAME
        + r")\s+"
        + verb_pattern
        + r"\s+(?P<to_node>"
        + _NAME
        + r")\b",
        re.IGNORECASE,
    )


# (relationship_type, compiled pattern) evaluated in order, first match wins
# per pattern. Ordering matters: more specific phrasings come first.
RELATION_PATTERNS: Tuple[Tuple[str, "re.Pattern[str]"], ...] = (
    ("WORKS_AT", _relation(r"works?\s+at")),
    ("PART_OF", _relation(r"is\s+part\s+of")),
    ("DEPENDS_ON", _relation(r"depends?\s+on")),
    ("MEMBER_OF", _relation(r"(?:is\s+)?members?\s+of")),
    ("LOCATED_IN", _relation(r"lives?\s+in")),
    ("CREATED_BY", _relation(r"created")),
    ("OWNS", _relation(r"owns?")),
    ("SUPPORTS", _relation(r"supports?")),
    ("PARTNER_WITH", _relation(r"is\s+partners?\s+of")),
    ("CONNECTS_TO", _relation(r"is\s+connected\s+to")),
)
