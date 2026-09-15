"""Entity, relationship and observation extraction from unstructured text.

Uses spaCy's ``en_core_web_sm`` when it is installed and falls back to the
deterministic rule engine in :mod:`extraction.patterns` otherwise. The spaCy
pipeline is resolved once per process rather than per call.
"""

from __future__ import annotations

import logging
import threading
from typing import Dict, List

from extraction.patterns import (
    ARTICLE_PREFIX_RE,
    DEFINITION_RE,
    NAMED_ENTITY_RE,
    PREFERENCE_RE,
    RELATION_PATTERNS,
    SENTENCE_SPLIT_RE,
    STOP_WORDS,
)

logger = logging.getLogger(__name__)

__all__ = ["ExtractionMixin", "load_spacy_pipeline"]

_SPACY_LOCK = threading.Lock()
_SPACY_PIPELINE = None
_SPACY_RESOLVED = False


def load_spacy_pipeline(model_name: str = "en_core_web_sm"):
    """Returns a cached spaCy pipeline, or ``None`` when unavailable."""
    global _SPACY_PIPELINE, _SPACY_RESOLVED
    if _SPACY_RESOLVED:
        return _SPACY_PIPELINE
    with _SPACY_LOCK:
        if not _SPACY_RESOLVED:
            try:
                import spacy

                _SPACY_PIPELINE = spacy.load(model_name)
            except Exception as exc:
                logger.info(
                    f"spaCy not available or failed to load: {exc}. Using regex fallback."
                )
                _SPACY_PIPELINE = None
            _SPACY_RESOLVED = True
    return _SPACY_PIPELINE


def _split_sentences(text: str) -> List[str]:
    sentences: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        sentences.extend(
            part for part in map(str.strip, SENTENCE_SPLIT_RE.split(line)) if part
        )
    return sentences


class ExtractionMixin:
    """Text-to-graph extraction half of :class:`~core.database.VectorDB`."""

    def extract_entities_and_relations(self, text: str) -> dict:
        """Extracts entities, relationships and observations from ``text``."""
        entities: Dict[str, dict] = {}
        relationships: List[dict] = []
        observations: List[dict] = []

        self._extract_named_entities(text, entities)
        for sentence in _split_sentences(text):
            self._extract_definition(sentence, entities, observations)
            self._extract_preference(sentence, entities, observations)
            self._extract_relations(sentence, entities, relationships)

        return {
            "entities": list(entities.values()),
            "relationships": relationships,
            "observations": observations,
        }

    # -- stages ------------------------------------------------------------
    def _extract_named_entities(self, text: str, entities: Dict[str, dict]) -> None:
        pipeline = load_spacy_pipeline()
        if pipeline is not None:
            try:
                for entity in pipeline(text).ents:
                    name = entity.text.strip()
                    if len(name) < 2 or name in STOP_WORDS or name in entities:
                        continue
                    entities[name] = {
                        "name": name,
                        "type": entity.label_,
                        "properties": {"description": f"Extracted as {entity.label_}"},
                    }
                return
            except Exception as exc:
                logger.error(f"Error during spaCy extraction: {exc}")

        for match in NAMED_ENTITY_RE.finditer(text):
            name = match.group(0).strip()
            if len(name) < 2 or name in STOP_WORDS or name in entities:
                continue
            entities[name] = {
                "name": name,
                "type": "entity",
                "properties": {"description": "Extracted named entity"},
            }

    def _extract_definition(
        self, sentence: str, entities: Dict[str, dict], observations: List[dict]
    ) -> None:
        """Handles "X is (a|an) Y" into a description plus an observation."""
        match = DEFINITION_RE.search(sentence)
        if not match:
            return
        name = match.group("entity").strip()
        definition = match.group("definition").strip()
        if name in STOP_WORDS or len(name) < 2:
            return

        if name in entities:
            entities[name]["properties"]["description"] = definition
        else:
            entities[name] = {
                "name": name,
                "type": "entity",
                "properties": {"description": definition},
            }
        observations.append({"entity_name": name, "content": f"{name} is {definition}"})

    def _extract_preference(
        self, sentence: str, entities: Dict[str, dict], observations: List[dict]
    ) -> None:
        """Handles "I prefer X" / "User likes X" into a PREFERENCE entity."""
        match = PREFERENCE_RE.search(sentence)
        if not match:
            return
        item = ARTICLE_PREFIX_RE.sub("", match.group("preference").strip())
        if not item or len(item) < 2:
            return

        entity = entities.setdefault(item, {"name": item, "properties": {}})
        entity["type"] = "PREFERENCE"
        entity["properties"]["description"] = "Preferred by user"

        observations.append({"entity_name": "User", "content": f"Prefers {item}"})
        entities.setdefault(
            "User",
            {
                "name": "User",
                "type": "person",
                "properties": {"description": "The active user"},
            },
        )

    def _extract_relations(
        self, sentence: str, entities: Dict[str, dict], relationships: List[dict]
    ) -> None:
        """Handles the verb patterns in ``RELATION_PATTERNS``."""
        for relation_type, pattern in RELATION_PATTERNS:
            match = pattern.search(sentence)
            if not match:
                continue
            source = match.group("from_node").strip()
            target = match.group("to_node").strip()
            if source in STOP_WORDS or target in STOP_WORDS:
                continue

            for name in (source, target):
                entities.setdefault(
                    name,
                    {
                        "name": name,
                        "type": "entity",
                        "properties": {"description": "Extracted named entity"},
                    },
                )
            relationships.append(
                {
                    "from": source,
                    "to": target,
                    "type": relation_type,
                    "properties": {},
                }
            )
