import os
import json
import logging
from typing import Set, Dict, Any, List
from memory.errors import SearchError, ONTOLOGY_VALIDATION_FAILED

logger = logging.getLogger(__name__)

DEFAULT_ONTOLOGY = {
    "version": 1,
    "entities": [
        "person",
        "project",
        "technology",
        "decision",
        "event",
        "concept",
        "file",
        "entity",
        "session",
        "breakthrough",
        "document",
        "preference"
    ],
    "relations": [
        "uses",
        "depends_on",
        "created_by",
        "replaced",
        "related_to",
        "caused_by",
        "implements",
        "contradicts",
        "PRECEDED_BY",
        "BREAKTHROUGH_IN",
        "BRIDGES_TO",
        "ANALOGOUS_TO",
        "ENABLES",
        "DECIDED_IN",
        "MENTIONED_IN",
        "SUPPORTS",
        "CONNECTS_TO",
        "KNOWS",
        "WORKS_AT"
    ]
}

class OntologyManager:
    def __init__(self, filepath: str = "ontology.json"):
        self.filepath = filepath
        self.entities: Set[str] = set()
        self.relations: Set[str] = set()
        self.version = 1
        self.load_ontology()

    def load_ontology(self) -> None:
        """Loads the ontology configuration from disk, creating a default one if missing."""
        if not os.path.exists(self.filepath):
            logger.info(f"Ontology file {self.filepath} not found. Creating default ontology.")
            self.entities = set(DEFAULT_ONTOLOGY["entities"])
            self.relations = set(DEFAULT_ONTOLOGY["relations"])
            self.version = DEFAULT_ONTOLOGY["version"]
            self.save_ontology()
            return

        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.version = data.get("version", 1)
                self.entities = set(data.get("entities", DEFAULT_ONTOLOGY["entities"]))
                self.relations = set(data.get("relations", DEFAULT_ONTOLOGY["relations"]))
                logger.info(f"Loaded ontology version {self.version} with {len(self.entities)} entity types and {len(self.relations)} relation types.")
        except Exception as e:
            logger.error(f"Failed to load ontology from {self.filepath}: {e}. Falling back to default.")
            self.entities = set(DEFAULT_ONTOLOGY["entities"])
            self.relations = set(DEFAULT_ONTOLOGY["relations"])
            self.version = DEFAULT_ONTOLOGY["version"]

    def save_ontology(self) -> None:
        """Saves the current ontology configuration to disk."""
        try:
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump({
                    "version": self.version,
                    "entities": sorted(list(self.entities)),
                    "relations": sorted(list(self.relations))
                }, f, indent=2)
            logger.info(f"Ontology saved to {self.filepath}.")
        except Exception as e:
            logger.error(f"Failed to save ontology to {self.filepath}: {e}")

    def add_entity_type(self, entity_type: str) -> bool:
        """Adds a new entity type to the ontology."""
        normalized = entity_type.strip().lower()
        if not normalized:
            raise SearchError(
                code=ONTOLOGY_VALIDATION_FAILED,
                message="Entity type cannot be empty",
                subsystem="ONTOLOGY",
                retry_safe=False
            )
        if normalized not in self.entities:
            self.entities.add(normalized)
            self.save_ontology()
            return True
        return False

    def add_relation_type(self, relation_type: str) -> bool:
        """Adds a new relationship type to the ontology."""
        normalized = relation_type.strip()  # Preserve case for relation types like BREAKTHROUGH_IN
        if not normalized:
            raise SearchError(
                code=ONTOLOGY_VALIDATION_FAILED,
                message="Relation type cannot be empty",
                subsystem="ONTOLOGY",
                retry_safe=False
            )
        if normalized not in self.relations:
            self.relations.add(normalized)
            self.save_ontology()
            return True
        return False

    def validate_entity_type(self, entity_type: str) -> None:
        """Validates if an entity type is registered in the ontology."""
        normalized = entity_type.strip().lower()
        if normalized not in self.entities:
            raise SearchError(
                code=ONTOLOGY_VALIDATION_FAILED,
                message=f"Entity type '{entity_type}' is not valid. Registered types: {sorted(list(self.entities))}",
                subsystem="ONTOLOGY",
                retry_safe=False
            )

    def validate_relation_type(self, relation_type: str) -> None:
        """Validates if a relationship type is registered in the ontology."""
        normalized = relation_type.strip()
        # Case insensitive/flexible check
        valid_lower = {r.lower() for r in self.relations}
        if normalized.lower() not in valid_lower:
            raise SearchError(
                code=ONTOLOGY_VALIDATION_FAILED,
                message=f"Relationship type '{relation_type}' is not valid. Registered types: {sorted(list(self.relations))}",
                subsystem="ONTOLOGY",
                retry_safe=False
            )

    def get_schema(self) -> Dict[str, Any]:
        """Returns the current ontology schema."""
        return {
            "version": self.version,
            "entities": sorted(list(self.entities)),
            "relations": sorted(list(self.relations))
        }
