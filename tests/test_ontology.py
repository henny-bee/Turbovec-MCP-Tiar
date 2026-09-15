"""The ontology must cover everything the engine itself emits."""

import pytest
from unittest.mock import patch

import numpy as np

from extraction.patterns import RELATION_PATTERNS
from memory.errors import SearchError
from memory.ontology import DEFAULT_ONTOLOGY, OntologyManager
from search.radar import _TYPE_RELATIONS
from vector_db import VectorDB


def test_every_extracted_relation_type_is_registered():
    """Regression: an unregistered type aborts the whole extraction pipeline."""
    registered = {name.lower() for name in DEFAULT_ONTOLOGY["relations"]}
    emitted = {relation_type.lower() for relation_type, _ in RELATION_PATTERNS}
    assert emitted <= registered, sorted(emitted - registered)


def test_every_inferred_radar_relation_type_is_registered():
    registered = {name.lower() for name in DEFAULT_ONTOLOGY["relations"]}
    emitted = {relation.lower() for _, relation in _TYPE_RELATIONS}
    emitted |= {"related_to", "bridges_to", "analogous_to", "enables"}
    assert emitted <= registered, sorted(emitted - registered)


def test_shipped_ontology_file_matches_defaults(tmp_path):
    """The repository's ontology.json must not lag behind DEFAULT_ONTOLOGY."""
    shipped = OntologyManager("ontology.json")
    assert set(DEFAULT_ONTOLOGY["relations"]) <= shipped.relations
    assert set(DEFAULT_ONTOLOGY["entities"]) <= shipped.entities


def test_relation_validation_is_case_insensitive(tmp_path):
    manager = OntologyManager(str(tmp_path / "ontology.json"))
    manager.validate_relation_type("depends_on")
    manager.validate_relation_type("DEPENDS_ON")
    with pytest.raises(SearchError, match="not valid"):
        manager.validate_relation_type("NOT_A_REAL_TYPE")


def test_ingesting_locative_text_keeps_edges_and_observations(tmp_path, monkeypatch):
    """ "X lives in Y" used to raise mid-pipeline and drop later observations."""
    monkeypatch.chdir(tmp_path)

    def encode(text, **kwargs):
        if isinstance(text, (list, tuple)):
            return np.zeros((len(text), 384), dtype=np.float32)
        return np.zeros(384, dtype=np.float32)

    with patch("embeddings.minilm.SentenceTransformer") as MockST:
        MockST.return_value.encode.side_effect = encode
        db = VectorDB(
            dimension=384,
            metadata_file=str(tmp_path / "metadata.json"),
            index_file=str(tmp_path / "index.tvim"),
            sqlite_db_file=str(tmp_path / "memory.db"),
        )
        db.add_knowledge(
            "Doc", "Charlie lives in Berlin. Charlie is a senior engineer."
        )

        edges = db.conn.execute("SELECT relationship_type FROM edges").fetchall()
        observations = db.conn.execute("SELECT COUNT(*) FROM observations").fetchone()[
            0
        ]

    assert [row[0] for row in edges] == ["LOCATED_IN"]
    assert observations >= 1
