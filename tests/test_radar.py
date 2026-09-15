"""Semantic radar behaviour: similarity gating, reachability and auto-creation."""

import numpy as np
import pytest
from unittest.mock import patch

from vector_db import VectorDB


@pytest.fixture
def temp_files(tmp_path):
    return (
        str(tmp_path / "metadata.json"),
        str(tmp_path / "index.tvim"),
        str(tmp_path / "memory.db"),
    )


@pytest.fixture
def db(temp_files, make_encoder):
    """Three topics: two nearly identical, one orthogonal."""

    def encode_one(text):
        vector = np.zeros(384, dtype=np.float32)
        lowered = text.lower()
        if "alpha" in lowered:
            vector[0] = 1.0
        elif "alpaca" in lowered:  # near-duplicate of alpha
            vector[0] = 0.99
            vector[1] = 0.14
        else:
            vector[5] = 1.0
        return vector

    metadata_file, index_file, sqlite_db_file = temp_files
    with patch("embeddings.minilm.SentenceTransformer") as MockST:
        MockST.return_value.encode.side_effect = make_encoder(encode_one)
        yield VectorDB(
            dimension=384,
            metadata_file=metadata_file,
            index_file=index_file,
            sqlite_db_file=sqlite_db_file,
        )


def _seed(db):
    db.create_node("alpha", "Alpha", "concept", properties={"description": "alpha"})
    db.create_node("alpaca", "Alpaca", "concept", properties={"description": "alpaca"})
    db.create_node("other", "Other", "concept", properties={"description": "unrelated"})


def test_radar_reports_similar_but_unlinked_pairs(db):
    _seed(db)

    suggestions = db.semantic_radar(similarity_threshold=0.9)

    pairs = {
        frozenset((item["node_1"]["id"], item["node_2"]["id"])) for item in suggestions
    }
    assert pairs == {frozenset(("alpha", "alpaca"))}
    assert suggestions[0]["similarity"] >= 0.9
    assert suggestions[0]["created_edge"] is False


def test_radar_ignores_pairs_that_already_have_a_path(db):
    _seed(db)
    db.create_edge("e1", "alpha", "alpaca", "related_to")

    assert db.semantic_radar(similarity_threshold=0.9) == []


def test_radar_ignores_indirect_paths_too(db):
    _seed(db)
    # alpha -> other -> alpaca means the pair is already reachable.
    db.create_edge("e1", "alpha", "other", "related_to")
    db.create_edge("e2", "other", "alpaca", "related_to")

    assert db.semantic_radar(similarity_threshold=0.9) == []


def test_radar_auto_create_writes_one_edge_and_is_idempotent(db):
    _seed(db)

    first = db.semantic_radar(similarity_threshold=0.9, auto_create=True)
    assert [item["created_edge"] for item in first] == [True]
    assert first[0]["edge_details"]["relationship_type"] == "RELATED_TO"

    edges = db.conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    assert edges == 1

    # The pair is now connected, so a second pass proposes nothing.
    assert db.semantic_radar(similarity_threshold=0.9, auto_create=True) == []
    assert db.conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0] == 1


def test_radar_sorts_by_similarity_descending(db):
    _seed(db)
    suggestions = db.semantic_radar(similarity_threshold=-1.0)

    similarities = [item["similarity"] for item in suggestions]
    assert similarities == sorted(similarities, reverse=True)
    assert len(suggestions) == 3  # every pair of three nodes


def test_radar_needs_at_least_two_nodes(db):
    assert db.semantic_radar() == []
    db.create_node("alpha", "Alpha", "concept")
    assert db.semantic_radar() == []


def test_radar_respects_node_budget(db):
    _seed(db)
    db.settings = db.settings.__class__.from_env(radar_max_nodes=1)

    assert db.semantic_radar(similarity_threshold=-1.0) == []
