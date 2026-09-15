"""Integration coverage for the Librarian cycle."""

import pytest
from unittest.mock import patch

import numpy as np

from vector_db import VectorDB


@pytest.fixture
def temp_files(tmp_path):
    return (
        str(tmp_path / "metadata.json"),
        str(tmp_path / "index.tvim"),
        str(tmp_path / "memory.db"),
    )


@pytest.fixture
def clustered_encoder(make_encoder):
    """Two tight semantic groups so DBSCAN has something real to find."""

    def encode_one(text):
        vector = np.zeros(384, dtype=np.float32)
        lowered = text.lower()
        if "storage" in lowered:
            vector[0] = 1.0
            vector[1] = 0.01
        elif "network" in lowered:
            vector[2] = 1.0
            vector[3] = 0.01
        else:
            np.random.seed(abs(hash(text)) % (2**32))
            return np.random.rand(384).astype(np.float32)
        return vector

    return make_encoder(encode_one)


@pytest.fixture
def db(temp_files, clustered_encoder):
    metadata_file, index_file, sqlite_db_file = temp_files
    with patch("embeddings.minilm.SentenceTransformer") as MockST:
        MockST.return_value.encode.side_effect = clustered_encoder
        yield VectorDB(
            dimension=384,
            metadata_file=metadata_file,
            index_file=index_file,
            sqlite_db_file=sqlite_db_file,
        )


def test_librarian_skips_when_too_few_nodes(db):
    result = db.run_librarian_cycle()
    assert result["status"] == "SKIPPED"
    assert result["synthesized_concepts"] == 0


def test_librarian_synthesizes_concepts_and_is_idempotent(db):
    for index in range(3):
        db.create_node(
            node_id=f"storage-{index}",
            name=f"Storage topic {index}",
            node_type="concept",
            properties={"description": "storage subsystem note"},
        )
    db.create_node(
        node_id="network-0",
        name="Network topic",
        node_type="entity",
        properties={"description": "network subsystem note"},
    )

    result = db.run_librarian_cycle()
    assert result["status"] == "COMPLETED"
    assert result["clusters_detected"] >= 1
    assert result["duplicates_detected"] >= 1  # the storage notes are near-identical

    concepts_after_first = _concept_count(db)
    assert concepts_after_first >= 3

    # A second cycle must not re-synthesize the same cluster.
    db.run_librarian_cycle()
    assert _concept_count(db) == concepts_after_first


def test_librarian_run_is_recorded(db):
    for index in range(2):
        db.create_node(
            node_id=f"storage-{index}",
            name=f"Storage topic {index}",
            node_type="entity",
            properties={"description": "storage subsystem note"},
        )
    db.run_librarian_cycle()

    rows = db.conn.execute(
        "SELECT clusters_detected, duration FROM librarian_runs"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0][1] >= 0.0


def _concept_count(db) -> int:
    return db.conn.execute(
        "SELECT COUNT(*) FROM nodes WHERE node_type = 'concept'"
    ).fetchone()[0]
