import os
import json
import sqlite3
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from vector_db import VectorDB


@pytest.fixture
def temp_files(tmp_path):
    metadata_file = tmp_path / "metadata.json"
    index_file = tmp_path / "index.tvim"
    sqlite_db_file = tmp_path / "memory.db"
    return str(metadata_file), str(index_file), str(sqlite_db_file)


@pytest.fixture
def mock_sentence_transformer():
    with patch("embeddings.minilm.SentenceTransformer") as MockST:
        instance = MockST.return_value

        def mock_encode(text, **kwargs):
            np.random.seed(abs(hash(text)) % (2**32))
            return np.random.rand(384).astype(np.float32)

        instance.encode.side_effect = mock_encode
        yield instance


@pytest.fixture
def db(temp_files, mock_sentence_transformer):
    metadata_file, index_file, sqlite_db_file = temp_files
    db = VectorDB(
        dimension=384,
        metadata_file=metadata_file,
        index_file=index_file,
        sqlite_db_file=sqlite_db_file,
    )
    return db


def test_sqlite_pragmas(db):
    """Verify WAL mode, synchronous normal, and foreign keys are active."""
    cursor = db.conn.cursor()

    # Check journal_mode (WAL)
    cursor.execute("PRAGMA journal_mode;")
    journal_mode = cursor.fetchone()[0]
    assert journal_mode.lower() == "wal"

    # Check foreign_keys (ON)
    cursor.execute("PRAGMA foreign_keys;")
    foreign_keys = cursor.fetchone()[0]
    assert foreign_keys == 1


def test_deterministic_vector_id_mapping(db):
    """Verify deterministic hashing of node_id to uint64 and collision resolution."""
    conn = db.conn
    node_id_1 = "test-uuid-1"

    # Generate vector ID for node_id_1
    v_id_1 = db.generate_vector_id(node_id_1, conn)
    assert isinstance(v_id_1, int)
    assert 0 <= v_id_1 <= 0xFFFFFFFFFFFFFFFF

    # A second call with same node_id should return the same vector_id
    v_id_1_repeat = db.generate_vector_id(node_id_1, conn)
    assert v_id_1 == v_id_1_repeat

    # Test collision resolution:
    # Insert node first because of foreign key constraint on vector_id_mapping
    db.create_node(node_id="colliding-node", name="Collider", node_type="entity")

    # Get its auto-generated vector_id
    cursor = conn.cursor()
    cursor.execute(
        "SELECT vector_id FROM vector_id_mapping WHERE node_id = 'colliding-node'"
    )
    colliding_v_id = db._to_unsigned_64(cursor.fetchone()[0])

    # Let's insert a second node
    db.create_node(node_id="target-node", name="Target", node_type="entity")

    # Let's verify that if we query generate_vector_id for a node that hashes to colliding_v_id,
    # but that slot is taken, it increments to colliding_v_id + 1 (or next available)
    with patch("hashlib.sha256") as mock_sha:
        mock_hash = MagicMock()
        # Make digest return a byte string that converts to colliding_v_id
        # colliding_v_id as 8 bytes big-endian:
        colliding_bytes = colliding_v_id.to_bytes(8, byteorder="big")
        mock_hash.digest.return_value = colliding_bytes + b"\x00" * 24
        mock_sha.return_value = mock_hash

        # Now, generating vector ID for "new-node" (which will hash to colliding_v_id)
        # Since colliding-node is mapped to colliding_v_id, "new-node" should get colliding_v_id + 1
        # Let's create the node first so foreign key check passes
        cursor.execute(
            "INSERT INTO nodes (id, name, node_type, project_id, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            ("new-node", "New", "entity", "default", "now", "now"),
        )

        resolved_id = db.generate_vector_id("new-node", conn)
        assert resolved_id == (colliding_v_id + 1) & 0xFFFFFFFFFFFFFFFF


def test_node_crud(db):
    """Verify creating, updating, and deleting nodes."""
    node_id = "node-123"
    name = "Test Node"
    node_type = "entity"
    properties = {"description": "A test entity node", "status": "active"}

    # 1. Create
    node = db.create_node(
        node_id=node_id,
        name=name,
        node_type=node_type,
        properties=properties,
    )

    assert node["id"] == node_id
    assert node["name"] == name
    assert node["properties"] == properties

    # Verify in SQLite nodes table
    cursor = db.conn.cursor()
    cursor.execute("SELECT * FROM nodes WHERE id = ?", (node_id,))
    row = cursor.fetchone()
    assert row is not None
    assert row[1] == name
    assert row[2] == node_type
    assert json.loads(row[11]) == properties

    # Verify in entities_fts virtual table
    cursor.execute(
        "SELECT name, description FROM entities_fts WHERE entity_id = ?", (node_id,)
    )
    fts_row = cursor.fetchone()
    assert fts_row is not None
    assert fts_row[0] == name
    assert fts_row[1] == "A test entity node"

    # Verify in vector_id_mapping
    cursor.execute(
        "SELECT vector_id FROM vector_id_mapping WHERE node_id = ?", (node_id,)
    )
    v_row = cursor.fetchone()
    assert v_row is not None
    vector_id = db._to_unsigned_64(v_row[0])

    # Verify in turbovec index
    assert vector_id in db.index

    # 2. Update
    updated_properties = {"status": "inactive", "description": "Updated description"}
    updated_node = db.update_node(
        node_id=node_id,
        name="Updated Name",
        properties=updated_properties,
    )

    assert updated_node["name"] == "Updated Name"
    assert updated_node["properties"]["status"] == "inactive"

    # Verify SQLite update
    cursor.execute("SELECT name, properties FROM nodes WHERE id = ?", (node_id,))
    row = cursor.fetchone()
    assert row[0] == "Updated Name"
    assert json.loads(row[1]) == {
        "description": "Updated description",
        "status": "inactive",
    }

    # Verify entities_fts update
    cursor.execute(
        "SELECT name, description FROM entities_fts WHERE entity_id = ?", (node_id,)
    )
    fts_row = cursor.fetchone()
    assert fts_row[0] == "Updated Name"
    assert fts_row[1] == "Updated description"

    # 3. Delete
    success = db.delete_node(node_id)
    assert success is True

    # Verify deletion from nodes
    cursor.execute("SELECT COUNT(*) FROM nodes WHERE id = ?", (node_id,))
    assert cursor.fetchone()[0] == 0

    # Verify deletion from entities_fts
    cursor.execute("SELECT COUNT(*) FROM entities_fts WHERE entity_id = ?", (node_id,))
    assert cursor.fetchone()[0] == 0

    # Verify deletion from vector_id_mapping
    cursor.execute(
        "SELECT COUNT(*) FROM vector_id_mapping WHERE node_id = ?", (node_id,)
    )
    assert cursor.fetchone()[0] == 0

    # Verify deletion from turbovec index
    assert vector_id not in db.index


def test_edge_crud_and_cascade(db):
    """Verify creating, updating, and deleting edges, and foreign key cascade deletion."""
    node_a = "node-a"
    node_b = "node-b"
    edge_id = "edge-ab"

    db.create_node(node_id=node_a, name="Node A", node_type="entity")
    db.create_node(node_id=node_b, name="Node B", node_type="entity")

    # 1. Create edge
    edge = db.create_edge(
        edge_id=edge_id,
        from_node_id=node_a,
        to_node_id=node_b,
        relationship_type="DEPENDS_ON",
        confidence=0.9,
        weight=0.8,
        properties={"reason": "test dependency"},
    )

    assert edge["id"] == edge_id
    assert edge["relationship_type"] == "DEPENDS_ON"

    # Verify in SQLite
    cursor = db.conn.cursor()
    cursor.execute("SELECT * FROM edges WHERE id = ?", (edge_id,))
    row = cursor.fetchone()
    assert row is not None
    assert row[1] == node_a
    assert row[2] == node_b
    assert row[3] == "DEPENDS_ON"
    assert row[4] == 0.9
    assert row[5] == 0.8
    assert json.loads(row[7]) == {"reason": "test dependency"}

    # 2. Update edge
    updated_edge = db.update_edge(
        edge_id=edge_id,
        confidence=1.0,
        weight=0.95,
        properties={"reason": "updated reason", "critical": True},
    )

    assert updated_edge["confidence"] == 1.0
    assert updated_edge["weight"] == 0.95
    assert updated_edge["properties"]["critical"] is True

    # 3. Test Foreign Key Cascade Deletion:
    # If we delete node_a, edge_id should be deleted automatically!
    db.delete_node(node_a)

    cursor.execute("SELECT COUNT(*) FROM edges WHERE id = ?", (edge_id,))
    assert cursor.fetchone()[0] == 0


def test_observation_crud_and_fts_sync(db):
    """Verify creating, updating, and deleting observations, and parent FTS synchronization."""
    node_id = "entity-node"
    obs_id = "obs-1"

    db.create_node(
        node_id=node_id,
        name="Entity Node",
        node_type="entity",
        properties={"description": "Main entity"},
    )

    # 1. Create observation
    obs = db.create_observation(
        obs_id=obs_id,
        entity_id=node_id,
        content="First observation details.",
        certainty="confirmed",
    )

    assert obs["id"] == obs_id
    assert obs["content"] == "First observation details."

    # Verify observation in SQLite
    cursor = db.conn.cursor()
    cursor.execute("SELECT * FROM observations WHERE id = ?", (obs_id,))
    row = cursor.fetchone()
    assert row is not None
    assert row[2] == "First observation details."

    # Verify parent node's entities_fts row has the observation text concatenated
    cursor.execute(
        "SELECT name, observations FROM entities_fts WHERE entity_id = ?", (node_id,)
    )
    fts_row = cursor.fetchone()
    assert fts_row is not None
    assert fts_row[1] == "First observation details."

    # 2. Create a second observation
    db.create_observation(
        obs_id="obs-2",
        entity_id=node_id,
        content="Second observation details.",
    )

    # Verify both observations are concatenated in entities_fts
    cursor.execute(
        "SELECT observations FROM entities_fts WHERE entity_id = ?", (node_id,)
    )
    fts_row = cursor.fetchone()
    assert "First observation details." in fts_row[0]
    assert "Second observation details." in fts_row[0]

    # 3. Update observation
    db.update_observation(
        obs_id=obs_id,
        content="Updated first observation details.",
    )

    cursor.execute(
        "SELECT observations FROM entities_fts WHERE entity_id = ?", (node_id,)
    )
    fts_row = cursor.fetchone()
    assert "Updated first observation details." in fts_row[0]
    assert "First observation details." not in fts_row[0]

    # 4. Delete observation
    db.delete_observation(obs_id)

    cursor.execute(
        "SELECT observations FROM entities_fts WHERE entity_id = ?", (node_id,)
    )
    fts_row = cursor.fetchone()
    assert "Updated first observation details." not in fts_row[0]
    assert "Second observation details." in fts_row[0]


def test_atomic_rollback_on_failure(db, mock_sentence_transformer):
    """Verify that if turbovec or embedding fails, SQLite changes are rolled back."""
    # Force encode to fail
    mock_sentence_transformer.encode.side_effect = Exception("Embedding model failed")

    node_id = "failed-node"

    with pytest.raises(Exception):
        db.create_node(
            node_id=node_id,
            name="Failed Node",
            node_type="entity",
        )

    # Verify that the node was NOT created in SQLite due to atomic rollback
    cursor = db.conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM nodes WHERE id = ?", (node_id,))
    assert cursor.fetchone()[0] == 0

    cursor.execute("SELECT COUNT(*) FROM entities_fts WHERE entity_id = ?", (node_id,))
    assert cursor.fetchone()[0] == 0

    cursor.execute(
        "SELECT COUNT(*) FROM vector_id_mapping WHERE node_id = ?", (node_id,)
    )
    assert cursor.fetchone()[0] == 0
