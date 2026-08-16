import os
import json
import sqlite3
import datetime
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


def test_extraction_pipeline_fallback(db):
    """Verify regex fallback extraction of entities, definitions, preferences, and relationships."""
    # Ensure spaCy is treated as missing to test regex fallback
    with patch.dict("sys.modules", {"spacy": None}):
        text = (
            "John Doe works at Google.\n"
            "Python is a programming language.\n"
            "I prefer VS Code.\n"
            "Alice lives in Paris."
        )

        result = db.extract_entities_and_relations(text)

        assert "entities" in result
        assert "relationships" in result
        assert "observations" in result

        # Verify entities
        entities = {e["name"]: e for e in result["entities"]}
        assert "John Doe" in entities
        assert "Google" in entities
        assert "Python" in entities
        assert "VS Code" in entities
        assert "Alice" in entities
        assert "Paris" in entities
        assert "User" in entities

        assert entities["Python"]["properties"]["description"] == "programming language"
        assert entities["VS Code"]["type"] == "PREFERENCE"

        # Verify relationships
        rels = result["relationships"]
        assert len(rels) >= 2

        # Check "John Doe works at Google" relationship
        works_at_rel = [
            r for r in rels if r["from"] == "John Doe" and r["to"] == "Google"
        ]
        assert len(works_at_rel) == 1
        assert works_at_rel[0]["type"] == "WORKS_AT"

        # Check "Alice lives in Paris" relationship
        lives_in_rel = [r for r in rels if r["from"] == "Alice" and r["to"] == "Paris"]
        assert len(lives_in_rel) == 1
        assert lives_in_rel[0]["type"] == "LIVES_IN"

        # Verify observations
        obs = result["observations"]
        assert len(obs) >= 2

        # Check definitions and preferences as observations
        python_obs = [o for o in obs if o["entity_name"] == "Python"]
        assert len(python_obs) == 1
        assert "is programming language" in python_obs[0]["content"]

        user_obs = [o for o in obs if o["entity_name"] == "User"]
        assert len(user_obs) == 1
        assert "Prefers VS Code" in user_obs[0]["content"]


def test_session_management(db):
    """Verify session start, end, and breakthrough recording."""
    # 1. Start Session 1
    session_id_1 = "session-001"
    name_1 = "Initial Exploration"
    properties_1 = {"focus": "Architecture"}

    sid_1 = db.start_session(session_id_1, name_1, properties_1)
    assert sid_1 == session_id_1

    # Check in SQLite
    cursor = db.conn.cursor()
    cursor.execute(
        "SELECT name, node_type, properties FROM nodes WHERE id = ?", (session_id_1,)
    )
    row = cursor.fetchone()
    assert row is not None
    assert row[0] == name_1
    assert row[1] == "session"
    props = json.loads(row[2])
    assert props["focus"] == "Architecture"
    assert props["status"] == "active"

    # 2. Record Breakthrough
    bt_title = "Discovered SQLite WAL speedup"
    bt_content = "Configuring WAL mode improves transaction throughput significantly."
    bt_id = db.record_breakthrough(session_id_1, bt_title, bt_content)
    assert bt_id is not None

    # Check breakthrough node
    cursor.execute(
        "SELECT name, node_type, properties FROM nodes WHERE id = ?", (bt_id,)
    )
    row = cursor.fetchone()
    assert row is not None
    assert row[0] == bt_title
    assert row[1] == "breakthrough"
    assert json.loads(row[2]) == {"content": bt_content}

    # Check breakthrough edge linking to session
    cursor.execute(
        "SELECT from_node_id, to_node_id, relationship_type FROM edges WHERE from_node_id = ?",
        (bt_id,),
    )
    edge_row = cursor.fetchone()
    assert edge_row is not None
    assert edge_row[0] == bt_id
    assert edge_row[1] == session_id_1
    assert edge_row[2] == "BREAKTHROUGH_IN"

    # 3. Start Session 2 (should link to Session 1 via PRECEDED_BY)
    session_id_2 = "session-002"
    name_2 = "Second Exploration"
    sid_2 = db.start_session(session_id_2, name_2)
    assert sid_2 == session_id_2

    # Check PRECEDED_BY edge between session 2 and session 1
    cursor.execute(
        "SELECT from_node_id, to_node_id, relationship_type FROM edges WHERE from_node_id = ? AND to_node_id = ?",
        (session_id_2, session_id_1),
    )
    edge_row = cursor.fetchone()
    assert edge_row is not None
    assert edge_row[2] == "PRECEDED_BY"

    # 4. End Session 1
    db.end_session(
        session_id_1, "Completed initial architecture exploration and WAL benchmark."
    )

    # Check updated properties
    cursor.execute("SELECT properties FROM nodes WHERE id = ?", (session_id_1,))
    row = cursor.fetchone()
    props = json.loads(row[0])
    assert props["status"] == "closed"
    assert "summary" in props
    assert "ended_at" in props


def test_point_in_time_queries(db):
    """Verify that get_graph_state_as_of correctly filters the graph state historically."""
    # Define timestamps
    t0 = "2026-08-16T08:00:00Z"
    t1 = "2026-08-16T08:10:00Z"
    t2 = "2026-08-16T08:20:00Z"
    t3 = "2026-08-16T08:30:00Z"

    # Create Node A at t1
    db.create_node(node_id="node-a", name="Node A", node_type="entity")
    with db.conn:
        db.conn.execute("UPDATE nodes SET created_at = ? WHERE id = 'node-a'", (t1,))

    # Create Node B at t2
    db.create_node(node_id="node-b", name="Node B", node_type="entity")
    with db.conn:
        db.conn.execute("UPDATE nodes SET created_at = ? WHERE id = 'node-b'", (t2,))

    # Create edge A -> B at t2
    db.create_edge(
        edge_id="edge-ab",
        from_node_id="node-a",
        to_node_id="node-b",
        relationship_type="DEPENDS_ON",
    )
    with db.conn:
        db.conn.execute("UPDATE edges SET created_at = ? WHERE id = 'edge-ab'", (t2,))

    # Create observation on Node A at t3
    db.create_observation(
        obs_id="obs-a-1", entity_id="node-a", content="Observation at t3"
    )
    with db.conn:
        db.conn.execute(
            "UPDATE observations SET created_at = ? WHERE id = 'obs-a-1'", (t3,)
        )

    # Query as of t0 (before everything)
    state_t0 = db.get_graph_state_as_of(t0)
    assert len(state_t0["nodes"]) == 0
    assert len(state_t0["edges"]) == 0

    # Query as of t1 (only Node A exists, no observations, no edges)
    state_t1 = db.get_graph_state_as_of(t1)
    assert len(state_t1["nodes"]) == 1
    assert state_t1["nodes"][0]["id"] == "node-a"
    assert len(state_t1["nodes"][0]["observations"]) == 0
    assert len(state_t1["edges"]) == 0

    # Query as of t2 (Node A, Node B, and Edge A -> B exist, but no observation)
    state_t2 = db.get_graph_state_as_of(t2)
    assert len(state_t2["nodes"]) == 2
    node_ids = {n["id"] for n in state_t2["nodes"]}
    assert node_ids == {"node-a", "node-b"}

    # Check observations on Node A at t2
    node_a_t2 = [n for n in state_t2["nodes"] if n["id"] == "node-a"][0]
    assert len(node_a_t2["observations"]) == 0

    assert len(state_t2["edges"]) == 1
    assert state_t2["edges"][0]["id"] == "edge-ab"

    # Query as of t3 (everything exists, including observation on Node A)
    state_t3 = db.get_graph_state_as_of(t3)
    assert len(state_t3["nodes"]) == 2
    assert len(state_t3["edges"]) == 1

    node_a_t3 = [n for n in state_t3["nodes"] if n["id"] == "node-a"][0]
    assert len(node_a_t3["observations"]) == 1
    assert node_a_t3["observations"][0]["id"] == "obs-a-1"
    assert node_a_t3["observations"][0]["content"] == "Observation at t3"
