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

        def controlled_encode(text, **kwargs):
            vec = np.zeros(384, dtype=np.float32)
            # Control the vector based on keywords in the text/query
            text_lower = text.lower()
            if "python" in text_lower:
                vec[0] = 1.0
            elif "neural" in text_lower:
                vec[1] = 1.0
            elif "database" in text_lower:
                vec[2] = 1.0
            else:
                # Default fallback deterministic vector
                np.random.seed(abs(hash(text)) % (2**32))
                return np.random.rand(384).astype(np.float32)
            return vec

        instance.encode.side_effect = controlled_encode
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


def test_fts_only_search(db):
    """Verify hybrid search with lexical channel only (vector weight = 0)."""
    # Create nodes with distinct terms
    db.create_node(node_id="node-a", name="Python Programming", node_type="concept")
    db.create_node(node_id="node-b", name="Neural Networks", node_type="concept")
    db.create_node(node_id="node-c", name="Database Systems and SQL", node_type="concept")

    # Lexical only search for SQL
    results = db.search_hybrid(
        query="SQL",
        project_id="default",
        limit=10,
        channel_weights={"vector": 0.0, "lexical": 1.0}
    )

    assert len(results) == 1
    assert results[0]["id"] == "node-c"
    assert results[0]["name"] == "Database Systems and SQL"
    assert results[0]["rrf_score"] == 1.0 / 61.0  # 1-based rank 1 in lexical, 0 in vector


def test_vector_only_search(db, mock_sentence_transformer):
    """Verify hybrid search with vector channel only (lexical weight = 0)."""
    # Create nodes
    db.create_node(node_id="node-a", name="Python Programming", node_type="concept")
    db.create_node(node_id="node-b", name="Neural Networks", node_type="concept")
    db.create_node(node_id="node-c", name="Database Systems and SQL", node_type="concept")

    # Vector only search for "neural"
    results = db.search_hybrid(
        query="neural",
        project_id="default",
        limit=10,
        channel_weights={"vector": 1.0, "lexical": 0.0}
    )

    assert len(results) > 0
    # node-b should be ranked first because its vector is closest to "neural"
    assert results[0]["id"] == "node-b"
    assert results[0]["rrf_score"] == 1.0 / 61.0  # 1-based rank 1 in vector, 0 in lexical


def test_combined_rrf_search(db, mock_sentence_transformer):
    """Verify RRF ranking and scoring with both channels active."""
    # Create nodes
    # Node A is highly related to "python" semantically, but has "neural" in name lexically
    db.create_node(
        node_id="node-a", 
        name="Neural Python", 
        node_type="concept",
        properties={"description": "Python is a great language."}
    )
    # Node B is highly related to "neural" semantically, but has "python" in name lexically
    db.create_node(
        node_id="node-b", 
        name="Python Neural", 
        node_type="concept",
        properties={"description": "Neural networks are cool."}
    )

    # Let's search for "python" with weights: vector=1.0, lexical=0.5
    # For query "python":
    # - Vector channel:
    #   - Node A matches "python" (it has python in name/description). Vector is closest. Rank 1.
    #   - Node B does not match "python" semantically. Vector is further. Rank 2.
    # - Lexical channel:
    #   - Node B has "Python Neural" in name. Let's assume it matches first (Rank 1).
    #   - Node A has "Neural Python" in name. Let's assume it matches second (Rank 2).
    #
    # Score calculation:
    # - Node A:
    #   - vector rank 1 -> 1.0 / (60 + 1) = 0.016393
    #   - lexical rank 2 -> 0.5 / (60 + 2) = 0.008064
    #   - Total RRF: 0.024457
    # - Node B:
    #   - vector rank 2 -> 1.0 / (60 + 2) = 0.016129
    #   - lexical rank 1 -> 0.5 / (60 + 1) = 0.008196
    #   - Total RRF: 0.024325
    # Node A should be ranked 1st!

    results = db.search_hybrid(
        query="python",
        project_id="default",
        limit=10,
        channel_weights={"vector": 1.0, "lexical": 0.5}
    )

    assert len(results) >= 2
    assert results[0]["id"] == "node-a"
    assert results[1]["id"] == "node-b"
    assert results[0]["rrf_score"] > results[1]["rrf_score"]
    
    # Check that retrieval_count was incremented
    cursor = db.conn.cursor()
    cursor.execute("SELECT retrieval_count FROM nodes WHERE id = 'node-a'")
    assert cursor.fetchone()[0] == 1


def test_get_hologram_subgraph(db):
    """Verify hologram neighbor and context extraction up to depth."""
    # Create a chain of nodes: Node A -> Node B -> Node C -> Node D
    db.create_node(node_id="node-a", name="Node A", node_type="concept")
    db.create_node(node_id="node-b", name="Node B", node_type="concept")
    db.create_node(node_id="node-c", name="Node C", node_type="concept")
    db.create_node(node_id="node-d", name="Node D", node_type="concept")

    # Create edges
    db.create_edge(edge_id="edge-ab", from_node_id="node-a", to_node_id="node-b", relationship_type="SUPPORTS")
    db.create_edge(edge_id="edge-bc", from_node_id="node-b", to_node_id="node-c", relationship_type="DEPENDS_ON")
    db.create_edge(edge_id="edge-cd", from_node_id="node-c", to_node_id="node-d", relationship_type="RELATED_TO")

    # Add observations
    db.create_observation(obs_id="obs-1", entity_id="node-b", content="Observation B1")
    db.create_observation(obs_id="obs-2", entity_id="node-a", content="Observation A1")

    # 1. Retrieve hologram for Node B with depth = 1
    hologram_d1 = db.get_hologram(node_id="node-b", depth=1)
    
    # Should include B, A, C (1-hop neighbors) but NOT D
    nodes_d1 = {n["id"] for n in hologram_d1["nodes"]}
    assert "node-b" in nodes_d1
    assert "node-a" in nodes_d1
    assert "node-c" in nodes_d1
    assert "node-d" not in nodes_d1

    # Edges should include ab and bc, but NOT cd
    edges_d1 = {e["id"] for e in hologram_d1["edges"]}
    assert "edge-ab" in edges_d1
    assert "edge-bc" in edges_d1
    assert "edge-cd" not in edges_d1

    # Check observations are hydrated
    node_b_data = next(n for n in hologram_d1["nodes"] if n["id"] == "node-b")
    assert len(node_b_data["observations"]) == 1
    assert node_b_data["observations"][0]["content"] == "Observation B1"

    node_a_data = next(n for n in hologram_d1["nodes"] if n["id"] == "node-a")
    assert len(node_a_data["observations"]) == 1
    assert node_a_data["observations"][0]["content"] == "Observation A1"

    # 2. Retrieve hologram for Node B with depth = 2
    hologram_d2 = db.get_hologram(node_id="node-b", depth=2)

    # Should now include D
    nodes_d2 = {n["id"] for n in hologram_d2["nodes"]}
    assert "node-d" in nodes_d2

    # Edges should now include cd
    edges_d2 = {e["id"] for e in hologram_d2["edges"]}
    assert "edge-cd" in edges_d2


def test_get_hologram_not_found(db):
    """Verify get_hologram raises ValueError if node is not found."""
    with pytest.raises(ValueError, match="Node non-existent not found"):
        db.get_hologram(node_id="non-existent")
