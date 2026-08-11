import os
import json
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from vector_db import VectorDB


@pytest.fixture
def temp_files(tmp_path):
    metadata_file = tmp_path / "metadata.json"
    index_file = tmp_path / "index.tvim"
    return str(metadata_file), str(index_file)


@pytest.fixture
def mock_sentence_transformer():
    with patch("vector_db.SentenceTransformer") as MockST:
        instance = MockST.return_value

        def mock_encode(text, **kwargs):
            # We create a deterministic vector based on string hash
            # If the text is exactly a specific test query, make it close to a test document
            # But normally we just want some floats
            np.random.seed(abs(hash(text)) % (2**32))
            return np.random.rand(384).astype(np.float32)

        instance.encode.side_effect = mock_encode
        yield instance


@pytest.fixture
def vector_db(temp_files, mock_sentence_transformer):
    metadata_file, index_file = temp_files
    db = VectorDB(dimension=384, metadata_file=metadata_file, index_file=index_file)
    return db


def test_initialization(vector_db, temp_files):
    metadata_file, index_file = temp_files
    assert vector_db.dimension == 384
    assert vector_db.metadata_file == metadata_file
    assert vector_db.index_file == index_file
    assert vector_db.document_store == {}
    assert vector_db.model is not None
    assert vector_db.index is not None


def test_chunk_text(vector_db):
    text = "A" * 1500
    chunks = vector_db.chunk_text(text, chunk_size=1000, overlap=200)
    assert len(chunks) == 2
    assert len(chunks[0]) == 1000
    assert len(chunks[1]) == 700  # 1500 - (1000 - 200) = 700


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [(0, 0), (-1, 0), (100, -1), (100, 100), (100, 101)],
)
def test_chunk_text_rejects_invalid_configuration(vector_db, chunk_size, overlap):
    with pytest.raises(ValueError):
        vector_db.chunk_text("content", chunk_size=chunk_size, overlap=overlap)


def test_add_knowledge(vector_db):
    text = "This is a test document."
    title = "Test Doc"

    result = vector_db.add_knowledge(title, text)

    assert "Successfully added" in result
    assert len(vector_db.document_store) == 1
    doc = vector_db.document_store[0]
    assert doc["title"] == title
    assert doc["content"] == text
    assert "deleted" not in doc
    assert doc["id"] == 0

    # Check if files were created
    assert os.path.exists(vector_db.metadata_file)
    assert os.path.exists(vector_db.index_file)

    # Verify that metadata can be loaded
    with open(vector_db.metadata_file, "r") as f:
        stored_metadata = json.load(f)
    assert len(stored_metadata["documents"]) == 1
    assert stored_metadata["documents"]["0"]["title"] == title


def test_add_knowledge_rejects_empty_input(vector_db):
    assert vector_db.add_knowledge("", "content") == "Error: Title must not be empty."
    assert vector_db.add_knowledge("title", "  ") == "Error: Content must not be empty."
    assert vector_db.document_store == {}


def test_add_knowledge_does_not_mutate_index_when_encoding_fails(
    vector_db, mock_sentence_transformer
):
    mock_sentence_transformer.encode.side_effect = [
        np.ones(384, dtype=np.float32),
        RuntimeError("encoding failed"),
    ]

    result = vector_db.add_knowledge("Doc", "A" * 1200)

    assert result == "Error: Failed to process 'Doc'."
    assert vector_db.document_store == {}
    assert len(vector_db.index) == 0


def test_search_knowledge(vector_db, mock_sentence_transformer):
    # Instead of random vectors, let's control the encode function specifically for search test
    def controlled_encode(text, **kwargs):
        vec = np.zeros(384, dtype=np.float32)
        if "apple" in text.lower():
            vec[0] = 1.0
        elif "banana" in text.lower():
            vec[1] = 1.0
        else:
            vec[2] = 1.0
        return vec

    mock_sentence_transformer.encode.side_effect = controlled_encode

    vector_db.add_knowledge("Doc 1", "I like apple.")
    vector_db.add_knowledge("Doc 2", "I like banana.")

    # Search for apple
    res = vector_db.search_knowledge("Where is the apple?", top_k=1)

    assert "Doc 1" in res
    assert "Doc 2" not in res

    # Search for banana
    res2 = vector_db.search_knowledge("Give me banana", top_k=1)
    assert "Doc 2" in res2
    assert "Doc 1" not in res2


def test_search_knowledge_validates_arguments(vector_db):
    vector_db.add_knowledge("Doc", "Content")

    assert vector_db.search_knowledge("", top_k=1) == (
        "Error: Search query must not be empty."
    )
    assert vector_db.search_knowledge("Content", top_k=0) == (
        "Error: top_k must be greater than zero."
    )


def test_delete_knowledge(vector_db):
    vector_db.add_knowledge("Doc 1", "Content 1")
    vector_db.add_knowledge("Doc 2", "Content 2")

    assert len(vector_db.document_store) == 2

    # Hard delete
    res = vector_db.delete_knowledge("Doc 1")
    assert "Successfully deleted 1 chunks" in res

    assert len(vector_db.document_store) == 1
    assert 0 not in vector_db.document_store
    assert 1 in vector_db.document_store


def test_optimize_index(vector_db):
    vector_db.add_knowledge("Doc 1", "Content 1")
    vector_db.add_knowledge("Doc 2", "Content 2")
    vector_db.add_knowledge("Doc 3", "Content 3")

    vector_db.delete_knowledge("Doc 2")

    assert len(vector_db.document_store) == 2
    assert 1 not in vector_db.document_store

    res = vector_db.optimize_index()
    assert "Optimization complete" in res
    assert "Current size: 2 chunks" in res


def test_load_storage_rebuilds_mismatched_index(temp_files, mock_sentence_transformer):
    metadata_file, index_file = temp_files
    db = VectorDB(dimension=384, metadata_file=metadata_file, index_file=index_file)
    db.add_knowledge("Doc", "Content")

    empty_index = MagicMock()
    empty_index.__len__.return_value = 0
    with patch("vector_db.turbovec.IdMapIndex") as mock_index_type:
        rebuilt_index = MagicMock()
        rebuilt_index.__len__.return_value = 0
        mock_index_type.return_value = rebuilt_index
        empty_index.load.return_value = None
        mock_index_type.load.side_effect = Exception("failed to load")

        loaded_db = VectorDB(
            dimension=384, metadata_file=metadata_file, index_file=index_file
        )

    assert loaded_db.document_store[0]["title"] == "Doc"
    rebuilt_index.add_with_ids.assert_called_once()
    rebuilt_index.write.assert_called_once_with(index_file)


def test_clear_memory(vector_db):
    vector_db.add_knowledge("Doc 1", "Content")
    assert len(vector_db.document_store) == 1
    assert os.path.exists(vector_db.metadata_file)

    vector_db.clear_memory()

    assert len(vector_db.document_store) == 0
    assert not os.path.exists(vector_db.metadata_file)
    assert not os.path.exists(vector_db.index_file)
