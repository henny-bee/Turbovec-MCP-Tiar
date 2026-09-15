"""Unit tests for the extracted core, storage, embedding and search modules."""

import numpy as np
import pytest

from core.config import Settings
from core.sqlite import chunked
from core.text import build_embedding_text, extract_description, sanitize_fts_query
from embeddings.service import EmbeddingService
from graph.traversal import UnionFind
from librarian.clustering import DBSCANClustering, pairwise_distances
from search.fusion import reciprocal_rank_fusion, top_n
from storage.vector_index import VectorIndex


# --------------------------------------------------------------------------
# core helpers
# --------------------------------------------------------------------------
def test_chunked_respects_sqlite_variable_ceiling():
    batches = list(chunked(list(range(2500)), 900))
    assert [len(batch) for batch in batches] == [900, 900, 700]
    assert sum(len(batch) for batch in batches) == 2500


def test_chunked_rejects_invalid_size():
    with pytest.raises(ValueError):
        list(chunked([1, 2, 3], 0))


def test_sanitize_fts_query_quotes_terms_and_neutralises_operators():
    assert sanitize_fts_query("hello world") == '"hello" "world"'
    # FTS5 operators must not survive as syntax.
    assert sanitize_fts_query("name:foo OR bar*") == '"name" "foo" "OR" "bar"'
    assert sanitize_fts_query('say "hi"') == '"say" """hi"""'
    assert sanitize_fts_query("  ()  ") == ""


def test_extract_description_handles_every_property_shape():
    assert extract_description(None) == ""
    assert extract_description({"description": "d"}) == "d"
    assert extract_description('{"description": "d"}') == "d"
    assert extract_description("not json") == "not json"


def test_build_embedding_text_truncates_observations():
    text = build_embedding_text(
        "N",
        "T",
        "D",
        ["x" * 10, "y", "z"],
        max_observations=2,
        max_chars_per_observation=3,
    )
    assert text == "N T D xxx y"


def test_settings_overrides_take_precedence_over_environment(monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "123")
    monkeypatch.setenv("RRF_K", "10")
    settings = Settings.from_env(chunk_size=999)
    assert settings.chunk_size == 999  # explicit override wins
    assert settings.rrf_k == 10.0  # environment still read


def test_settings_resolves_database_next_to_metadata():
    settings = Settings.from_env(metadata_file="/data/metadata.json")
    assert settings.resolved_sqlite_path().replace("\\", "/") == "/data/memory.db"


# --------------------------------------------------------------------------
# rank fusion
# --------------------------------------------------------------------------
def test_reciprocal_rank_fusion_weights_and_ranks():
    scores = reciprocal_rank_fusion(((["a", "b"], 1.0), (["b"], 0.5)), k=60.0)
    assert scores["a"] == pytest.approx(1 / 61)
    assert scores["b"] == pytest.approx(1 / 62 + 0.5 / 61)
    assert [node for node, _ in top_n(scores, 1)] == ["b"]


def test_reciprocal_rank_fusion_ignores_empty_channels():
    assert reciprocal_rank_fusion((([], 1.0), ([], 1.0))) == {}


# --------------------------------------------------------------------------
# embedding service
# --------------------------------------------------------------------------
class _CountingProvider:
    """Records every call so caching and batching are observable."""

    dimension = 4

    def __init__(self):
        self.calls = []

    @property
    def model(self):
        return self

    def encode(self, text):
        self.calls.append(text)
        if isinstance(text, list):
            return np.asarray([[len(t)] * 4 for t in text], dtype=np.float32)
        return np.asarray([len(text)] * 4, dtype=np.float32)


def test_embedding_service_caches_repeated_text():
    provider = _CountingProvider()
    service = EmbeddingService(provider, cache_size=8)

    first = service.encode("hello")
    second = service.encode("hello")

    assert np.array_equal(first, second)
    assert provider.calls == ["hello"]
    assert service.cache_stats()["hits"] == 1


def test_embedding_service_batches_misses_into_one_call():
    provider = _CountingProvider()
    service = EmbeddingService(provider, cache_size=8, batch_size=64)

    matrix = service.encode_matrix(["a", "bb", "ccc"])

    assert matrix.shape == (3, 4)
    assert provider.calls == [["a", "bb", "ccc"]]
    # A second pass is served entirely from the cache.
    service.encode_matrix(["a", "bb", "ccc"])
    assert len(provider.calls) == 1


def test_embedding_service_respects_cache_capacity():
    provider = _CountingProvider()
    service = EmbeddingService(provider, cache_size=2)
    for text in ("a", "b", "c"):
        service.encode(text)
    assert service.cache_stats()["size"] == 2


# --------------------------------------------------------------------------
# vector index
# --------------------------------------------------------------------------
def test_vector_index_defers_writes_until_threshold(tmp_path):
    path = str(tmp_path / "index.tvim")
    index = VectorIndex(8, path, flush_ops=3, flush_interval=1e9)

    for identifier in range(2):
        index.add_one(np.ones(8, dtype=np.float32), identifier)
    assert not (tmp_path / "index.tvim").exists()  # still buffered

    index.add_one(np.ones(8, dtype=np.float32), 99)
    assert (tmp_path / "index.tvim").exists()  # threshold crossed
    assert len(index) == 3


def test_vector_index_flush_forces_a_write(tmp_path):
    path = str(tmp_path / "index.tvim")
    index = VectorIndex(8, path, flush_ops=1000, flush_interval=1e9)
    index.add_one(np.ones(8, dtype=np.float32), 1)

    assert not (tmp_path / "index.tvim").exists()
    assert index.flush(force=True) is True
    assert (tmp_path / "index.tvim").exists()


def test_vector_index_reload_round_trips(tmp_path):
    path = str(tmp_path / "index.tvim")
    index = VectorIndex(8, path, flush_ops=1, flush_interval=1e9)
    index.add_one(np.ones(8, dtype=np.float32), 7)
    index.flush(force=True)

    reloaded = VectorIndex(8, path)
    assert reloaded.load() is True
    assert 7 in reloaded


def test_vector_index_load_is_false_when_missing(tmp_path):
    assert VectorIndex(8, str(tmp_path / "absent.tvim")).load() is False


# --------------------------------------------------------------------------
# connectivity and clustering
# --------------------------------------------------------------------------
def test_union_find_tracks_components():
    components = UnionFind(["a", "b", "c"])
    assert not components.connected("a", "c")
    components.union("a", "b")
    components.union("b", "c")
    assert components.connected("a", "c")


def test_pairwise_distances_matches_direct_norm():
    rng = np.random.default_rng(0)
    points = rng.random((6, 3)).astype(np.float32)
    expected = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    assert np.allclose(pairwise_distances(points), expected, atol=1e-5)


def test_dbscan_separates_two_dense_groups():
    points = np.asarray(
        [[0, 0], [0.05, 0], [0, 0.05], [5, 5], [5.05, 5], [50, 50]],
        dtype=np.float32,
    )
    labels = DBSCANClustering(eps=0.5, min_samples=2).fit(points)

    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4]
    assert labels[0] != labels[3]
    assert labels[5] == -1  # isolated point is noise


def test_dbscan_handles_empty_input():
    assert DBSCANClustering().fit(np.zeros((0, 4), dtype=np.float32)) == []
