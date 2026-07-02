import tempfile

import pytest

pytest.importorskip("infino")
pytest.importorskip("pyarrow")

from mem0.vector_stores.infino import InfinoVectorStore, OutputData  # noqa: E402

DIM = 16


def vec(seed: int):
    return [float((seed + i) % 5) / 5 for i in range(DIM)]


@pytest.fixture
def store():
    with tempfile.TemporaryDirectory() as tmp:
        yield InfinoVectorStore(
            collection_name="test",
            path=tmp,
            embedding_model_dims=DIM,
            n_cent=1,
            distance_strategy="cosine",
        )


def test_insert_and_get(store):
    store.insert(
        [vec(1), vec(2)],
        [{"data": "alpha", "user_id": "u1"}, {"data": "beta", "user_id": "u1"}],
        ["a", "b"],
    )
    assert store.col_info()["count"] == 2
    got = store.get("a")
    assert isinstance(got, OutputData)
    assert got.id == "a"
    assert got.payload["data"] == "alpha"


def test_search_returns_similarity(store):
    store.insert([vec(1), vec(2)], [{"data": "alpha"}, {"data": "beta"}], ["a", "b"])
    results = store.search("alpha", vec(1), top_k=2)
    assert results
    assert results[0].id == "a"  # nearest vector ranked first
    # search() must return similarity in [0, 1], higher = better (mem0 contract)
    assert all(0.0 <= r.score <= 1.0 for r in results)


def test_search_filters(store):
    store.insert(
        [vec(1), vec(2)],
        [{"data": "x", "user_id": "u1"}, {"data": "y", "user_id": "u2"}],
        ["a", "b"],
    )
    results = store.search("x", vec(1), top_k=5, filters={"user_id": "u2"})
    assert results
    assert all(r.payload.get("user_id") == "u2" for r in results)


def test_keyword_search_bm25(store):
    store.insert(
        [vec(1), vec(2)],
        [{"data": "the quick brown fox"}, {"data": "lazy dog sleeps"}],
        ["a", "b"],
    )
    results = store.keyword_search("fox", top_k=2)
    assert results
    assert "fox" in results[0].payload["data"]


def test_update(store):
    store.insert([vec(1)], [{"data": "old", "user_id": "u1"}], ["a"])
    store.update("a", vector=vec(3), payload={"data": "new", "user_id": "u1"})
    assert store.get("a").payload["data"] == "new"
    assert store.col_info()["count"] == 1


def test_delete(store):
    store.insert([vec(1), vec(2)], [{"data": "x"}, {"data": "y"}], ["a", "b"])
    store.delete("a")
    assert store.get("a") is None
    assert store.col_info()["count"] == 1


def test_list_with_filters(store):
    store.insert(
        [vec(1), vec(2)],
        [{"data": "x", "user_id": "u1"}, {"data": "y", "user_id": "u2"}],
        ["a", "b"],
    )
    batches = store.list(filters={"user_id": "u1"}, top_k=10)
    assert len(batches) == 1
    assert len(batches[0]) == 1
    assert batches[0][0].payload["user_id"] == "u1"


def test_reset(store):
    store.insert([vec(1)], [{"data": "x"}], ["a"])
    store.reset()
    assert store.col_info()["count"] == 0
