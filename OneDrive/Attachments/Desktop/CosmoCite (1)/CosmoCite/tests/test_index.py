import numpy as np
import pytest
from index import FaissIndex, NumpyIndex, best, load_index


def unit(n=200, dim=16):
    emb = np.random.default_rng(0).normal(size=(n, dim)).astype(np.float32)
    return emb / np.linalg.norm(emb, axis=1, keepdims=True)


def test_best_orders_by_score_and_handles_k_larger_than_the_list():
    assert best(np.array([0.1, 0.9, 0.5]), 2).tolist() == [1, 2]
    assert len(best(np.array([0.1, 0.9]), 9)) == 2


def test_numpy_index_returns_best_matches_first():
    emb = np.eye(3, dtype=np.float32)
    ids, scores = NumpyIndex(emb).top(emb[1], 2)
    assert ids[0] == 1 and scores[0] == pytest.approx(1.0) and len(ids) == 2


def test_top_handles_k_larger_than_the_index():
    assert len(NumpyIndex(np.eye(2, dtype=np.float32)).top(np.ones(2, dtype=np.float32), 9)[0]) == 2


def test_faiss_agrees_with_numpy_on_the_nearest_neighbour():
    pytest.importorskip("faiss")
    emb = unit()
    assert FaissIndex(emb).top(emb[7], 1)[0][0] == 7


def test_faiss_never_returns_padding_when_k_exceeds_the_index():
    pytest.importorskip("faiss")
    emb = unit(5)
    ids, _ = FaissIndex(emb).top(emb[0], 50)
    assert sorted(ids.tolist()) == [0, 1, 2, 3, 4]


def test_a_saved_faiss_index_is_reused(tmp_path):
    pytest.importorskip("faiss")
    emb, path = unit(), tmp_path / "i.faiss"
    FaissIndex(emb, path)
    before = path.stat().st_mtime_ns
    assert FaissIndex(emb, path).top(emb[3], 1)[0][0] == 3
    assert path.stat().st_mtime_ns == before


def test_a_saved_faiss_index_is_rebuilt_after_papers_are_added(tmp_path):
    pytest.importorskip("faiss")
    emb, path = unit(), tmp_path / "i.faiss"
    FaissIndex(emb[:100], path)
    assert FaissIndex(emb, path).top(emb[150], 1)[0][0] == 150


def test_unknown_index_kind_is_rejected():
    with pytest.raises(ValueError):
        load_index("magic", np.eye(2, dtype=np.float32))
