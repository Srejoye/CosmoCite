import json
import numpy as np
import pytest
import retrieval
from index import NumpyIndex
from retrieval import Retriever, fuse


def fake():
    r = Retriever.__new__(Retriever)
    r.chunks = [{"paper_id": p} for p in ("a", "a", "b", "c")]
    r.emb = np.array([[1, 0], [1, 0], [0.9, 0.1], [0, 1]], dtype=np.float32)
    r.titles = {"a": "A", "b": "B", "c": "C"}
    r.pmc = {"a": "PMC1", "b": "PMC2", "c": ""}
    return r


def test_related_ranks_by_similarity_and_leaves_out_the_paper_itself():
    assert [p["paper_id"] for p in fake().related("a", 2)] == ["b", "c"]


def test_related_returns_titles_and_links():
    assert fake().related("a", 1)[0] == {"paper_id": "b", "title": "B", "pmc_id": "PMC2"}


def test_related_rejects_an_unknown_paper():
    with pytest.raises(KeyError):
        fake().related("zzz", 2)


def test_fuse_puts_chunks_found_by_both_lists_first():
    ids, scores = fuse([[0, 1], [1, 2]], 10)
    assert ids[0] == 1 and len(ids) == 3 and scores[0] > scores[1]


def test_fuse_ranks_a_missing_chunk_just_past_the_list():
    ids, scores = fuse([[7], []], 40)
    assert ids.tolist() == [7] and scores[0] == pytest.approx(1 / 60 + 1 / 100)


class Keyword:
    def __init__(self, scores):
        self.scores = np.array(scores, dtype=float)

    def get_scores(self, _):
        return self.scores


def searchable(monkeypatch, bm):
    r = Retriever.__new__(Retriever)
    r.chunks = [{"chunk_id": str(i), "paper_id": "a", "heading": "H", "section_role": "r", "text": "t"} for i in range(3)]
    r.emb = np.array([[1, 0], [0, 1], [0.6, 0.8]], dtype=np.float32)
    r.index, r.bm25 = NumpyIndex(r.emb), Keyword(bm)
    r.titles, r.pmc, r.years, r.meta = {"a": "A"}, {"a": ""}, {"a": 2014}, {"": {"authors": ["Smith A"], "journal": "Bone"}}
    monkeypatch.setattr(retrieval, "embed", lambda _: np.array([[1, 0]], dtype=np.float32))
    monkeypatch.setitem(retrieval.CFG["retrieval"], "candidate_k", 1)
    return r


def test_dense_search_uses_the_index(monkeypatch):
    assert searchable(monkeypatch, [0, 0, 0]).search("q", "dense", 1)[0]["chunk_id"] == "0"


def test_keyword_search_ranks_by_bm25(monkeypatch):
    assert searchable(monkeypatch, [0, 0, 9]).search("q", "bm25", 1)[0]["chunk_id"] == "2"


def test_hits_found_only_by_keyword_still_carry_their_dense_score(monkeypatch):
    hits = searchable(monkeypatch, [0, 9, 0]).search("q", "hybrid", 2)
    assert {h["chunk_id"]: h["dense"] for h in hits}["1"] == pytest.approx(0.0)


def test_unknown_mode_is_rejected(monkeypatch):
    with pytest.raises(ValueError):
        searchable(monkeypatch, [0, 0, 0]).search("q", "magic", 1)


def test_passages_and_vectors_out_of_step_stop_the_load(tmp_path, monkeypatch):
    (tmp_path / "chunks.jsonl").write_text(json.dumps({"text": "bone"}) + "\n" + json.dumps({"text": "muscle"}) + "\n", encoding="utf-8")
    (tmp_path / "corpus.jsonl").write_text("", encoding="utf-8")
    np.save(tmp_path / "chunk_emb.npy", np.zeros((1, 2), dtype=np.float32))
    monkeypatch.setattr(retrieval, "PROC", tmp_path)
    with pytest.raises(ValueError, match="chunk_emb"):
        Retriever()


def test_hits_carry_year_and_any_known_authors(monkeypatch):
    h = searchable(monkeypatch, [0, 0, 0]).search("q", "dense", 1)[0]
    assert (h["year"], h["authors"], h["journal"]) == (2014, ["Smith A"], "Bone")


def test_hits_work_without_metadata(monkeypatch):
    r = searchable(monkeypatch, [0, 0, 0])
    r.meta = {}
    assert r.search("q", "dense", 1)[0]["authors"] == []
