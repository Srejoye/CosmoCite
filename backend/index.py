"""Dense vector search behind one interface, so the corpus can outgrow exact search. reports/scale.json shows when."""
from pathlib import Path
import numpy as np


def best(scores, k):
    """Indices of the k highest scores, best first."""
    k = min(k, len(scores))
    ids = np.argpartition(-scores, k - 1)[:k]
    return ids[np.argsort(-scores[ids], kind="stable")]


class NumpyIndex:
    """Exact search. It reads every vector on every query."""

    def __init__(self, emb):
        self.emb = emb

    def top(self, q, k):
        scores = self.emb @ q
        ids = best(scores, k)
        return ids, scores[ids]


class FaissIndex:
    """Approximate HNSW search. A saved index is reused only while it still covers every vector."""

    def __init__(self, emb, path=None):
        import faiss
        if path and Path(path).exists():
            self.index = faiss.read_index(str(path))
            if self.index.ntotal == len(emb):
                return
        self.index = faiss.IndexHNSWFlat(emb.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
        self.index.add(np.ascontiguousarray(emb, dtype=np.float32))
        if path:
            faiss.write_index(self.index, str(path))

    def top(self, q, k):
        k = min(k, self.index.ntotal)
        self.index.hnsw.efSearch = max(64, k)  # HNSW cannot return more results than it explores
        scores, ids = self.index.search(np.asarray(q, dtype=np.float32)[None], k)
        found = ids[0] >= 0
        return ids[0][found], scores[0][found]


def load_index(kind, emb, path=None):
    if kind == "numpy":
        return NumpyIndex(emb)
    if kind == "faiss":
        return FaissIndex(emb, path)
    raise ValueError(f"index must be numpy or faiss, got {kind}")
