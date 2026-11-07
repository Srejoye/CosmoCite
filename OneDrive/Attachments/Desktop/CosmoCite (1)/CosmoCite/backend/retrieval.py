"""BM25, dense, hybrid (RRF), cross-encoder rerank + paper-level metrics. Run: python backend/retrieval.py"""
import functools, json
import numpy as np
from rank_bm25 import BM25Okapi
from data import CFG, ROOT, PROC, read_jsonl
from features import tokenize, embed
from index import best, load_index

MODES = ("bm25", "dense", "hybrid", "rerank")


@functools.lru_cache(maxsize=1)
def _cross_encoder():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(CFG["rerank_model"])


def paper_metrics(ranked, gold, k=10):
    """Paper-level metrics for one query. recall_capped divides by min(|gold|, k): plain recall@k cannot
    reach 1.0 when a question has more than k gold papers, so it understates the system."""
    gold, top = set(gold), list(ranked)[:k]
    hits = [p in gold for p in top]
    dcg = sum(h / np.log2(r + 2) for r, h in enumerate(hits))
    ideal = sum(1 / np.log2(r + 2) for r in range(min(len(gold), k)))
    return {f"recall@{k}": sum(hits) / len(gold), f"recall_capped@{k}": sum(hits) / min(len(gold), k),
            f"precision@{k}": sum(hits) / k, "mrr": next((1 / (r + 1) for r, h in enumerate(hits) if h), 0.0),
            f"ndcg@{k}": dcg / ideal}


def fuse(lists, depth):
    """Reciprocal rank fusion. A chunk missing from a list counts as ranked just past that list's end."""
    ranks = [{int(i): r for r, i in enumerate(ids)} for ids in lists]
    ids = np.array(sorted(set().union(*ranks)))
    scores = np.array([sum(1 / (60 + r.get(int(i), depth)) for r in ranks) for i in ids])
    order = np.argsort(-scores, kind="stable")
    return ids[order], scores[order]


class Retriever:
    def __init__(self):
        self.chunks = read_jsonl(PROC / "chunks.jsonl")
        self.emb = np.load(PROC / "chunk_emb.npy", mmap_mode="r")
        if len(self.chunks) != len(self.emb):
            raise ValueError(f"chunks.jsonl has {len(self.chunks)} passages but chunk_emb.npy has {len(self.emb)} vectors")
        self.index = load_index(CFG["retrieval"]["index"], self.emb, PROC / "chunk.faiss")
        corpus = read_jsonl(PROC / "corpus.jsonl")
        self.titles = {d["paper_id"]: d["title"] for d in corpus}
        self.pmc = {d["paper_id"]: d["pmc_id"] for d in corpus}
        self.years = {d["paper_id"]: d["year"] for d in corpus}
        meta = PROC / "metadata.json"
        self.meta = json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else {}
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])

    def _ranked(self, q, mode, k):
        """-> (chunk indices, {index: score}, {index: rerank logit}, query vector)"""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode}")
        qv, depth = embed([q])[0], max(k, CFG["retrieval"]["candidate_k"])
        if mode == "dense":
            order, s = self.index.top(qv, depth)
        else:
            bm = self.bm25.get_scores(tokenize(q))
            keyword = best(bm, depth)
            order, s = (keyword, bm[keyword]) if mode == "bm25" else fuse([keyword, self.index.top(qv, depth)[0]], depth)
        score, ce = dict(zip(order.tolist(), s.tolist())), {}
        if mode == "rerank":  # re-score the fused pool; anything past the pool keeps its fused order
            pool = order[:CFG["retrieval"]["pool_k"]]
            logits = _cross_encoder().predict([(q, self.chunks[i]["text"]) for i in pool], show_progress_bar=False)
            ce = {int(i): float(l) for i, l in zip(pool, logits)}
            order = np.concatenate([pool[np.argsort(-np.asarray(logits))], order[len(pool):]])
        return order[:k], score, ce, qv

    def search(self, q, mode="hybrid", k=None):
        k = k or CFG["retrieval"]["top_k"]
        idx, score, ce, qv = self._ranked(q, mode, k)
        out = []
        for i, dense in zip(idx, np.asarray(self.emb[idx]) @ qv):
            c = self.chunks[i]
            pid = c["paper_id"]
            about = self.meta.get(self.pmc[pid], {})
            out.append({"chunk_id": c["chunk_id"], "paper_id": pid, "pmc_id": self.pmc[pid], "title": self.titles[pid],
                        "year": self.years.get(pid), "authors": about.get("authors", []), "journal": about.get("journal", ""),
                        "section": c["heading"] or c["section_role"], "text": c["text"],
                        "score": ce.get(int(i), score[int(i)]), "dense": float(dense), "rerank": ce.get(int(i))})
        return out

    @functools.cached_property
    def paper_vectors(self):
        """One unit vector per paper: the mean of its passages."""
        rows = {}
        for i, c in enumerate(self.chunks):
            rows.setdefault(c["paper_id"], []).append(i)
        ids = list(rows)
        m = np.stack([np.asarray(self.emb[rows[p]]).mean(axis=0) for p in ids])
        return ids, m / np.linalg.norm(m, axis=1, keepdims=True)

    def related(self, paper_id, k=5):
        ids, m = self.paper_vectors
        scores = m @ m[ids.index(paper_id) if paper_id in ids else self._unknown(paper_id)]
        picks = [ids[i] for i in np.argsort(-scores) if ids[i] != paper_id][:k]
        return [{"paper_id": p, "title": self.titles[p], "pmc_id": self.pmc[p]} for p in picks]

    @staticmethod
    def _unknown(paper_id):
        raise KeyError(paper_id)

    def ranked_papers(self, q, mode="hybrid", depth=100):
        seen = []
        for h in self.search(q, mode, depth):
            if h["paper_id"] not in seen:
                seen.append(h["paper_id"])
        return seen

    def per_query(self, questions, mode="hybrid", k=10):
        return [paper_metrics(self.ranked_papers(q["question"], mode), q["gold_paper_ids"], k) for q in questions]

    def evaluate(self, questions, mode="hybrid", k=10):
        """questions: [{qid, question, gold_paper_ids}] -> mean paper-level metrics."""
        rows = self.per_query(questions, mode, k)
        return {"mode": mode, **{m: float(np.mean([r[m] for r in rows])) for m in rows[0]}}


if __name__ == "__main__":
    qs = json.loads((ROOT / CFG["paths"]["eval"] / "questions.json").read_text())
    r = Retriever()
    for m in MODES:
        print(r.evaluate(qs, m))
