"""Key-sentence summaries: each paper's most central sentences, copied exactly. Run: python backend/summaries.py
Sentences are ranked by TF-IDF similarity to the document's centroid, so nothing is generated or rewritten."""
import json, re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from data import PROC, read_jsonl

KEY_SENTENCES = 2


def key_sentences(text, n=KEY_SENTENCES):
    parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if len(parts) <= n:
        return "  ".join(parts)
    m = TfidfVectorizer(stop_words="english").fit_transform(parts)
    scores = np.asarray(m @ np.asarray(m.mean(axis=0)).ravel()).ravel()
    keep = sorted(np.argsort(-scores)[:n])  # the original order reads better than score order
    return "  ".join(parts[i] for i in keep)


def source_text(doc):
    return doc["abstract"] or (doc["sections"][0]["text"] if doc["sections"] else "")


def add_summaries(papers, corpus):
    text = {d["paper_id"]: source_text(d) for d in corpus}
    return [{**p, "summary": key_sentences(text.get(p["paper_id"], ""))} for p in papers]


def main():
    path = PROC / "papers.json"
    papers = add_summaries(json.loads(path.read_text(encoding="utf-8")), read_jsonl(PROC / "corpus.jsonl"))
    path.write_text(json.dumps(papers), encoding="utf-8")
    print(sum(bool(p["summary"]) for p in papers), "of", len(papers), "papers have key sentences")


if __name__ == "__main__":
    main()
