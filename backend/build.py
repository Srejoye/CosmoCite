"""Build papers.json (summary, entities, t-SNE) + gaps.json. Run: python backend/build.py [--no-summary]
Needs: features.py (tsne.json). Summaries are cached/resumable in data/processed/summaries.json."""
import json, sys
from data import PROC, read_jsonl
from ner import paper_entities

ORGS, STRESS = ["mouse", "human", "plant", "cells"], ["microgravity", "radiation", "other"]


def paper_row(doc, summary, tsne):
    results = " ".join(s["text"] for s in doc["sections"] if s["role"] == "results")
    return {"paper_id": doc["paper_id"], "pmc_id": doc["pmc_id"], "title": doc["title"], "year": doc["year"], "summary": summary,
            "entities": paper_entities(doc["title"], doc["abstract"], results), "tsne": tsne}


def gaps_of(papers):
    gaps = []
    for o in ORGS:
        for s in STRESS:
            ids = [p["paper_id"] for p in papers if o in p["entities"]["organism"] and s in p["entities"]["stressor"]]
            gaps.append({"organism": o, "stressor": s, "count": len(ids), "paper_ids": ids})
    return gaps


def main():
    skip = "--no-summary" in sys.argv
    if not skip:
        from seq2seq import summ_source, summarize
    corpus = read_jsonl(PROC / "corpus.jsonl")
    tsne = json.loads((PROC / "tsne.json").read_text())
    cp = PROC / "summaries.json"
    cache = json.loads(cp.read_text()) if cp.exists() else {}
    papers = []
    for n, d in enumerate(corpus):
        pid = d["paper_id"]
        if not skip and pid not in cache:
            cache[pid] = summarize(summ_source(d)[:6000])
            if n % 20 == 0:
                cp.write_text(json.dumps(cache)); print(n, "/", len(corpus))
        papers.append(paper_row(d, cache.get(pid, ""), tsne.get(pid)))
    cp.write_text(json.dumps(cache))
    gaps = gaps_of(papers)
    (PROC / "papers.json").write_text(json.dumps(papers))
    (PROC / "gaps.json").write_text(json.dumps(gaps))
    print(len(papers), "papers;", sum(1 for g in gaps if g["count"] == 0), "empty gap cells")


if __name__ == "__main__":
    main()