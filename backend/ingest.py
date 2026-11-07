"""Adds papers to the corpus without rebuilding it. Run: python backend/ingest.py <page.html or folder> ...
Restart the server afterwards. Existing papers keep their ids and splits, so evaluation stays leak-free.
New papers have no map position until python backend/features.py runs, and python backend/data.py rebuilds from
data/raw_html alone, so it drops anything that was only ingested."""
import json, os, sys
from pathlib import Path
import numpy as np
from build import gaps_of, paper_row
from data import PROC, assign_split, chunks_of, parse, read_jsonl, write_jsonl
from features import embed
from summaries import key_sentences, source_text


def next_paper_id(existing):
    return f"{max(map(int, existing), default=0) + 1:04d}"


def identity(doc):
    return doc["pmc_id"] or doc["title"]


def new_docs(paths, corpus):
    """-> (papers worth adding, number skipped). Duplicates and pages with no article are skipped."""
    ids, known, docs = {d["paper_id"] for d in corpus}, {identity(d) for d in corpus}, []
    for path in paths:
        doc = parse(Path(path), next_paper_id(ids))
        if doc["title"] and (doc["abstract"] or doc["sections"]) and identity(doc) not in known:
            ids.add(doc["paper_id"])
            known.add(identity(doc))
            docs.append(doc)
    return docs, len(paths) - len(docs)


def ingest_files(paths, proc=PROC):
    corpus = read_jsonl(proc / "corpus.jsonl")
    docs, skipped = new_docs(paths, corpus)
    if not docs:
        return {"added": 0, "skipped": skipped, "chunks": 0}
    rows = [c for d in docs for c in chunks_of(d, assign_split(d["paper_id"]))]
    emb = np.concatenate([np.load(proc / "chunk_emb.npy"), embed([c["text"] for c in rows])])
    papers = json.loads((proc / "papers.json").read_text(encoding="utf-8"))
    papers += [paper_row(d, key_sentences(source_text(d)), None) for d in docs]

    # Everything is written beside the real files first, so a failure here leaves the corpus as it was.
    staged = {name: proc / (name + ".tmp") for name in ("corpus.jsonl", "chunks.jsonl", "chunk_emb.npy", "papers.json", "gaps.json")}
    write_jsonl(staged["corpus.jsonl"], corpus + docs)
    write_jsonl(staged["chunks.jsonl"], read_jsonl(proc / "chunks.jsonl") + rows)
    with open(staged["chunk_emb.npy"], "wb") as f:
        np.save(f, emb)
    staged["papers.json"].write_text(json.dumps(papers), encoding="utf-8")
    staged["gaps.json"].write_text(json.dumps(gaps_of(papers)), encoding="utf-8")
    for name, tmp in staged.items():
        os.replace(tmp, proc / name)
    return {"added": len(docs), "skipped": skipped, "chunks": len(rows)}


def main():
    paths = []
    for arg in map(Path, sys.argv[1:]):
        paths += sorted(arg.glob("*.html")) if arg.is_dir() else [arg]
    print(ingest_files(paths))


if __name__ == "__main__":
    main()
