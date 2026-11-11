"""All evaluations -> reports/*.json.
Run: python backend/evaluate.py [questions|retrieval|pool|ablation|w2v|answers|calibrate|abstention|faith_sample|faithfulness|gaps|heldout|heldout_report]
(default: all). 'questions' writes the SILVER set: its gold ids are the rule tagger's output, so retrieval scores on it
measure agreement with the tagger. retrieval_known and retrieval_judged do not depend on the tagger."""
import json, random, re, sys
from collections import Counter
import numpy as np
from data import CFG, ROOT, PROC, read_jsonl, write_jsonl
from features import tokenize

REP = ROOT / "reports"
EVAL = ROOT / CFG["paths"]["eval"]
QF = EVAL / "questions.json"
ABST = EVAL / "abstention.json"       # answerable (source paper known) + unanswerable (key term absent from corpus)
QRELS = EVAL / "qrels.json"           # pooled relevance judgments
FAITH = EVAL / "faithfulness.jsonl"   # sentence-level support labels
HELD = EVAL / "heldout.jsonl"         # answer sentences for questions written after the sentence filter was frozen
HELD_AI = EVAL / "heldout_ai.json"    # a model's on-topic labels for the same sentences, kept out of the labeling sheet
ORG_P = {"mouse": "mice", "human": "humans", "plant": "plants", "cells": "cultured cells"}
STR_P = {"microgravity": "microgravity", "radiation": "radiation exposure"}


def dump(name, obj):
    REP.mkdir(exist_ok=True)
    (REP / f"{name}.json").write_text(json.dumps(obj, indent=1), encoding="utf-8")
    print(name, json.dumps(obj)[:400])


def bootstrap_ci(values, n=2000, seed=None):
    """95% percentile interval of the mean."""
    v = np.asarray(values, dtype=float)
    means = np.random.default_rng(CFG["seed"] if seed is None else seed).choice(v, (n, len(v))).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def questions():
    if QF.exists() and QF.stat().st_size:
        return print("questions.json exists - keeping it")
    papers = json.loads((PROC / "papers.json").read_text())
    tissues = sorted({t for p in papers for t in p["entities"]["tissue"]})
    cand = []
    for o in ORG_P:
        for s in STR_P:
            for t in tissues:
                gold = [p["paper_id"] for p in papers if o in p["entities"]["organism"]
                        and s in p["entities"]["stressor"] and t in p["entities"]["tissue"]]
                if 3 <= len(gold) <= 40:
                    cand.append({"question": f"How does {STR_P[s]} affect {t} in {ORG_P[o]}?", "gold_paper_ids": gold})
    random.Random(CFG["seed"]).shuffle(cand)
    qs = [{"qid": f"q{i:03d}", **c} for i, c in enumerate(cand[:60])]
    QF.write_text(json.dumps(qs, indent=1))
    print(len(qs), "questions written (silver - verify gold ids by hand)")


def load_q():
    return json.loads(QF.read_text(encoding="utf-8"))


def load_abst():
    return json.loads(ABST.read_text(encoding="utf-8"))


def retrieval():
    from retrieval import Retriever, MODES
    r, qs = Retriever(), load_q()
    dump("retrieval", [r.evaluate(qs, m) for m in MODES])
    known = [{"question": q["question"], "gold_paper_ids": [q["source_paper"]]} for q in load_abst() if q["answerable"]]
    dump("retrieval_known", [{k: v for k, v in r.evaluate(known, m).items() if k in ("mode", "recall@10", "mrr")}
                             for m in MODES])
    a, b = (np.array([x["mrr"] for x in r.per_query(known, m)]) for m in ("rerank", "bm25"))
    dump("retrieval_known_delta", {"metric": "mrr", "a": "rerank", "b": "bm25", "n": len(known),
                                   "mean_delta": float((a - b).mean()), "ci95": list(bootstrap_ci(a - b))})
    if QRELS.exists():  # judged to pool depth 5, so scored at k=5; unjudged papers count as not relevant
        qrels = [q for q in json.loads(QRELS.read_text(encoding="utf-8")) if q["relevant"]]
        judged = [{"question": q["question"], "gold_paper_ids": q["relevant"]} for q in qrels]
        dump("retrieval_judged", [{k: v for k, v in r.evaluate(judged, m, k=5).items()
                                   if k in ("mode", "precision@5", "mrr", "ndcg@5")} for m in MODES])
        a, b = (np.array([x["ndcg@5"] for x in r.per_query(judged, m, 5)]) for m in ("dense", "rerank"))
        dump("retrieval_delta", {"metric": "ndcg@5", "a": "dense", "b": "rerank", "n": len(judged),
                                 "mean_delta": float((a - b).mean()), "ci95": list(bootstrap_ci(a - b)),
                                 "label_sources": dict(Counter(q.get("label_source", "unknown") for q in qrels))})


def merge_pool(old, new):
    """Questions already in the pool keep their entry, so finished judgments stay valid."""
    seen = {q["qid"] for q in old}
    return old + [q for q in new if q["qid"] not in seen]


def pool(n=25, depth=5):
    """Candidates for relevance judging: the union of each mode's top papers, grown to n questions."""
    from retrieval import Retriever, MODES
    path = EVAL / "qrels_pool.json"
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    have = {q["qid"] for q in old}
    rest = [q for q in load_q() if q["qid"] not in have]
    r, new = Retriever(), []
    for q in random.Random(CFG["seed"]).sample(rest, max(n - len(old), 0)):
        ids = sorted({p for m in MODES for p in r.ranked_papers(q["question"], m)[:depth]})
        new.append({"qid": q["qid"], "question": q["question"], "pool": [{"paper_id": i, "title": r.titles[i]} for i in ids]})
    path.write_text(json.dumps(merge_pool(old, new), indent=1, ensure_ascii=False), encoding="utf-8")
    print(sum(len(o["pool"]) for o in new), "new papers to judge")


def ablation():  # Unit I: preprocessing ablation with BM25
    from rank_bm25 import BM25Okapi
    qs, ch, out = load_q(), read_jsonl(PROC / "chunks.jsonl"), []
    for stop in (False, True):
        for stem in (False, True):
            bm = BM25Okapi([tokenize(c["text"], stop, stem) for c in ch])
            rec, mrr = [], []
            for q in qs:
                sc, seen = bm.get_scores(tokenize(q["question"], stop, stem)), []
                for i in np.argsort(-sc):
                    if ch[i]["paper_id"] not in seen:
                        seen.append(ch[i]["paper_id"])
                    if len(seen) == 10:
                        break
                gold = set(q["gold_paper_ids"]); hits = [p in gold for p in seen]
                rec.append(sum(hits) / len(gold))
                mrr.append(next((1 / (k + 1) for k, h in enumerate(hits) if h), 0.0))
            out.append({"stopwords_removed": stop, "stemmed": stem,
                        "recall@10": float(np.mean(rec)), "mrr": float(np.mean(mrr))})
    dump("ablation", out)


def w2v():  # Unit II: neighbours + t-SNE of frequent words
    from gensim.models import Word2Vec
    from sklearn.manifold import TSNE
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    wv = Word2Vec.load(str(PROC / "w2v.model")).wv
    nn_ = {w: [x for x, _ in wv.most_similar(w, topn=5)] for w in
           ("microgravity", "radiation", "bone", "muscle", "mice", "plants") if w in wv}
    words = wv.index_to_key[:80]
    xy = TSNE(n_components=2, perplexity=15, init="pca", random_state=CFG["seed"]).fit_transform(wv[words])
    REP.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(xy[:, 0], xy[:, 1], s=6)
    for (x, y), w in zip(xy, words):
        ax.annotate(w, (x, y), fontsize=7)
    fig.tight_layout(); fig.savefig(REP / "w2v_tsne.png", dpi=120); plt.close(fig)
    dump("w2v", nn_)


def answers(n=None, only=None):
    """Every flan decoding strategy, plus the local multi-passage engine. only= reruns one and keeps the others."""
    from retrieval import Retriever
    from generate import answer, DECODING
    r, qs = Retriever(), load_q()[:n]
    REP.mkdir(exist_ok=True)
    res = json.loads((REP / "generation.json").read_text(encoding="utf-8")) if only else {}
    runs = [x for x in [(d, d, "flan") for d in DECODING] + [("synth", "greedy", "synth"), ("rank", "greedy", "rank")] if not only or x[0] == only]
    kept = [x for x in read_jsonl(REP / "answers_sample.jsonl") if x["decoding"] != only] if only else []
    with open(REP / "answers_sample.jsonl", "w", encoding="utf-8") as sample:
        for x in kept:
            sample.write(json.dumps(x, ensure_ascii=False) + "\n")
        for name, d, engine in runs:
            gate = filt = gold_hit = fallback = multi = 0; words, cites = [], []
            for q in qs:
                o = answer(q["question"], d, r, engine=engine)
                fallback += o["engine"] != engine
                if not o["sufficient"]:
                    gate += o["reason"] == "gate"; filt += o["reason"] == "filter"; continue
                papers = {c["paper_id"] for c in o["citations"]}
                gold_hit += bool(papers & set(q["gold_paper_ids"])); multi += len(papers) > 1
                words.append(len(re.sub(r"\[\d+\]", "", o["answer"]).split())); cites.append(len(o["citations"]))
                sample.write(json.dumps({"decoding": name, "q": q["question"], "answer": o["answer"],
                                         "citations": o["citations"]}, ensure_ascii=False) + "\n")
            k = max(len(qs) - gate - filt, 1)
            res[name] = {"abstention_rate": (gate + filt) / len(qs), "refused_by_gate": gate, "refused_by_filter": filt,
                         "fallbacks": fallback, "mean_answer_words": float(np.mean(words)) if words else 0.0,
                         "mean_citations": float(np.mean(cites)) if cites else 0.0, "multi_paper_rate": multi / k,
                         "cited_silver_paper": gold_hit / k}
    dump("generation", res)


def faith_rows(samples, n, seed, engine="rank"):
    """Sentence-level rows for a person to label, from one engine's answers."""
    from generate import parse_claims
    rows = []
    for s in samples:
        if s["decoding"] != engine:
            continue
        cites = s["citations"]
        for i, sentence in parse_claims(s["answer"], len(cites)):
            rows.append({"q": s["q"], "decoding": s["decoding"], "sentence": sentence, "chunk_id": cites[i]["chunk_id"],
                         "quote": cites[i]["quote"], "supported": None, "on_topic": None})
    random.Random(seed).shuffle(rows)
    return rows[:n]


def faith_sample(n=60):
    from labels import FAITH_CSV, has_answers
    if has_answers(FAITH_CSV, "supported", "on_topic"):
        return print("the labeling sheet has answers - import them before resampling")
    if FAITH.exists() and any(x["supported"] is not None for x in read_jsonl(FAITH)):
        return print("faithfulness.jsonl already has labels - keeping it")
    rows = faith_rows(read_jsonl(REP / "answers_sample.jsonl"), n, CFG["seed"])
    text = {c["chunk_id"]: c["text"] for c in read_jsonl(PROC / "chunks.jsonl")}
    write_jsonl(FAITH, [{**x, "passage": text[x["chunk_id"]]} for x in rows])
    print(len(rows), "sentences to label in", FAITH)


def faithfulness():
    """Share of answer sentences supported by the passage they cite. Labels live in data/eval/faithfulness.jsonl."""
    rows = [x for x in read_jsonl(FAITH) if x["supported"] is not None]
    if not rows:
        return print("no labeled sentences yet")

    def summary(g):
        vals = [float(x["supported"]) for x in g]
        topic = [float(x["on_topic"]) for x in g if x.get("on_topic") is not None]
        return {"sentences": len(g), "supported": float(np.mean(vals)), "ci95": list(bootstrap_ci(vals)),
                "on_topic": float(np.mean(topic)) if topic else None}

    out = {d: summary([x for x in rows if x["decoding"] == d]) for d in sorted({x["decoding"] for x in rows})}
    out["all"] = summary(rows)
    dump("faithfulness", out)


def kappa(a, b):
    """Cohen's kappa for two yes/no label lists: agreement beyond what chance alone would give."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    seen, chance = float((a == b).mean()), float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    return 1.0 if chance == 1 else (seen - chance) / (1 - chance)


def rate(labels):
    return {"sentences": len(labels), "on_topic": float(np.mean(labels)), "ci95": list(bootstrap_ci([float(x) for x in labels]))}


def heldout_summary(rows, ai):
    """rows: labeling rows, on_topic filled by a person or None. ai: [{q, sentence, on_topic}] from a model."""
    model = {(x["q"], x["sentence"]): x["on_topic"] for x in ai}
    person = [x for x in rows if x["on_topic"] is not None]
    both = [(x["on_topic"], model[x["q"], x["sentence"]]) for x in person if (x["q"], x["sentence"]) in model]
    return {"human": rate([x["on_topic"] for x in person]) if person else None,
            "model": rate([model[k] for k in ((x["q"], x["sentence"]) for x in rows) if k in model]) if model else None,
            "agreement": {"sentences": len(both), "same_label": float(np.mean([h == m for h, m in both])),
                          "kappa": kappa(*zip(*both))} if both else None}


def heldout():
    """Answers questions the sentence filter was never developed on, and lists every answer sentence for a person to label."""
    from labels import HELD_CSV, has_answers
    from retrieval import Retriever
    from generate import answer
    if has_answers(HELD_CSV, "supported", "on_topic") or (HELD.exists() and any(x["on_topic"] is not None for x in read_jsonl(HELD))):
        return print("the held-out sentences already have labels - keeping them")
    r, qs = Retriever(), json.loads((EVAL / "heldout.json").read_text(encoding="utf-8"))
    runs = [answer(q, "greedy", r, engine="rank") for q in qs]
    samples = [{"decoding": "rank", "q": q, "answer": o["answer"], "citations": o["citations"]} for q, o in zip(qs, runs) if o["sufficient"]]
    text = {c["chunk_id"]: c["text"] for c in read_jsonl(PROC / "chunks.jsonl")}
    rows = [{**x, "passage": text[x["chunk_id"]]} for x in faith_rows(samples, None, CFG["seed"])]
    write_jsonl(HELD, rows)
    dump("heldout", {"questions": len(qs), "answered": len(samples), "refused_by_gate": sum(o["reason"] == "gate" for o in runs),
                     "refused_by_filter": sum(o["reason"] == "filter" for o in runs), "sentences": len(rows)})


def heldout_report():
    """On-topic rate on the held-out sentences, from a person's labels and a model's, and how often the two agree."""
    path = REP / "heldout.json"
    ai = json.loads(HELD_AI.read_text(encoding="utf-8")) if HELD_AI.exists() else {"label_source": None, "labels": []}
    dump("heldout", {**json.loads(path.read_text(encoding="utf-8")), "model_label_source": ai["label_source"],
                     **heldout_summary(read_jsonl(HELD), ai["labels"])})


def gaps():  # sample for hand-check of the gap matrix
    papers = json.loads((PROC / "papers.json").read_text())
    abst = {d["paper_id"]: d["abstract"] for d in read_jsonl(PROC / "corpus.jsonl")}
    pick = random.Random(CFG["seed"]).sample(papers, 8)
    dump("gap_check", [{"paper_id": p["paper_id"], "title": p["title"], "entities": p["entities"],
                        "abstract": abst[p["paper_id"]][:600]} for p in pick])


OFFTOPIC = ["What is the capital of France?", "Who won the 2018 football world cup?", "How do I bake sourdough bread?",
            "What is the price of Bitcoin?", "Explain how a car engine works.", "Who wrote Hamlet?",
            "What is the best programming language?", "How tall is Mount Everest?",
            "What is the plot of the movie Titanic?", "How do airplanes fly?"]


def gate_signals(r, q):
    hits = r.search(q, "rerank", CFG["retrieval"]["top_k"])
    return {"rerank": max(h["rerank"] for h in hits), "dense": max(h["dense"] for h in hits)}


def balanced_accuracy(rows, signal, t):
    pos = [x for x in rows if x["answerable"]]
    neg = [x for x in rows if not x["answerable"]]
    return (sum(x[signal] >= t for x in pos) / len(pos) + sum(x[signal] < t for x in neg) / len(neg)) / 2


def calibrate():
    """Sweep each gate signal on the dev split only; the test split stays untouched until `abstention`."""
    from retrieval import Retriever
    from sklearn.metrics import roc_auc_score
    r = Retriever()
    rows = [{**q, **gate_signals(r, q["question"])} for q in load_abst()]
    dev = [x for x in rows if x["split"] == "dev"]
    out = {}
    for sig in ("rerank", "dense"):
        cuts = sorted({x[sig] for x in dev})
        best = max(cuts, key=lambda t: balanced_accuracy(dev, sig, t))
        out[sig] = {"dev_auc": roc_auc_score([x["answerable"] for x in dev], [x[sig] for x in dev]),
                    "best_threshold": float(best), "dev_balanced_accuracy": balanced_accuracy(dev, sig, best)}
    out["scores"] = [{k: x[k] for k in ("question", "answerable", "split", "rerank", "dense")} for x in dev]
    dump("abstention_calibration", out)


def natural():
    """Everyday questions. Templated questions hid regressions that these exposed, so they stay as a regression check."""
    from retrieval import Retriever
    from generate import answer
    r = Retriever()
    qs = json.loads((ROOT / CFG["paths"]["eval"] / "natural.json").read_text(encoding="utf-8"))
    runs = [answer(q, "greedy", r, engine="rank") for q in qs]
    dump("natural", {"questions": len(qs), "answered": sum(o["sufficient"] for o in runs), "refused_by_gate": sum(o["reason"] == "gate" for o in runs),
                     "refused_by_filter": sum(o["reason"] == "filter" for o in runs),
                     "multi_paper_rate": float(np.mean([len({c["paper_id"] for c in o["citations"]}) > 1 for o in runs if o["sufficient"]]))})


def abstention(engine="rank"):
    """End-to-end behaviour at the configured threshold, on questions the threshold was not tuned on."""
    from retrieval import Retriever
    from generate import answer
    r = Retriever()
    rows = []
    off = sum(not answer(q, "greedy", r, engine=engine)["sufficient"] for q in OFFTOPIC)
    rows.append({"set": "off-topic", "questions": len(OFFTOPIC), "correct": off / len(OFFTOPIC)})
    for split in ("dev", "test"):
        for flag, name in ((False, "unanswerable (should refuse)"), (True, "answerable (should answer)")):
            qs = [q for q in load_abst() if q["split"] == split and q["answerable"] is flag]
            ok = sum(answer(q["question"], "greedy", r, engine=engine)["sufficient"] is flag for q in qs)
            rows.append({"set": f"{split}: {name}", "questions": len(qs), "correct": ok / len(qs)})
    dump("abstention", rows)


if __name__ == "__main__":
    for a in sys.argv[1:] or ["questions", "retrieval", "ablation", "w2v", "answers", "calibrate", "abstention",
                              "faithfulness", "gaps"]:
        {"questions": questions, "retrieval": retrieval, "pool": pool, "ablation": ablation, "w2v": w2v,
         "answers": answers, "answers_synth": lambda: answers(only="synth"), "answers_rank": lambda: answers(only="rank"),
 "calibrate": calibrate, "abstention": abstention, "natural": natural, "faith_sample": faith_sample,
         "faithfulness": faithfulness,
         "gaps": gaps, "heldout": heldout, "heldout_report": heldout_report}[a]()