"""Dictionary NER (baseline) + BioBERT (final: trained on rule labels, scored on hand-corrected gold).
Run: python backend/ner.py bootstrap   (writes data/eval/ner_gold.jsonl -> hand-correct spans)
     python backend/ner.py train | eval"""
import json, random, re, sys, functools
from collections import Counter
from data import CFG, ROOT, PROC, read_jsonl, write_jsonl

REP, MOD = ROOT / "reports", ROOT / "models"
GOLD = ROOT / CFG["paths"]["eval"] / "ner_gold.jsonl"
BIO = "dmis-lab/biobert-base-cased-v1.1"

LEX = {
    "organism": {"mouse": r"mice|mouse|murine",
                 "human": r"humans?|astronauts?|cosmonauts?|crew ?members?|volunteers|participants|patients",
                 "plant": r"plants?|arabidopsis|seedlings?|wheat|rice|lettuce",
                 "cells": r"cells?|cell lines?|fibroblasts?|osteoblasts?|osteoclasts?|lymphocytes|cultures?",
                 "other": r"rats?|drosophila|fruit flies|c\. elegans|nematodes?|zebrafish|yeast|bacteria|e\. coli|microb\w+"},
    "stressor": {"microgravity": r"microgravity|weightlessness|space ?flight|hindlimb unloading|clinostat|"
                                 r"rotating wall vessel|bed rest|parabolic flight|simulated microgravity",
                 "radiation": r"radiation|irradiat\w*|cosmic rays?|heavy ions?|protons?|gamma[- ]rays?|x-rays?|hze",
                 "other": r"isolation|confinement|hypergravity|hypoxia|hyperoxia|centrifug\w*|diet"},
    "tissue": {"bone": r"bones?|osteo\w*|trabecular|cortical|femur|tibia|vertebra\w*|skeleton",
               "muscle": r"muscles?|muscular|myo\w*|soleus|skeletal muscle",
               "immune": r"immune|immunity|t[- ]cells?|lymph\w*|thymus|spleen|leukocytes?|macrophages?",
               "cardiovascular": r"cardi\w*|heart|vascular|blood vessels?|arter\w+|endothel\w*",
               "brain": r"brain|neural|neuro\w*|hippocamp\w*|cortex",
               "eye": r"eyes?|retina\w*|ocular|optic|vision",
               "liver": r"liver|hepat\w*", "kidney": r"kidneys?|renal", "skin": r"skin|dermal|epiderm\w*",
               "gut": r"gut|intestin\w*|microbiome|colon",
               "plant tissue": r"roots?|shoots?|leaf|leaves|hypocotyl|cell wall"},
    "outcome": {"decrease": r"loss|decreas\w*|reduc\w*|declin\w*|atrophy|impair\w*|deficits?|downregulat\w*",
                "increase": r"increas\w*|elevat\w*|enhanc\w*|upregulat\w*|induc\w*",
                "damage": r"dna damage|oxidative stress|apoptosis|inflammation|fibrosis|mutations?",
                "expression": r"gene expression|transcript\w*|differentially expressed|proteom\w*|methylation",
                "growth": r"growth|development|differentiation|proliferation|morphology"},
}
RX = [(t, c, re.compile(r"\b(?:%s)\b" % p, re.I)) for t, d in LEX.items() for c, p in d.items()]
TYPES = list(LEX)
TAGS = ["O"] + [f"{p}-{t}" for t in TYPES for p in "BI"]


def extract(text):
    ms = sorted(((m.start(), m.end(), t, c) for t, c, rx in RX for m in rx.finditer(text)),
                key=lambda x: (-(x[1] - x[0]), x[0]))  # longest first, drop overlaps
    keep = []
    for s, e, t, c in ms:
        if all(e <= a or s >= b for a, b, _, _ in keep):
            keep.append((s, e, t, c))
    return [{"start": s, "end": e, "type": t, "cat": c, "text": text[s:e]} for s, e, t, c in sorted(keep)]


def spans(text):
    return [(x["start"], x["end"], x["type"]) for x in extract(text)]


def paper_entities(title, abstract, results):
    """{type: [cats]} - a category needs weight 3: one title mention, or repeated mentions in the abstract and
    results. One passing abstract mention ("Human expansion in space...") is not enough to tag a mouse study human."""
    w = Counter()
    for text, wt in ((title, 3), (abstract, 2), (results, 1)):
        for x in extract(text):
            w[(x["type"], x["cat"])] += wt
    return {t: sorted(c for (tt, c), v in w.items() if tt == t and v >= 3) for t in TYPES}


def tags(off, sp):
    out, prev = [], None
    for a, b in off:
        if a == b:
            out.append(-100); continue
        hit = next((k for k, (s, e, t) in enumerate(sp) if a < e and b > s), None)
        if hit is None:
            out.append(0); prev = None
        else:
            out.append(TAGS.index(("I-" if hit == prev else "B-") + sp[hit][2])); prev = hit
    return out


def decode(off, ids):
    out, cur = [], None
    for (a, b), t in zip(off, ids):
        if a == b:
            continue
        tg = TAGS[t]
        if tg == "O":
            cur = None; continue
        p, ty = tg.split("-")
        if p == "B" or cur is None or cur[2] != ty:
            cur = [a, b, ty]; out.append(cur)
        else:
            cur[1] = b
    return [tuple(x) for x in out]


def bootstrap(n=80):
    if GOLD.exists() and GOLD.stat().st_size:
        return print("gold file exists - hand-correct it instead")
    ch = [c["text"] for c in read_jsonl(PROC / "chunks.jsonl") if c["split"] == "test"]
    random.Random(CFG["seed"]).shuffle(ch)
    write_jsonl(GOLD, [{"text": t, "spans": [list(s) for s in spans(t)]} for t in ch[:n]])
    print("wrote", GOLD, "- fix the [start, end, type] spans by hand")


def train(n=3000, epochs=1):
    import torch
    from transformers import AutoTokenizer, AutoModelForTokenClassification, set_seed
    set_seed(CFG["seed"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    texts = [c["text"] for c in read_jsonl(PROC / "chunks.jsonl") if c["split"] == "train"]
    random.Random(CFG["seed"]).shuffle(texts)
    texts = texts[:n]
    tok = AutoTokenizer.from_pretrained(BIO)
    m = AutoModelForTokenClassification.from_pretrained(BIO, num_labels=len(TAGS)).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-5)
    for ep in range(epochs):
        random.Random(ep).shuffle(texts)
        for i in range(0, len(texts), 16):
            b = texts[i:i + 16]
            enc = tok(b, return_offsets_mapping=True, truncation=True, max_length=384, padding=True, return_tensors="pt")
            off = enc.pop("offset_mapping").tolist()
            labels = torch.tensor([tags(o, spans(t)) for o, t in zip(off, b)]).to(dev)
            m(**enc.to(dev), labels=labels).loss.backward(); opt.step(); opt.zero_grad()
        print("biobert epoch", ep, "done")
    m.save_pretrained(str(MOD / "biobert")); tok.save_pretrained(str(MOD / "biobert"))


@functools.lru_cache(maxsize=1)
def _bert():
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    return (AutoTokenizer.from_pretrained(str(MOD / "biobert")),
            AutoModelForTokenClassification.from_pretrained(str(MOD / "biobert")).eval())


def bert_spans(texts):
    import torch
    tok, m = _bert()
    out = []
    for t in texts:
        enc = tok(t, return_offsets_mapping=True, truncation=True, max_length=384, return_tensors="pt")
        off = enc.pop("offset_mapping")[0].tolist()
        with torch.no_grad():
            out.append(decode(off, m(**enc).logits[0].argmax(-1).tolist()))
    return out


def predict_ner(text):
    if (MOD / "biobert").exists():
        return [{"start": s, "end": e, "type": t, "text": text[s:e]} for s, e, t in bert_spans([text])[0]]
    return [{k: x[k] for k in ("start", "end", "type", "text")} for x in extract(text)]

 
def prf(pred, gold):
    tp = fp = fn = 0
    for p, g in zip(pred, gold):
        p, g = set(map(tuple, p)), set(map(tuple, g))
        tp += len(p & g); fp += len(p - g); fn += len(g - p)
    pr, rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"precision": pr, "recall": rc, "f1": 2 * pr * rc / max(pr + rc, 1e-9)}


def evaluate():
    g = read_jsonl(GOLD)
    texts, gold = [x["text"] for x in g], [x["spans"] for x in g]
    res = {"rules": prf([spans(t) for t in texts], gold)}
    if (MOD / "biobert").exists():
        res["biobert"] = prf(bert_spans(texts), gold)
    REP.mkdir(exist_ok=True)
    (REP / "ner.json").write_text(json.dumps(res, indent=1))
    print(res)


def review():
    rows = read_jsonl(GOLD)
    short = {"org": "organism", "str": "stressor", "tis": "tissue", "out": "outcome"}
    for r in rows:
        if r.get("done"):
            continue
        while True:
            sp = sorted(map(tuple, r["spans"]))
            print("\n" + r["text"] + "\n")
            for i, (s, e, t) in enumerate(sp):
                print(f"  {i}: [{t}] {r['text'][s:e]}")
            cmd = input("Enter=ok | d 1 2 = delete | a word org|str|tis|out = add | q = quit > ").split()
            if not cmd:
                r["spans"] = [list(x) for x in sp]; r["done"] = True; break
            if cmd[0] == "q":
                write_jsonl(GOLD, rows); return
            if cmd[0] == "d":
                drop = {int(x) for x in cmd[1:] if x.isdigit()}
                r["spans"] = [list(x) for i, x in enumerate(sp) if i not in drop]
            elif cmd[0] == "a" and len(cmd) >= 3 and cmd[-1] in short:
                w, t = " ".join(cmd[1:-1]), short[cmd[-1]]
                cur = [tuple(x) for x in sp]
                for m in re.finditer(r"\b" + re.escape(w) + r"\b", r["text"], re.I):
                    if all(m.end() <= a or m.start() >= b for a, b, _ in cur):
                        cur.append((m.start(), m.end(), t))
                r["spans"] = [list(x) for x in cur]
        write_jsonl(GOLD, rows)
    write_jsonl(GOLD, rows)
    print("all reviewed")


if __name__ == "__main__":
    {"bootstrap": bootstrap, "train": train, "eval": evaluate, "review": review}[sys.argv[1] if len(sys.argv) > 1 else "eval"]()