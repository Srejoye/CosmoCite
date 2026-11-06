"""LSTM language model, attention seq2seq (titles), T5 titles + summaries.
Run: python backend/seq2seq.py [lm|lstm|t5|summ]   (default: lm lstm)"""
import functools, json, math, random, re, sys
from collections import Counter
import numpy as np, torch, torch.nn as nn
from data import CFG, ROOT, PROC, read_jsonl
from features import tokenize

REP, MOD = ROOT / "reports", ROOT / "models"
PAD, UNK, BOS, EOS = 0, 1, 2, 3
TITLE = "Write a title for this abstract: "
SUMM = "Summarize the key findings in 3 sentences: "


def save(key, val):
    REP.mkdir(exist_ok=True)
    p = REP / "seq2seq.json"
    d = json.loads(p.read_text()) if p.exists() else {}
    d[key] = val
    p.write_text(json.dumps(d, indent=1))


def pairs():
    sp = {c["paper_id"]: c["split"] for c in read_jsonl(PROC / "chunks.jsonl")}
    out = {"train": [], "val": [], "test": []}
    for d in read_jsonl(PROC / "corpus.jsonl"):
        if d["abstract"] and d["title"] and d["paper_id"] in sp:
            out[sp[d["paper_id"]]].append((d["abstract"], d["title"]))
    return out


def norm(s):
    return " ".join(tokenize(s, stop=False))


def scores(preds, refs):
    from rouge_score import rouge_scorer
    import sacrebleu
    sc = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"])
    r = [sc.score(norm(a), norm(b)) for a, b in zip(refs, preds)]
    out = {k: float(np.mean([x[k].fmeasure for x in r])) for k in ("rouge1", "rouge2", "rougeL")}
    out["bleu"] = sacrebleu.corpus_bleu([norm(p) for p in preds], [[norm(x) for x in refs]]).score
    return out


# ---------- LSTM language model (perplexity) ----------
class LM(nn.Module):
    def __init__(s, V):
        super().__init__()
        s.e, s.r, s.o, s.d = nn.Embedding(V, 128), nn.LSTM(128, 256, batch_first=True), nn.Linear(256, V), nn.Dropout(0.3)

    def forward(s, x, h=None):
        y, h = s.r(s.d(s.e(x)), h)
        return s.o(s.d(y)), h


def run_lm(epochs=6, L=35, B=32):
    torch.manual_seed(CFG["seed"])
    ch = read_jsonl(PROC / "chunks.jsonl")
    toks = {s: [t for c in ch if c["split"] == s for t in tokenize(c["text"], stop=False) + ["<eos>"]]
            for s in ("train", "val", "test")}
    vocab = {w: i + 1 for i, (w, _) in enumerate(Counter(toks["train"]).most_common(8000))}  # 0 = unk

    def batchify(t):
        x = torch.tensor([vocab.get(w, 0) for w in t])
        n = len(x) // B
        return x[:n * B].view(B, n)

    data = {s: batchify(t) for s, t in toks.items()}
    m = LM(len(vocab) + 1)
    opt = torch.optim.Adam(m.parameters(), 2e-3)
    ce = nn.functional.cross_entropy

    def ppl(d):
        m.eval(); tot = n = 0; h = None
        with torch.no_grad():
            for i in range(0, d.size(1) - 1, L):
                y = d[:, i + 1:i + 1 + L]; x = d[:, i:i + L][:, :y.size(1)]
                lg, h = m(x, h)
                tot += ce(lg.reshape(-1, lg.size(-1)), y.reshape(-1), reduction="sum").item(); n += y.numel()
        return math.exp(tot / n)

    for ep in range(epochs):
        m.train(); h = None; d = data["train"]
        for i in range(0, d.size(1) - 1, L):
            y = d[:, i + 1:i + 1 + L]; x = d[:, i:i + L][:, :y.size(1)]
            if h: h = tuple(t.detach() for t in h)
            lg, h = m(x, h)
            loss = ce(lg.reshape(-1, lg.size(-1)), y.reshape(-1))
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        print("lm epoch", ep, "val ppl", round(ppl(data["val"]), 1))
    save("lm_perplexity", {"val": ppl(data["val"]), "test": ppl(data["test"])})


# ---------- attention encoder-decoder (Bahdanau / Luong) ----------
def pad(s):
    m = max(map(len, s))
    return torch.tensor([x + [0] * (m - len(x)) for x in s])


def enc_src(texts, v):
    return pad([[v.get(t, UNK) for t in tokenize(x, stop=False)[:250]] or [UNK] for x in texts])


def enc_tgt(texts, v):
    return pad([[BOS] + [v.get(t, UNK) for t in tokenize(x, stop=False)[:24]] + [EOS] for x in texts])


class S2S(nn.Module):
    def __init__(s, V, att="bahdanau", E=128, H=128):
        super().__init__()
        D = 2 * H; s.att = att
        s.emb, s.enc = nn.Embedding(V, E, padding_idx=0), nn.LSTM(E, H, batch_first=True, bidirectional=True)
        s.init, s.cell = nn.Linear(D, D), nn.LSTMCell(E + D, D)
        s.W1, s.W2, s.v = nn.Linear(D, D, bias=False), nn.Linear(D, D, bias=False), nn.Linear(D, 1, bias=False)
        s.out, s.drop = nn.Linear(2 * D, V), nn.Dropout(0.3)

    def attend(s, h, enc, mask):
        if s.att == "bahdanau":
            sc = s.v(torch.tanh(s.W1(h).unsqueeze(1) + s.W2(enc))).squeeze(-1)
        else:  # luong (general)
            sc = (enc @ s.W1(h).unsqueeze(-1)).squeeze(-1)
        a = sc.masked_fill(~mask, -1e9).softmax(-1)
        return (a.unsqueeze(-1) * enc).sum(1), a

    def encode(s, x):
        mask = x != 0
        enc, _ = s.enc(s.drop(s.emb(x)))
        m = mask.unsqueeze(-1).float()
        h = torch.tanh(s.init((enc * m).sum(1) / m.sum(1)))
        return enc, mask, (h, torch.zeros_like(h))

    def step(s, y, st, enc, mask):
        ctx, a = s.attend(st[0], enc, mask)
        h, c = s.cell(torch.cat([s.drop(s.emb(y)), ctx], -1), st)
        return s.out(torch.cat([h, ctx], -1)), (h, c), a

    def forward(s, x, y):
        enc, mask, st = s.encode(x); out = []
        for t in range(y.size(1) - 1):
            lg, st, _ = s.step(y[:, t], st, enc, mask); out.append(lg)
        return torch.stack(out, 1)

    @torch.no_grad()
    def greedy(s, x, maxlen=24):
        enc, mask, st = s.encode(x)
        y = torch.full((x.size(0),), BOS); ids, att = [], []
        for _ in range(maxlen):
            lg, st, a = s.step(y, st, enc, mask); y = lg.argmax(-1); ids.append(y); att.append(a)
        return torch.stack(ids, 1), torch.stack(att, 1)


def run_lstm(att="bahdanau", epochs=30):
    torch.manual_seed(CFG["seed"]); random.seed(CFG["seed"])
    P = pairs()
    cnt = Counter(t for a, b in P["train"] for t in tokenize(a, stop=False) + tokenize(b, stop=False))
    vocab = {w: i + 4 for i, (w, _) in enumerate(cnt.most_common(8000))}
    inv = {i: w for w, i in vocab.items()}
    V = len(vocab) + 4
    net = S2S(V, att)
    opt = torch.optim.Adam(net.parameters(), 2e-3)

    def loss_on(rows):
        x, y = enc_src([a for a, _ in rows], vocab), enc_tgt([t for _, t in rows], vocab)
        return nn.functional.cross_entropy(net(x, y).reshape(-1, V), y[:, 1:].reshape(-1), ignore_index=0)

    best, tr = (1e9, None), list(P["train"])
    for ep in range(epochs):
        net.train(); random.shuffle(tr)
        for i in range(0, len(tr), 16):
            loss = loss_on(tr[i:i + 16])
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        net.eval()
        with torch.no_grad():
            vl = loss_on(P["val"]).item()
        if vl < best[0]:
            best = (vl, {k: v.clone() for k, v in net.state_dict().items()})
    net.load_state_dict(best[1]); net.eval()

    def words(row):
        out = []
        for i in row.tolist():
            if i == EOS: break
            out.append(inv.get(i, "<unk>"))
        return out

    texts, refs = [a for a, _ in P["test"]], [t for _, t in P["test"]]
    ids, _ = net.greedy(enc_src(texts, vocab))
    with torch.no_grad():
        tl = loss_on(P["test"]).item()
    res = scores([" ".join(words(r)) for r in ids], refs)
    res["test_perplexity"] = math.exp(tl)
    save(f"lstm_{att}", res); print(att, res)
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ids, a = net.greedy(enc_src(texts[:1], vocab))
    w, sw = words(ids[0]) or ["<eos>"], tokenize(texts[0], stop=False)[:40]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(a[0, :len(w), :len(sw)].numpy(), cmap="viridis")
    ax.set_xticks(range(len(sw)), sw, rotation=90, fontsize=6); ax.set_yticks(range(len(w)), w)
    fig.tight_layout(); fig.savefig(REP / f"attn_{att}.png", dpi=120); plt.close(fig)


# ---------- T5: titles (fine-tuned) + summaries ----------
@functools.lru_cache(maxsize=2)
def _lm(path):
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    return AutoTokenizer.from_pretrained(path), AutoModelForSeq2SeqLM.from_pretrained(path).eval()


def gen(texts, prompt, path=None, max_in=512, max_new=32, beams=4):
    tok, m = _lm(path or CFG["gen_model"]); out = []
    for i in range(0, len(texts), 8):
        enc = tok([prompt + t for t in texts[i:i + 8]], return_tensors="pt", truncation=True,
                  max_length=max_in, padding=True)
        with torch.no_grad():
            ids = m.generate(**enc, max_new_tokens=max_new, num_beams=beams, no_repeat_ngram_size=3)
        out += tok.batch_decode(ids, skip_special_tokens=True)
    return out


def summ_source(d):
    parts = [s["text"] for s in d["sections"] if s["role"] == "results" or re.search("conclusion", s["heading"], re.I)]
    return " ".join(parts) or d["abstract"]


def summarize(text):
    return gen([text], SUMM, max_in=768, max_new=120)[0]


def run_t5(epochs=3):
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, set_seed
    set_seed(CFG["seed"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    base, P = CFG["gen_model"], pairs()
    tok = AutoTokenizer.from_pretrained(base)
    m = AutoModelForSeq2SeqLM.from_pretrained(base).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    tr = list(P["train"])
    for ep in range(epochs):
        m.train(); random.Random(ep).shuffle(tr)
        for i in range(0, len(tr), 8):
            b = tr[i:i + 8]
            x = tok([TITLE + a for a, _ in b], return_tensors="pt", truncation=True, max_length=512, padding=True).to(dev)
            y = tok([t for _, t in b], return_tensors="pt", truncation=True, max_length=32, padding=True).input_ids.to(dev)
            y[y == tok.pad_token_id] = -100
            m(**x, labels=y).loss.backward(); opt.step(); opt.zero_grad()
        print("t5 epoch", ep, "done")
    m.save_pretrained(str(MOD / "t5_title")); tok.save_pretrained(str(MOD / "t5_title"))
    del m
    texts, refs = [a for a, _ in P["test"]], [t for _, t in P["test"]]
    save("t5_title", {"zero_shot": scores(gen(texts, TITLE), refs),
                      "finetuned": scores(gen(texts, TITLE, str(MOD / "t5_title")), refs)})


def run_summ():
    sp = {c["paper_id"]: c["split"] for c in read_jsonl(PROC / "chunks.jsonl")}
    docs = [d for d in read_jsonl(PROC / "corpus.jsonl") if sp.get(d["paper_id"]) == "test" and d["abstract"]]
    preds = gen([summ_source(d) for d in docs], SUMM, max_in=768, max_new=120)
    save("summary_vs_abstract", scores(preds, [d["abstract"] for d in docs]))
    REP.mkdir(exist_ok=True)
    with open(REP / "summary_check.jsonl", "w", encoding="utf-8") as f:  # manual check of 20
        for d, p in list(zip(docs, preds))[:20]:
            f.write(json.dumps({"paper_id": d["paper_id"], "title": d["title"], "generated": p,
                                "abstract": d["abstract"]}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    for a in sys.argv[1:] or ["lm", "lstm"]:
        {"lm": run_lm, "lstm": lambda: [run_lstm("bahdanau"), run_lstm("luong")],
         "t5": run_t5, "summ": run_summ}[a]()