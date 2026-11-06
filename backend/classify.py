"""Section classifier: TF-IDF+LogReg -> BiLSTM/GRU+attention -> SciBERT.
Run: python backend/classify.py [tfidf|lstm|gru|bert|all]   (all = tfidf lstm gru)"""
import functools, json, sys
from collections import Counter
import numpy as np, torch, torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix
from data import CFG, ROOT, PROC, read_jsonl
from features import tokenize

LABELS = ["intro", "methods", "results", "discussion", "other"]
REP, MOD = ROOT / "reports", ROOT / "models"
BERT = "allenai/scibert_scivocab_uncased"


def load():  # text only: headings are never fed to the model
    rows = [c for c in read_jsonl(PROC / "chunks.jsonl") if c["section_role"] in LABELS]
    return {s: ([r["text"] for r in rows if r["split"] == s],
                [LABELS.index(r["section_role"]) for r in rows if r["split"] == s]) for s in ("train", "val", "test")}


def save(key, val):
    REP.mkdir(exist_ok=True)
    p = REP / "classify.json"
    d = json.loads(p.read_text()) if p.exists() else {}
    d[key] = val
    p.write_text(json.dumps(d, indent=1))


def report(name, y, p):
    f1 = float(f1_score(y, p, average="macro"))
    cm = confusion_matrix(y, p, labels=list(range(5)))
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    REP.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(5), LABELS, rotation=45)
    ax.set_yticks(range(5), LABELS)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    ax.set_title(f"{name} macro-F1={f1:.3f}")
    fig.tight_layout(); fig.savefig(REP / f"cm_{name}.png", dpi=120); plt.close(fig)
    print(name, "macro-F1", round(f1, 4))
    return {"macro_f1": f1, "confusion": cm.tolist()}


def run_tfidf(D):
    v = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None, ngram_range=(1, 2),
                        min_df=2, sublinear_tf=True)
    m = LogisticRegression(max_iter=2000, class_weight="balanced").fit(v.fit_transform(D["train"][0]), D["train"][1])
    save("tfidf", report("tfidf", D["test"][1], m.predict(v.transform(D["test"][0]))))


class Net(nn.Module):
    def __init__(s, V, cell="lstm", E=128, H=128):
        super().__init__()
        s.emb = nn.Embedding(V, E, padding_idx=0)
        s.rnn = (nn.LSTM if cell == "lstm" else nn.GRU)(E, H, batch_first=True, bidirectional=True)
        s.att, s.drop, s.out = nn.Linear(2 * H, 1), nn.Dropout(0.3), nn.Linear(2 * H, 5)

    def forward(s, x):
        h, _ = s.rnn(s.drop(s.emb(x)))
        a = s.att(h).squeeze(-1).masked_fill(x == 0, -1e9).softmax(-1)
        return s.out(s.drop((a.unsqueeze(-1) * h).sum(1))), a


def encode(texts, vocab, L=200):
    ids = [[vocab.get(t, 1) for t in tokenize(x, stop=False)[:L]] or [1] for x in texts]
    m = max(map(len, ids))
    return torch.tensor([i + [0] * (m - len(i)) for i in ids])


def run_rnn(D, cell="lstm", epochs=8):
    torch.manual_seed(CFG["seed"])
    cnt = Counter(t for x in D["train"][0] for t in tokenize(x, stop=False))
    vocab = {w: i + 2 for i, (w, c) in enumerate(cnt.most_common()) if c >= 2}
    net = Net(len(vocab) + 2, cell)
    opt = torch.optim.Adam(net.parameters(), 2e-3)
    X, y = D["train"][0], torch.tensor(D["train"][1])
    w = torch.tensor(len(y) / (5 * np.maximum(np.bincount(y.numpy(), minlength=5), 1)), dtype=torch.float)
    lossf, best = nn.CrossEntropyLoss(weight=w), (-1, None)

    def pred(texts):
        net.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(texts), 128):
                out += net(encode(texts[i:i + 128], vocab))[0].argmax(-1).tolist()
        return out

    for ep in range(epochs):
        net.train()
        idx = torch.randperm(len(y))
        for i in range(0, len(y), 64):
            b = idx[i:i + 64]
            loss = lossf(net(encode([X[j] for j in b.tolist()], vocab))[0], y[b])
            opt.zero_grad(); loss.backward(); opt.step()
        f = f1_score(D["val"][1], pred(D["val"][0]), average="macro")
        print(cell, "epoch", ep, "val macro-F1", round(f, 4))
        if f > best[0]:
            best = (f, {k: v.clone() for k, v in net.state_dict().items()})
    net.load_state_dict(best[1])
    MOD.mkdir(exist_ok=True)
    if cell == "lstm":  # lstm.pt powers the Analyze page attention
        torch.save({"state": best[1], "vocab": vocab, "cell": cell}, MOD / "lstm.pt")
    save(cell, report(cell, D["test"][1], pred(D["test"][0])))


def run_bert(D, epochs=1):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, set_seed
    set_seed(CFG["seed"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(BERT)
    m = AutoModelForSequenceClassification.from_pretrained(BERT, num_labels=5).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-5)
    X, y = D["train"]
    keep = np.random.RandomState(CFG["seed"]).permutation(len(X))[:2000]  # CPU budget
    X, y = [X[i] for i in keep], [y[i] for i in keep]

    def enc(t):
        return tok(t, truncation=True, max_length=96, padding=True, return_tensors="pt").to(dev)

    for ep in range(epochs):
        m.train()
        idx = np.random.RandomState(ep).permutation(len(X))
        for i in range(0, len(X), 16):
            b = idx[i:i + 16]
            loss = m(**enc([X[j] for j in b]), labels=torch.tensor([y[j] for j in b]).to(dev)).loss
            loss.backward(); opt.step(); opt.zero_grad()
        print("bert epoch", ep, "done")
    m.eval(); preds = []
    with torch.no_grad():
        for i in range(0, len(D["test"][0]), 64):
            preds += m(**enc(D["test"][0][i:i + 64])).logits.argmax(-1).tolist()
    m.save_pretrained(str(MOD / "bert")); tok.save_pretrained(str(MOD / "bert"))
    save("bert", report("bert", D["test"][1], preds))


@functools.lru_cache(maxsize=1)
def _lstm():
    ck = torch.load(MOD / "lstm.pt", weights_only=False)
    net = Net(len(ck["vocab"]) + 2, ck["cell"])
    net.load_state_dict(ck["state"])
    return net.eval(), ck["vocab"]


@functools.lru_cache(maxsize=1)
def _bert():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    return (AutoTokenizer.from_pretrained(str(MOD / "bert")),
            AutoModelForSequenceClassification.from_pretrained(str(MOD / "bert")).eval())


def predict(text):
    """-> {label, probs, model, attention, attention_model, attention_label}. label comes from SciBERT if trained,
    else BiLSTM. Attention always comes from the BiLSTM, so attention_label says what those weights were predicting."""
    net, vocab = _lstm()
    toks = tokenize(text, stop=False)[:200] or ["the"]
    with torch.no_grad():
        lg, a = net(torch.tensor([[vocab.get(t, 1) for t in toks]]))
    p, model = lg.softmax(-1)[0], "bilstm"
    attention_label = LABELS[int(p.argmax())]
    if (MOD / "bert").exists():
        tok, bm = _bert()
        with torch.no_grad():
            p = bm(**tok(text, truncation=True, max_length=128, return_tensors="pt")).logits.softmax(-1)[0]
        model = "scibert"
    return {"label": LABELS[int(p.argmax())], "model": model,
            "probs": {l: float(p[i]) for i, l in enumerate(LABELS)},
            "attention": [{"word": t, "weight": float(w)} for t, w in zip(toks, a[0])],
            "attention_model": "bilstm", "attention_label": attention_label}


if __name__ == "__main__":
    modes = sys.argv[1:] or ["all"]
    if "all" in modes:
        modes = ["tfidf", "lstm", "gru"]
    D = load()
    for m in modes:
        {"tfidf": lambda: run_tfidf(D), "lstm": lambda: run_rnn(D, "lstm"),
         "gru": lambda: run_rnn(D, "gru"), "bert": lambda: run_bert(D)}[m]()