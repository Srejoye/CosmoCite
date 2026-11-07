"""Tokenizing, chunk embeddings, Word2Vec, paper t-SNE. Run: python backend/features.py"""
import functools, json, re, sys
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from data import CFG, PROC, read_jsonl

_stemmer = None


def tokenize(text, stop=True, stem=False):
    """Preprocessing switches (stop/stem) double as the Unit I ablation."""
    global _stemmer
    toks = re.findall(r"[a-z0-9]{2,}", text.lower())
    if stop:
        toks = [t for t in toks if t not in ENGLISH_STOP_WORDS]
    if stem:
        if _stemmer is None:
            from nltk.stem import PorterStemmer
            _stemmer = PorterStemmer()
        toks = [_stemmer.stem(t) for t in toks]
    return toks


@functools.lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(CFG["embed_model"])


def embed(texts, batch_size=64):
    texts = list(texts)
    return _model().encode(texts, batch_size=batch_size, normalize_embeddings=True,
                           show_progress_bar=len(texts) > 100).astype(np.float32)


def embed_chunks():
    """The only step the running app needs. Run: python backend/features.py embed"""
    chunks = read_jsonl(PROC / "chunks.jsonl")
    emb = embed([c["text"] for c in chunks])
    np.save(PROC / "chunk_emb.npy", emb)
    return chunks, emb


def main():
    chunks, emb = embed_chunks()

    by = {}
    for i, c in enumerate(chunks):
        by.setdefault(c["paper_id"], []).append(i)
    pids = sorted(by)
    P = np.stack([emb[by[p]].mean(0) for p in pids])
    from sklearn.manifold import TSNE
    xy = TSNE(n_components=2, perplexity=30, init="pca", random_state=CFG["seed"]).fit_transform(P)
    (PROC / "tsne.json").write_text(json.dumps({p: xy[i].round(3).tolist() for i, p in enumerate(pids)}))

    from gensim.models import Word2Vec
    w2v = Word2Vec([tokenize(c["text"]) for c in chunks], vector_size=100, window=5, min_count=5,
                   workers=1, seed=CFG["seed"])
    w2v.save(str(PROC / "w2v.model"))
    print("embeddings", emb.shape, "| tsne papers", len(pids), "| w2v vocab", len(w2v.wv))


if __name__ == "__main__":
    embed_chunks() if sys.argv[1:] == ["embed"] else main()