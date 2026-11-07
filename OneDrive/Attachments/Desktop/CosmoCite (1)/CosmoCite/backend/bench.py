"""Search latency and recall as the index grows. Run: python backend/bench.py   writes reports/scale.json
Each search asks for candidate_k results, as the app does. Random vectors are the hard case for approximate search,
so the "real" rows repeat the measurement on the corpus with the evaluation questions. Timings are from one machine."""
import json, time
import numpy as np
from data import CFG, PROC, ROOT
from index import NumpyIndex, load_index

K = 10
SIZES = (20_242, 200_000, 500_000)


def unit_vectors(n, dim, rng):
    v = rng.standard_normal((n, dim), dtype=np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


def measure(kind, emb, queries, data):
    start = time.perf_counter()
    index, exact = load_index(kind, emb), NumpyIndex(emb)
    build, depth, ms, found = time.perf_counter() - start, CFG["retrieval"]["candidate_k"], [], 0
    for q in queries:
        start = time.perf_counter()
        ids, _ = index.top(q, depth)
        ms.append((time.perf_counter() - start) * 1000)
        found += len(set(ids[:K].tolist()) & set(exact.top(q, K)[0].tolist()))
    return {"kind": kind, "size": len(emb), "data": data, "build_s": round(build, 2), "p50_ms": round(float(np.percentile(ms, 50)), 3),
            "p95_ms": round(float(np.percentile(ms, 95)), 3), "recall@10": round(found / (K * len(queries)), 4)}


def bench(kinds, sizes, dim=384, queries=200, seed=42, on_row=lambda rows: None):
    rng, rows = np.random.default_rng(seed), []
    for size in sizes:
        emb, qs = unit_vectors(size, dim, rng), unit_vectors(queries, dim, rng)
        for kind in kinds:
            rows.append(measure(kind, emb, qs, "random"))
            on_row(rows)
    return rows


def main():
    from features import embed
    questions = json.loads((ROOT / CFG["paths"]["eval"] / "questions.json").read_text(encoding="utf-8"))
    emb, qs = np.load(PROC / "chunk_emb.npy"), embed([q["question"] for q in questions])
    real = [measure(kind, emb, qs, "real") for kind in ("numpy", "faiss")]

    def save(rows):  # after every row, so an interrupted run keeps what it measured
        (ROOT / "reports" / "scale.json").write_text(json.dumps(real + rows, indent=1), encoding="utf-8")
        print(rows[-1], flush=True)

    bench(("numpy", "faiss"), SIZES, on_row=save)


if __name__ == "__main__":
    main()
