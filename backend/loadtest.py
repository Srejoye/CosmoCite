"""Load test against a running server -> reports/load.json. Run: python backend/loadtest.py [base url]
Every request is a new question, so the cache does not flatter the result. Start the server fresh first."""
import json, statistics, sys, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor
from data import CFG, ROOT

LEVELS = ((1, 5), (8, 24), (20, 26))  # (people asking at once, questions)


def summary(results, seconds):
    served = sorted(s for code, s in results if code == 200)
    return {"served": len(served), "busy": sum(code == 429 for code, _ in results),
            "errors": sum(code not in (200, 429) for code, _ in results),
            "p50_s": round(statistics.median(served), 2) if served else None,
            "p95_s": round(served[max(int(len(served) * 0.95) - 1, 0)], 2) if served else None,
            "per_second": round(len(served) / seconds, 2)}


def ask(base, question):
    body = json.dumps({"question": question}).encode()
    req = urllib.request.Request(f"{base}/api/ask", body, {"Content-Type": "application/json"})
    start = time.time()
    try:
        urllib.request.urlopen(req, timeout=120).read()
        return 200, time.time() - start
    except urllib.error.HTTPError as e:
        return e.code, time.time() - start


def main(base="http://127.0.0.1:8000"):
    questions = [q["question"] for q in json.loads((ROOT / CFG["paths"]["eval"] / "questions.json").read_text(encoding="utf-8"))]
    rows, offset = [], 0
    for people, n in LEVELS:
        start = time.time()
        with ThreadPoolExecutor(people) as pool:
            results = list(pool.map(lambda q: ask(base, q), questions[offset:offset + n]))
        rows.append({"people": people, "questions": n, **summary(results, time.time() - start)})
        offset += n
        print(rows[-1])
    (ROOT / "reports" / "load.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main(*sys.argv[1:2])
