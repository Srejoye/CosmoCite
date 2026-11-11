"""Turns label files into spreadsheets for people, and reads their answers back.
Run: python backend/labels.py export    writes data/eval/labeling/*.csv
     python backend/labels.py import    reads answers into faithfulness.jsonl and qrels.json
     python backend/labels.py export_heldout | import_heldout    the same for the held-out sentences (heldout.csv)

Answers are yes or no. Blanks mean "not done yet", and anything else stops the import, so nothing is guessed."""
import csv, json, sys
from data import CFG, PROC, ROOT, read_jsonl, write_jsonl

EVAL = ROOT / CFG["paths"]["eval"]
SHEETS = EVAL / "labeling"
FAITH_CSV = SHEETS / "faithfulness.csv"
RELEVANCE_CSV = SHEETS / "relevance.csv"
HELD_CSV = SHEETS / "heldout.csv"
ANSWERS = {"yes": True, "y": True, "no": False, "n": False, "": None}
ABSTRACT_CHARS = 600


def parse_answer(text):
    key = text.strip().lower()
    if key not in ANSWERS:
        raise ValueError(f"expected yes or no, got {text!r}")
    return ANSWERS[key]


def faith_sheet(rows):
    return [{"row": n, "question": x["q"], "sentence": x["sentence"], "passage": x["passage"], "supported": "", "on_topic": ""}
            for n, x in enumerate(rows, 1)]


def apply_faith(rows, sheet):
    out = [dict(x) for x in rows]
    for line in sheet:
        n = int(line["row"])
        if not 1 <= n <= len(out):
            raise ValueError(f"row {n} is not in the sample")
        if "sentence" in line and line["sentence"].strip() != out[n - 1]["sentence"].strip():
            raise ValueError(f"row {n} no longer matches the sample, so its answers would label the wrong sentence")
        try:
            supported, on_topic = parse_answer(line["supported"]), parse_answer(line["on_topic"])
        except ValueError as e:
            raise ValueError(f"row {n}: {e}") from e
        if supported is not None:
            out[n - 1]["supported"] = supported
        if on_topic is not None:
            out[n - 1]["on_topic"] = on_topic
    return out


def relevance_sheet(pool, qrels, abstracts):
    done = {q["qid"] for q in qrels}
    return [{"qid": q["qid"], "question": q["question"], "paper_id": p["paper_id"], "title": p["title"],
             "abstract": abstracts.get(p["paper_id"], ""), "relevant": ""}
            for q in pool if q["qid"] not in done for p in q["pool"]]


def apply_relevance(qrels, pool, sheet):
    """Adds a row for each question with every paper answered. Existing judgments are never replaced."""
    answers = {}
    for line in sheet:
        try:
            answers[line["qid"], line["paper_id"]] = parse_answer(line["relevant"])
        except ValueError as e:
            raise ValueError(f"{line['qid']} paper {line['paper_id']}: {e}") from e
    done, out = {q["qid"] for q in qrels}, list(qrels)
    for q in pool:
        ids = [p["paper_id"] for p in q["pool"]]
        got = [answers.get((q["qid"], i)) for i in ids]
        if q["qid"] in done or None in got:
            continue
        out.append({"qid": q["qid"], "question": q["question"], "judged": ids,
                    "relevant": [i for i, yes in zip(ids, got) if yes], "label_source": "human"})
    return out


def write_csv(path, rows):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:  # the BOM lets Excel read accents and symbols
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def has_answers(path, *columns):
    return path.exists() and any(line[c].strip() for line in read_csv(path) for c in columns)


def export():
    SHEETS.mkdir(exist_ok=True)
    if has_answers(FAITH_CSV, "supported", "on_topic") or has_answers(RELEVANCE_CSV, "relevant"):
        return print("A sheet already has answers, so nothing was overwritten.")
    abstracts = {d["paper_id"]: d["abstract"][:ABSTRACT_CHARS] for d in read_jsonl(PROC / "corpus.jsonl")}
    pool = json.loads((EVAL / "qrels_pool.json").read_text(encoding="utf-8"))
    qrels = json.loads((EVAL / "qrels.json").read_text(encoding="utf-8"))
    faith, relevance = faith_sheet(read_jsonl(EVAL / "faithfulness.jsonl")), relevance_sheet(pool, qrels, abstracts)
    write_csv(FAITH_CSV, faith)
    write_csv(RELEVANCE_CSV, relevance)
    print(len(faith), "sentences and", len(relevance), "papers to label in", SHEETS)


def load():
    path = EVAL / "faithfulness.jsonl"
    rows = apply_faith(read_jsonl(path), read_csv(FAITH_CSV))
    pool = json.loads((EVAL / "qrels_pool.json").read_text(encoding="utf-8"))
    qrels_path = EVAL / "qrels.json"
    qrels = json.loads(qrels_path.read_text(encoding="utf-8"))
    merged = apply_relevance(qrels, pool, read_csv(RELEVANCE_CSV))
    write_jsonl(path, rows)
    qrels_path.write_text(json.dumps(merged, indent=1, ensure_ascii=False), encoding="utf-8")
    done = sum(x["supported"] is not None and x["on_topic"] is not None for x in rows)
    print(f"{done} of {len(rows)} sentences labeled; {len(merged) - len(qrels)} questions added, {len(pool) - len(merged)} left")


def export_heldout():
    SHEETS.mkdir(exist_ok=True)
    if has_answers(HELD_CSV, "supported", "on_topic"):
        return print("The sheet already has answers, so nothing was overwritten.")
    sheet = faith_sheet(read_jsonl(EVAL / "heldout.jsonl"))
    write_csv(HELD_CSV, sheet)
    print(len(sheet), "sentences to label in", HELD_CSV)


def load_heldout():
    path = EVAL / "heldout.jsonl"
    rows = apply_faith(read_jsonl(path), read_csv(HELD_CSV))
    write_jsonl(path, rows)
    print(f"{sum(x['on_topic'] is not None for x in rows)} of {len(rows)} sentences labeled")


if __name__ == "__main__":
    {"export": export, "import": load, "export_heldout": export_heldout, "import_heldout": load_heldout}[sys.argv[1] if len(sys.argv) > 1 else "export"]()
