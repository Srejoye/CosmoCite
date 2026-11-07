"""Authors and journals from NCBI's free summary service. Run: python backend/metadata.py
Writes data/processed/metadata.json, keyed by PMC id. Citations work without it, showing only the year."""
import json, urllib.parse, urllib.request
from data import PROC, read_jsonl

URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
BATCH = 200


def parse_summary(result):
    out = {}
    for uid in result["uids"]:
        x = result[uid]
        if "error" not in x:
            out["PMC" + uid] = {"authors": [a["name"] for a in x.get("authors", []) if a.get("authtype") == "Author"], "journal": x.get("fulljournalname", "")}
    return out


def fetch(ids):
    body = urllib.parse.urlencode({"db": "pmc", "retmode": "json", "id": ",".join(ids)}).encode()
    req = urllib.request.Request(URL, body, {"User-Agent": "CosmoCite student project"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return parse_summary(json.load(r)["result"])


def main():
    ids = [d["pmc_id"].removeprefix("PMC") for d in read_jsonl(PROC / "corpus.jsonl") if d["pmc_id"]]
    out = {}
    for i in range(0, len(ids), BATCH):
        out.update(fetch(ids[i:i + BATCH]))
    (PROC / "metadata.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(len(out), "of", len(ids), "papers;", sum(bool(x["authors"]) for x in out.values()), "with authors")


if __name__ == "__main__":
    main()
