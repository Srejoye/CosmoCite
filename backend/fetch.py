"""Download PMC article pages. Run: python backend/fetch.py [limit]
Reads data/SB_publication_PMC.csv (any column holding a PMC link/id) -> data/raw_html/paper_0001.html ...
With --csv <list> any PMC list works: pages go to data/raw_html/incoming/PMC123.html, ready for backend/ingest.py."""
import csv, re, sys, time, urllib.request
from pathlib import Path
from data import CFG, ROOT

CSV = ROOT / "data" / "SB_publication_PMC.csv"
OUT = ROOT / CFG["paths"]["raw_html"]
UA = {"User-Agent": "Mozilla/5.0 (CosmoCite student project; contact: runasaha1095@gmail.com)"}


def pmc_ids(path):
    ids = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.reader(f):
            m = re.search(r"PMC\d+", " ".join(row))
            if m and m.group(0) not in ids:
                ids.append(m.group(0))
    return ids


def targets(ids, out, by_id):
    return [(pmc, out / (f"{pmc}.html" if by_id else f"paper_{n:04d}.html")) for n, pmc in enumerate(ids, 1)]


def main():
    args = sys.argv[1:]
    custom = args.pop(args.index("--csv") + 1) if "--csv" in args else None
    if custom:
        args.remove("--csv")
    src, out = (Path(custom), OUT / "incoming") if custom else (CSV, OUT)
    out.mkdir(parents=True, exist_ok=True)
    ids = pmc_ids(src)
    limit = int(args[0]) if args else len(ids)
    print(len(ids), "PMC ids found; fetching", limit)
    ok = fail = 0
    for n, (pmc, path) in enumerate(targets(ids[:limit], out, bool(custom)), 1):
        if path.exists() and path.stat().st_size > 5000:
            ok += 1; continue
        for attempt in range(3):
            try:
                req = urllib.request.Request(f"https://pmc.ncbi.nlm.nih.gov/articles/{pmc}/", headers=UA)
                html = urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "ignore")
                if "citation_title" not in html:
                    raise ValueError("no article metadata")
                path.write_text(html, encoding="utf-8"); ok += 1
                break
            except Exception as e:
                time.sleep(2 * (attempt + 1))
                if attempt == 2:
                    fail += 1; print("FAILED", pmc, e)
        time.sleep(0.4)
        if n % 25 == 0:
            print(n, "done | ok", ok, "fail", fail)
    print("finished | ok", ok, "fail", fail)


if __name__ == "__main__":
    main()