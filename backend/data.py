"""Raw PMC HTML -> corpus.jsonl + chunks.jsonl (split by paper). Run: python backend/data.py"""
import os
os.environ["USE_TF"] = "0"; os.environ["TRANSFORMERS_NO_TF"] = "1"

import hashlib, json, random, re, sys
from pathlib import Path
import yaml
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())
PROC = ROOT / CFG["paths"]["processed"]

DROP = re.compile(r"reference|associated data|footnote|acknowledg|contribution|funding|competing|conflict|"
                  r"supplement|data availability|contributor|abbreviation|ethic|consent|review board|"
                  r"disclosure|disclaimer|author information|additional information|appendix|glossary|availability|peer review|"
                  r"declaration|publisher|grants|biograph|accession|supporting|accessibility", re.I)
ROLES = [("methods", r"method|material|procedure|experimental|study design|protocol"),
         ("results", r"result|finding"),
         ("discussion", r"discussion|conclusion|limitation|summary|outlook|perspective"),
         ("intro", r"introduc|background|overview")]
CITE = re.compile(r"\[\s*\d+(?:\s*[,–-]\s*\d+)*\s*\]")


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(p, rows):
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def role_of(heading):
    h = re.sub(r"^[\dIVX.\s]+", "", heading.lower())
    for role, pat in ROLES:  # "results and discussion" -> results (first match wins)
        if re.search(pat, h):
            return role
    return "other"


def clean(t):
    return re.sub(r"\s+", " ", CITE.sub("", t)).strip()


def paras(node):
    out = []
    for p in node.find_all("p"):
        if p.find_parent(["figure", "figcaption", "table", "aside"]) or p.find_parent(class_=re.compile("fn|caption|table-wrap")):
            continue
        t = clean(p.get_text(" ", strip=True))
        if len(t.split()) >= 8:
            out.append(t)
    return out


def meta(soup, name):
    m = soup.find("meta", attrs={"name": name})
    return m["content"].strip() if m and m.get("content") else ""


def parse(path, paper_id=None):
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "lxml")
    pid = paper_id or re.search(r"paper_(\d+)", path.name).group(1)
    canon = soup.find("link", rel="canonical")
    pmc = re.search(r"PMC\d+", canon["href"]).group(0) if canon and canon.get("href") and "PMC" in canon["href"] else ""
    year = re.search(r"(19|20)\d{2}", meta(soup, "citation_publication_date") or meta(soup, "citation_date"))
    ab = soup.select_one("section.abstract")
    abstract = " ".join(paras(ab)) if ab else ""
    sections, body = [], soup.select_one("section.body") or soup.select_one("div.main-article-body")
    if body:
        tops = [s for s in body.find_all("section", recursive=False) if "abstract" not in (s.get("class") or [])
                and "ack" not in (s.get("class") or [])]
        for s in tops:
            h = s.find(["h2", "h3"], recursive=False)
            heading = h.get_text(" ", strip=True) if h else ""
            if DROP.search(heading):
                continue
            text = " ".join(paras(s))
            if text:
                sections.append({"role": role_of(heading) if heading else "other", "heading": heading, "text": text})
        if not sections:  # pages without headed sections
            text = " ".join(paras(body))
            if text:
                sections.append({"role": "other", "heading": "", "text": text})
    return {"paper_id": pid, "pmc_id": pmc, "title": meta(soup, "citation_title"),
            "year": int(year.group(0)) if year else None, "abstract": abstract, "sections": sections}


def chunk(text, max_words):
    sents, cur, n, out = re.split(r"(?<=[.!?])\s+", text), [], 0, []
    for s in sents:
        w = len(s.split())
        if cur and n + w > max_words:
            out.append(" ".join(cur)); cur, n = [], 0
        cur.append(s); n += w
    if cur:
        out.append(" ".join(cur))
    return out


def chunks_of(doc, split):
    parts = [("abstract", "Abstract", doc["abstract"])] if doc["abstract"] else []
    parts += [(s["role"], s["heading"], s["text"]) for s in doc["sections"]]
    out = []
    for role, heading, text in parts:
        for c in chunk(text, CFG["chunk"]["max_words"]):
            out.append({"chunk_id": f"{doc['paper_id']}_{len(out):03d}", "paper_id": doc["paper_id"], "section_role": role,
                        "heading": heading, "text": c, "split": split})
    return out


def assign_split(paper_id):
    """A split for a paper added later. It depends only on the id, so adding papers never moves an existing one."""
    bucket = int(hashlib.sha256(f"{CFG['seed']}{paper_id}".encode()).hexdigest(), 16) % 10_000 / 10_000
    sp = CFG["split"]
    return "train" if bucket < sp["train"] else "val" if bucket < sp["train"] + sp["val"] else "test"


def main():
    files = sorted((ROOT / CFG["paths"]["raw_html"]).glob("paper_*.html"))
    corpus = [d for d in map(parse, files) if d["title"] and (d["abstract"] or d["sections"])]
    ids = [d["paper_id"] for d in corpus]
    random.Random(CFG["seed"]).shuffle(ids)
    n, sp = len(ids), CFG["split"]
    cut1, cut2 = int(n * sp["train"]), int(n * (sp["train"] + sp["val"]))
    split = {i: "train" if k < cut1 else "val" if k < cut2 else "test" for k, i in enumerate(ids)}
    chunks = [c for d in corpus for c in chunks_of(d, split[d["paper_id"]])]
    PROC.mkdir(parents=True, exist_ok=True)
    write_jsonl(PROC / "corpus.jsonl", corpus)
    write_jsonl(PROC / "chunks.jsonl", chunks)
    roles = {}
    for c in chunks:
        roles[c["section_role"]] = roles.get(c["section_role"], 0) + 1
    print(f"{len(corpus)}/{len(files)} papers, {len(chunks)} chunks, roles={roles}")
    print("no body sections:", sum(1 for d in corpus if not d["sections"]), "| no abstract:", sum(1 for d in corpus if not d["abstract"]))


if __name__ == "__main__":
    sys.exit(main())