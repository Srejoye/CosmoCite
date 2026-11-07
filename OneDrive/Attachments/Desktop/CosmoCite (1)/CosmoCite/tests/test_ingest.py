import json
import numpy as np
import pytest
import data
import fetch
import ingest
from data import assign_split, chunks_of, read_jsonl, write_jsonl
from ingest import ingest_files, next_paper_id

PAGE = """<html><head><meta name="citation_title" content="{title}"><meta name="citation_publication_date" content="2024/01/02">
<link rel="canonical" href="https://pmc.ncbi.nlm.nih.gov/articles/{pmc}/"></head>
<body><section class="abstract"><p>Mice flown in microgravity lost bone density during the long mission in orbit.</p></section></body></html>"""


def doc(pid, pmc):
    return {"paper_id": pid, "pmc_id": pmc, "title": "Old " + pid, "year": 2020, "abstract": "Old abstract text here.", "sections": []}


@pytest.fixture
def proc(tmp_path, monkeypatch):
    p = tmp_path / "proc"
    p.mkdir()
    corpus = [doc("0001", "PMC1"), doc("0607", "PMC2")]
    write_jsonl(p / "corpus.jsonl", corpus)
    write_jsonl(p / "chunks.jsonl", [c for d in corpus for c in chunks_of(d, "train")])
    np.save(p / "chunk_emb.npy", np.zeros((2, 4), dtype=np.float32))
    (p / "papers.json").write_text(json.dumps([{"paper_id": d["paper_id"], "entities": {"organism": [], "stressor": []}} for d in corpus]), encoding="utf-8")
    (p / "gaps.json").write_text("[]", encoding="utf-8")
    monkeypatch.setattr(ingest, "embed", lambda t: np.ones((len(t), 4), dtype=np.float32))
    return p


@pytest.fixture
def html_file(tmp_path):
    f = tmp_path / "PMC999.html"
    f.write_text(PAGE.format(title="Mice in microgravity", pmc="PMC999"), encoding="utf-8")
    return f


def test_ingest_appends_new_papers_and_keeps_rows_aligned(proc, html_file):
    out = ingest_files([html_file], proc)
    assert out == {"added": 1, "skipped": 0, "chunks": 1}
    assert len(read_jsonl(proc / "chunks.jsonl")) == np.load(proc / "chunk_emb.npy").shape[0] == 3


def test_new_papers_continue_the_numbering_from_any_file_name(proc, html_file):
    ingest_files([html_file], proc)
    assert read_jsonl(proc / "corpus.jsonl")[-1]["paper_id"] == "0608"
    assert read_jsonl(proc / "chunks.jsonl")[-1]["chunk_id"] == "0608_000"


def test_ingested_papers_reach_explore_and_the_gap_table(proc, html_file):
    ingest_files([html_file], proc)
    paper = json.loads((proc / "papers.json").read_text(encoding="utf-8"))[-1]
    assert paper["paper_id"] == "0608" and paper["summary"] and paper["tsne"] is None
    assert "mouse" in paper["entities"]["organism"]
    gaps = json.loads((proc / "gaps.json").read_text(encoding="utf-8"))
    assert next(g for g in gaps if (g["organism"], g["stressor"]) == ("mouse", "microgravity"))["paper_ids"] == ["0608"]


def test_ingesting_the_same_paper_twice_changes_nothing(proc, html_file):
    ingest_files([html_file], proc)
    assert ingest_files([html_file], proc) == {"added": 0, "skipped": 1, "chunks": 0}


def test_a_duplicate_inside_one_batch_is_added_once(proc, html_file, tmp_path):
    copy = tmp_path / "copy.html"
    copy.write_text(html_file.read_text(encoding="utf-8"), encoding="utf-8")
    assert ingest_files([html_file, copy], proc) == {"added": 1, "skipped": 1, "chunks": 1}


def test_a_page_without_an_article_is_skipped(proc, tmp_path):
    bad = tmp_path / "bad.html"
    bad.write_text("<html><body>Not found</body></html>", encoding="utf-8")
    assert ingest_files([bad], proc) == {"added": 0, "skipped": 1, "chunks": 0}


def test_a_failed_embedding_leaves_the_corpus_untouched(proc, html_file, monkeypatch):
    before = (proc / "chunks.jsonl").read_text(encoding="utf-8")
    monkeypatch.setattr(ingest, "embed", lambda t: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError):
        ingest_files([html_file], proc)
    assert (proc / "chunks.jsonl").read_text(encoding="utf-8") == before


def test_a_failed_write_leaves_the_corpus_untouched(proc, html_file, monkeypatch):
    before = (proc / "chunks.jsonl").read_text(encoding="utf-8")
    monkeypatch.setattr(ingest.np, "save", lambda *a: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError):
        ingest_files([html_file], proc)
    assert (proc / "chunks.jsonl").read_text(encoding="utf-8") == before


def test_existing_papers_keep_their_split_after_ingest(proc, html_file):
    before = {c["chunk_id"]: c["split"] for c in read_jsonl(proc / "chunks.jsonl")}
    ingest_files([html_file], proc)
    after = {c["chunk_id"]: c["split"] for c in read_jsonl(proc / "chunks.jsonl")}
    assert all(after[k] == v for k, v in before.items())


def test_next_paper_id_continues_the_numbering():
    assert next_paper_id({"0001", "0607"}) == "0608"
    assert next_paper_id(set()) == "0001"


def test_split_is_stable_and_follows_the_configured_shares():
    picks = [assign_split(f"{i:04d}") for i in range(2000)]
    assert picks == [assign_split(f"{i:04d}") for i in range(2000)]
    assert abs(picks.count("train") / 2000 - data.CFG["split"]["train"]) < 0.05
    assert set(picks) == {"train", "val", "test"}


def test_chunks_of_numbers_the_abstract_and_sections_in_order():
    d = {**doc("0005", "PMC5"), "sections": [{"role": "results", "heading": "Results", "text": "Bone fell."}]}
    rows = chunks_of(d, "val")
    assert [(c["chunk_id"], c["section_role"], c["heading"], c["split"]) for c in rows] == [
        ("0005_000", "abstract", "Abstract", "val"), ("0005_001", "results", "Results", "val")]


def test_parse_accepts_a_paper_id_for_any_file_name(html_file):
    assert data.parse(html_file, "0700")["paper_id"] == "0700"


def test_fetch_reads_ids_from_any_list(tmp_path):
    f = tmp_path / "list.csv"
    f.write_text("title,link\nA,https://x/PMC11/\nB,PMC22\nC,PMC11\n", encoding="utf-8")
    assert fetch.pmc_ids(f) == ["PMC11", "PMC22"]


def test_a_custom_list_is_saved_by_id_so_it_cannot_collide_with_numbered_pages(tmp_path):
    assert fetch.targets(["PMC1"], tmp_path, by_id=False) == [("PMC1", tmp_path / "paper_0001.html")]
    assert fetch.targets(["PMC1"], tmp_path, by_id=True) == [("PMC1", tmp_path / "PMC1.html")]
