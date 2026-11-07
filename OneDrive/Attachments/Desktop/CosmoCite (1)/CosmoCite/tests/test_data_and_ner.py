import json
from collections import defaultdict
from data import PROC, chunk, read_jsonl, role_of
from features import tokenize
from ner import extract, paper_entities


def test_no_paper_appears_in_two_splits():
    splits = defaultdict(set)
    for c in read_jsonl(PROC / "chunks.jsonl"):
        splits[c["paper_id"]].add(c["split"])
    assert all(len(s) == 1 for s in splits.values())


def test_chunks_respect_the_word_budget_unless_one_sentence_exceeds_it():
    text = " ".join(["word " * 9 + "end."] * 10)
    parts = chunk(text, 30)
    assert len(parts) > 1 and all(len(p.split()) <= 30 for p in parts)
    assert " ".join(parts) == text.strip()


def test_section_headings_map_to_roles():
    assert role_of("2. Materials and Methods") == "methods"
    assert role_of("Results and Discussion") == "results"
    assert role_of("Funding notes") == "other"


def test_tokenize_switches():
    assert "the" in tokenize("the bones", stop=False)
    assert "the" not in tokenize("the bones")
    assert tokenize("the bones", stem=True) == ["bone"]


def test_extract_prefers_the_longest_overlapping_match():
    found = extract("We studied simulated microgravity in mice.")
    assert {"simulated microgravity", "mice"} == {x["text"] for x in found}


def test_title_mention_is_enough_to_tag_a_paper():
    assert "mouse" in paper_entities("Bone loss in mice", "", "")["organism"]


def test_one_passing_abstract_mention_does_not_tag_a_paper():
    e = paper_entities("Muscle atrophy during spaceflight", "Human expansion in space is hard. We used mice and mice.", "")
    assert "human" not in e["organism"] and "mouse" in e["organism"]


def test_every_gap_cell_lists_as_many_papers_as_it_counts():
    assert all(g["count"] == len(g["paper_ids"]) for g in json.loads((PROC / "gaps.json").read_text(encoding="utf-8")))


def test_embed_chunks_saves_one_row_per_chunk(tmp_path, monkeypatch):
    import numpy as np
    import features
    (tmp_path / "chunks.jsonl").write_text('{"text": "a"}\n{"text": "b"}\n')
    monkeypatch.setattr(features, "PROC", tmp_path)
    monkeypatch.setattr(features, "embed", lambda texts: np.zeros((len(list(texts)), 4), dtype=np.float32))
    features.embed_chunks()
    assert np.load(tmp_path / "chunk_emb.npy").shape == (2, 4)
