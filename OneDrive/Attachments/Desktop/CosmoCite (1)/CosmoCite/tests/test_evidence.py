from evidence import evidence_profile

PAPERS = [{"paper_id": "0001", "year": 2014, "entities": {"organism": ["mouse"], "stressor": ["microgravity"], "tissue": ["bone"]}},
          {"paper_id": "0002", "year": 2020, "entities": {"organism": ["mouse", "human"], "stressor": ["radiation"], "tissue": []}}]


def h(p, s):
    return {"paper_id": p, "rerank": s}


def test_profile_counts_each_paper_once():
    out = evidence_profile([h("0001", 5), h("0001", 6), h("0002", 5)], PAPERS, 4.1)
    assert out["papers"] == 2 and out["organisms"] == {"mouse": 2, "human": 1}
    assert (out["year_min"], out["year_max"]) == (2014, 2020)


def test_profile_ignores_hits_below_the_threshold():
    assert evidence_profile([h("0001", 5), h("0002", 1)], PAPERS, 4.1)["papers"] == 1


def test_profile_ignores_hits_without_a_rerank_score():
    assert evidence_profile([h("0001", 5), h("0002", None)], PAPERS, 4.1)["papers"] == 1


def test_profile_is_none_without_supporting_hits():
    assert evidence_profile([h("0001", 1)], PAPERS, 4.1) is None


def test_profile_skips_papers_missing_from_the_index():
    assert evidence_profile([h("9999", 5), h("0001", 5)], PAPERS, 4.1)["papers"] == 1


def test_profile_tolerates_unknown_years():
    rows = [{**PAPERS[0], "year": None}]
    assert evidence_profile([h("0001", 5)], rows, 4.1)["year_min"] is None
