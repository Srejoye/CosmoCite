import evaluate
import pytest
from retrieval import paper_metrics


def test_perfect_ranking_with_more_gold_than_k_shows_the_recall_ceiling():
    gold = [str(i) for i in range(20)]
    m = paper_metrics(gold, gold, k=10)
    assert m["recall@10"] == 0.5          # plain recall cannot reach 1.0 here
    assert m["recall_capped@10"] == 1.0   # the capped version can
    assert m["precision@10"] == 1.0
    assert m["ndcg@10"] == pytest.approx(1.0)


def test_mrr_uses_first_relevant_rank():
    assert paper_metrics(["x", "y", "a"], ["a"], k=10)["mrr"] == pytest.approx(1 / 3)


def test_no_hits_scores_zero():
    m = paper_metrics(["x", "y"], ["a"], k=10)
    assert m["recall@10"] == m["mrr"] == m["ndcg@10"] == 0


def test_only_top_k_counts():
    assert paper_metrics(["x", "a"], ["a"], k=1)["recall@1"] == 0


def test_bootstrap_ci_brackets_the_mean_and_is_repeatable():
    v = [0.0, 1.0] * 20
    lo, hi = evaluate.bootstrap_ci(v, seed=1)
    assert lo < 0.5 < hi and (lo, hi) == evaluate.bootstrap_ci(v, seed=1)


def test_bootstrap_ci_of_a_constant_has_no_width():
    assert evaluate.bootstrap_ci([1.0] * 5, seed=1) == (1.0, 1.0)


def test_merge_pool_keeps_judged_questions_and_adds_new_ones():
    out = evaluate.merge_pool([{"qid": "a", "pool": [1]}], [{"qid": "a", "pool": [2]}, {"qid": "b", "pool": [3]}])
    assert out == [{"qid": "a", "pool": [1]}, {"qid": "b", "pool": [3]}]


def test_faith_rows_pair_each_sentence_with_its_source_and_blank_labels():
    s = [{"decoding": "synth", "q": "q", "answer": "Bone fell. [1]", "citations": [{"quote": "Bone volume fell.", "chunk_id": "c1"}]},
         {"decoding": "greedy", "q": "q", "answer": "x y z. [1]", "citations": [{"quote": "x", "chunk_id": "c2"}]}]
    assert evaluate.faith_rows(s, 10, 1, engine="synth") == [{"q": "q", "decoding": "synth", "sentence": "Bone fell.", "chunk_id": "c1",
                                              "quote": "Bone volume fell.", "supported": None, "on_topic": None}]


def test_faithfulness_skips_unlabeled_rows(tmp_path, monkeypatch):
    f = tmp_path / "f.jsonl"
    f.write_text('{"decoding": "synth", "supported": true, "on_topic": true}\n{"decoding": "synth", "supported": null, "on_topic": null}\n')
    monkeypatch.setattr(evaluate, "FAITH", f)
    got = {}
    monkeypatch.setattr(evaluate, "dump", lambda n, o: got.update(o))
    evaluate.faithfulness()
    assert got["all"]["sentences"] == 1 and got["all"]["on_topic"] == 1.0


def test_faithfulness_reports_nothing_before_labeling(tmp_path, monkeypatch):
    f = tmp_path / "f.jsonl"
    f.write_text('{"decoding": "synth", "supported": null, "on_topic": null}\n')
    monkeypatch.setattr(evaluate, "FAITH", f)
    monkeypatch.setattr(evaluate, "dump", lambda n, o: (_ for _ in ()).throw(AssertionError("dumped")))
    evaluate.faithfulness()


def test_faith_rows_sample_the_requested_engine():
    s = [{"decoding": "rank", "q": "q", "answer": "Bone fell. [1]", "citations": [{"quote": "Bone fell.", "chunk_id": "c1"}]}]
    assert evaluate.faith_rows(s, 10, 1, engine="rank")[0]["decoding"] == "rank"
    assert evaluate.faith_rows(s, 10, 1, engine="synth") == []



def test_faith_sample_keeps_a_sheet_people_have_filled_in(tmp_path, monkeypatch):
    import labels
    sheet = tmp_path / "faith.csv"
    sheet.write_text("row,supported,on_topic\n1,yes,\n")
    monkeypatch.setattr(labels, "FAITH_CSV", sheet)
    monkeypatch.setattr(evaluate, "FAITH", tmp_path / "missing.jsonl")
    monkeypatch.setattr(evaluate, "faith_rows", lambda *a, **k: (_ for _ in ()).throw(AssertionError("resampled")))
    evaluate.faith_sample()


def test_kappa_is_one_for_identical_labels_and_zero_at_chance():
    assert evaluate.kappa([True, False, True, False], [True, False, True, False]) == pytest.approx(1.0)
    assert evaluate.kappa([True, True, False, False], [True, False, True, False]) == pytest.approx(0.0)


def test_kappa_handles_everyone_saying_yes():
    assert evaluate.kappa([True, True], [True, True]) == 1.0


def test_heldout_summary_compares_a_person_with_a_model():
    rows = [{"q": "Q", "sentence": "a", "on_topic": True}, {"q": "Q", "sentence": "b", "on_topic": False},
            {"q": "Q", "sentence": "c", "on_topic": None}]
    ai = [{"q": "Q", "sentence": "a", "on_topic": True}, {"q": "Q", "sentence": "b", "on_topic": True},
          {"q": "Q", "sentence": "c", "on_topic": True}]
    out = evaluate.heldout_summary(rows, ai)
    assert out["human"]["sentences"] == 2 and out["human"]["on_topic"] == 0.5
    assert out["model"]["sentences"] == 3 and out["model"]["on_topic"] == 1.0
    assert out["agreement"]["sentences"] == 2 and out["agreement"]["same_label"] == 0.5


def test_heldout_summary_before_anyone_has_labeled():
    out = evaluate.heldout_summary([{"q": "Q", "sentence": "a", "on_topic": None}], [])
    assert out == {"human": None, "model": None, "agreement": None}
