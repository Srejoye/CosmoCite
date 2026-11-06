import pytest
import generate
from generate import best_quote, compose, parse_claims, select, select_claims, sufficient, support

PASSAGE = ("Spaceflight reduced trabecular bone volume in mice. "
           "Osteoclast activity increased after thirty days in orbit.")


def hit(text=PASSAGE, rerank=5.0, paper="0001"):
    return {"chunk_id": f"{paper}_000", "paper_id": paper, "title": "T", "section": "Results", "text": text,
            "rerank": rerank, "dense": 0.6}


def test_support_measures_words_found_in_the_passage():
    assert support("trabecular bone volume", PASSAGE) == 1.0
    assert support("cardiac output rose", PASSAGE) == 0.0
    assert support("", PASSAGE) == 0.0


def test_select_drops_sentences_not_supported_by_their_own_passage():
    assert select(["Cardiac output rose sharply in astronauts"], [hit()]) == []


def test_select_keeps_grounded_sentence_with_its_source_index():
    other = hit("Plant roots grew slower under simulated gravity conditions.", paper="0002")
    drafts = ["Plant roots grew slower", "Spaceflight reduced trabecular bone volume in mice"]
    assert [i for i, _ in select(drafts, [other, hit()])] == [0, 1]


def test_select_drops_fragments_and_repeats():
    same = "Spaceflight reduced trabecular bone volume in mice"
    assert select(["bone"], [hit()]) == []
    assert len(select([same, same], [hit(), hit(paper="0002")])) == 1


def test_select_ignores_citation_marks_the_model_wrote_itself():
    assert "[" not in select(["Spaceflight reduced trabecular bone volume [7]"], [hit()])[0][1]


def test_compose_only_cites_passages_that_produced_a_sentence():
    hits = [hit(paper="0001"), hit(paper="0002"), hit(paper="0003")]
    text, cites = compose([(2, "Spaceflight reduced trabecular bone volume")], hits)
    assert [c["paper_id"] for c in cites] == ["0003"]
    assert text.endswith("[1]")


def test_compose_marks_always_index_the_returned_citations():
    hits = [hit(paper="0001"), hit(paper="0002")]
    text, cites = compose([(0, "Bone volume fell"), (1, "Osteoclast activity increased")], hits)
    assert "[1]" in text and "[2]" in text and len(cites) == 2


def test_quote_is_a_real_sentence_from_the_passage():
    q = best_quote("osteoclast activity rose", PASSAGE)
    assert q == "Osteoclast activity increased after thirty days in orbit."


def test_gate_compares_the_best_passage_with_the_threshold(monkeypatch):
    monkeypatch.setitem(generate.CFG["retrieval"], "evidence_threshold", 1.0)
    assert sufficient([hit(rerank=0.0), hit(rerank=2.0)])
    assert not sufficient([hit(rerank=0.5)])
    assert not sufficient([])
    assert not sufficient([hit(rerank=None)])


class FakeRetriever:
    def __init__(self, hits):
        self.hits = hits

    def search(self, q, mode="hybrid", k=None):
        return self.hits


def test_answer_abstains_without_calling_the_model_below_threshold(monkeypatch):
    monkeypatch.setitem(generate.CFG["retrieval"], "evidence_threshold", 1.0)
    monkeypatch.setattr(generate, "draft", lambda *a: (_ for _ in ()).throw(AssertionError("model was called")))
    out = generate.answer("q", retriever=FakeRetriever([hit(rerank=0.0)]))
    assert out["sufficient"] is False and out["citations"] == []


def test_answer_abstains_when_no_draft_survives(monkeypatch):
    monkeypatch.setitem(generate.CFG["retrieval"], "evidence_threshold", 1.0)
    monkeypatch.setattr(generate, "draft", lambda q, hits, d: ["Cardiac output rose sharply"] * len(hits))
    out = generate.answer("q", retriever=FakeRetriever([hit()]))
    assert out["sufficient"] is False and out["answer"] == generate.NO_EVIDENCE


def test_answer_returns_cited_sentence(monkeypatch):
    monkeypatch.setitem(generate.CFG["retrieval"], "evidence_threshold", 1.0)
    monkeypatch.setattr(generate, "draft", lambda q, hits, d: ["Spaceflight reduced trabecular bone volume"] * len(hits))
    out = generate.answer("q", retriever=FakeRetriever([hit()]))
    assert out["sufficient"] is True and out["citations"][0]["quote"] and "[1]" in out["answer"]


def test_parse_claims_reads_marks_before_sentences():
    assert parse_claims("[1] Bone fell. [2] Muscle shrank.", 3) == [(0, "Bone fell."), (1, "Muscle shrank.")]


def test_parse_claims_reads_marks_after_sentences():
    assert parse_claims("Bone fell. [1] Muscle shrank. [2]", 3) == [(0, "Bone fell."), (1, "Muscle shrank.")]


def test_parse_claims_splits_several_sentences_under_one_mark():
    assert parse_claims("[2] Bone fell. Muscle shrank.", 3) == [(1, "Bone fell."), (1, "Muscle shrank.")]


def test_parse_claims_drops_passages_that_do_not_exist():
    assert parse_claims("[9] Bone fell. [0] Muscle shrank.", 3) == []


def test_parse_claims_returns_nothing_without_marks():
    assert parse_claims("Bone fell in mice.", 3) == []


def test_select_claims_caps_the_answer_at_max_claims():
    lines = ["Bone volume fell sharply", "Muscle fibers shrank quickly", "Plant roots curved upward",
             "Heart rate rose slightly", "Liver enzymes dropped steadily"]
    claims = [(0, s) for s in lines]
    assert len(select_claims(claims, [hit(". ".join(lines))])) == generate.MAX_CLAIMS


def test_compose_gives_one_number_to_sentences_from_one_passage():
    text, cites = compose([(0, "Bone volume fell"), (0, "Osteoclast activity increased")], [hit()])
    assert len(cites) == 1 and text.count("[1]") == 2


def test_on_topic_is_a_no_op_without_a_threshold(monkeypatch):
    monkeypatch.setitem(generate.CFG["retrieval"], "claim_threshold", None)
    assert generate.on_topic("q", [(0, "Bone fell")]) == [(0, "Bone fell")]


def test_on_topic_drops_claims_below_the_threshold(monkeypatch):
    monkeypatch.setitem(generate.CFG["retrieval"], "claim_threshold", 0.0)
    monkeypatch.setattr(generate, "_claim_scores", lambda q, s: [1.0, -1.0])
    assert generate.on_topic("q", [(0, "A b c"), (0, "D e f")]) == [(0, "A b c")]


def test_synth_answers_from_all_passages(monkeypatch):
    monkeypatch.setitem(generate.CFG["retrieval"], "evidence_threshold", 1.0)
    monkeypatch.setattr(generate.llm, "complete", lambda p: "[1] Spaceflight reduced trabecular bone volume in mice.")
    out = generate.answer("q", retriever=FakeRetriever([hit()]), engine="synth")
    assert out["sufficient"] and out["engine"] == "synth" and out["reason"] is None and "[1]" in out["answer"]


def test_uncited_model_output_is_a_filter_refusal(monkeypatch):
    monkeypatch.setitem(generate.CFG["retrieval"], "evidence_threshold", 1.0)
    monkeypatch.setattr(generate.llm, "complete", lambda p: "Spaceflight reduced trabecular bone volume in mice.")
    out = generate.answer("q", retriever=FakeRetriever([hit()]), engine="synth")
    assert out["sufficient"] is False and out["reason"] == "filter" and out["citations"] == []


def test_ollama_outage_falls_back_to_flan(monkeypatch):
    monkeypatch.setitem(generate.CFG["retrieval"], "evidence_threshold", 1.0)

    def down(p):
        raise generate.llm.LLMUnavailable("down")

    monkeypatch.setattr(generate.llm, "complete", down)
    monkeypatch.setattr(generate, "draft", lambda q, hits, d: ["Spaceflight reduced trabecular bone volume"] * len(hits))
    out = generate.answer("q", retriever=FakeRetriever([hit()]), engine="synth")
    assert out["sufficient"] and out["engine"] == "flan"


def test_weak_evidence_refuses_with_gate_reason(monkeypatch):
    monkeypatch.setitem(generate.CFG["retrieval"], "evidence_threshold", 1.0)
    out = generate.answer("q", retriever=FakeRetriever([hit(rerank=0.0)]), engine="synth")
    assert out["reason"] == "gate"


def test_unknown_engine_is_rejected():
    with pytest.raises(ValueError):
        generate.answer("q", retriever=FakeRetriever([hit()]), engine="psychic")


def test_answer_reports_no_evidence_on_refusal(monkeypatch):
    monkeypatch.setitem(generate.CFG["retrieval"], "evidence_threshold", 1.0)
    assert generate.answer("q", retriever=FakeRetriever([hit(rerank=0.0)]))["evidence"] is None


def test_answer_reports_evidence_when_it_answers(monkeypatch):
    monkeypatch.setitem(generate.CFG["retrieval"], "evidence_threshold", 1.0)
    monkeypatch.setattr(generate, "draft", lambda q, hits, d: ["Spaceflight reduced trabecular bone volume"] * len(hits))
    monkeypatch.setattr(generate, "_papers", lambda: [{"paper_id": "0001", "year": 2014, "entities": {"organism": [], "stressor": [], "tissue": []}}])
    assert generate.answer("q", retriever=FakeRetriever([hit()]))["evidence"]["papers"] == 1


def test_citations_carry_the_paper_link_id():
    _, cites = compose([(0, "Bone volume fell")], [{**hit(), "pmc_id": "PMC123"}])
    assert cites[0]["pmc_id"] == "PMC123"


def test_citations_tolerate_a_missing_paper_link():
    _, cites = compose([(0, "Bone volume fell")], [hit()])
    assert cites[0]["pmc_id"] == ""


TWO = "Plant roots grew slower under simulated gravity conditions. Unrelated filler text sits here today."


def test_diverse_takes_one_passage_per_paper_first(monkeypatch):
    hits = [hit(paper="0001"), hit(paper="0001"), hit(paper="0002"), hit(paper="0003")]
    assert [h["paper_id"] for h in generate.diverse(hits, 3, 1.0)] == ["0001", "0002", "0003"]


def test_diverse_ignores_passages_below_the_evidence_threshold():
    hits = [hit(paper="0001"), hit(paper="0002", rerank=0.5)]
    assert [h["paper_id"] for h in generate.diverse(hits, 3, 1.0)] == ["0001"]


def test_diverse_fills_from_the_same_paper_when_few_papers_qualify():
    hits = [hit(paper="0001"), hit(paper="0001")]
    assert len(generate.diverse(hits, 3, 1.0)) == 2


def test_extract_claim_picks_the_sentence_closest_to_the_question():
    assert generate.extract_claim("How do plant roots grow?", {"text": TWO}) == "Plant roots grew slower under simulated gravity conditions."


def test_extract_claim_returns_nothing_without_shared_words():
    assert generate.extract_claim("cardiac output", {"text": TWO}) is None


def test_balance_adds_a_sentence_from_each_uncited_passage():
    hits = [hit(), hit(TWO, paper="0002")]
    kept = generate.balance("plant roots bone", [(0, "Spaceflight reduced trabecular bone volume in mice")], hits)
    assert sorted({i for i, _ in kept}) == [0, 1]


def test_balance_limits_how_much_one_passage_can_say():
    claims = [(0, "Bone volume fell sharply"), (0, "Muscle fibers shrank quickly"), (0, "Plant roots curved upward")]
    text = ". ".join(s for _, s in claims)
    assert len(generate.balance("q", claims, [hit(text)])) == generate.MAX_PER_PASSAGE


def test_balance_rescues_an_answer_when_the_model_fails():
    assert generate.balance("plant roots", [], [hit(TWO)]) != []


def test_unusable_model_output_falls_back_to_extracted_sentences(monkeypatch):
    monkeypatch.setitem(generate.CFG["retrieval"], "evidence_threshold", 1.0)
    monkeypatch.setattr(generate.llm, "complete", lambda p: "(iv)")
    out = generate.answer("trabecular bone volume in mice", retriever=FakeRetriever([hit()]), engine="synth")
    assert out["sufficient"] and "[1]" in out["answer"] and out["engine"] == "synth"


def test_answer_cites_several_papers_when_several_qualify(monkeypatch):
    monkeypatch.setitem(generate.CFG["retrieval"], "evidence_threshold", 1.0)
    monkeypatch.setattr(generate.llm, "complete", lambda p: "[1] Spaceflight reduced trabecular bone volume in mice.")
    hits = [hit(), hit(paper="0001"), hit(TWO, paper="0002")]
    out = generate.answer("plant roots and bone volume", retriever=FakeRetriever(hits), engine="synth")
    assert {c["paper_id"] for c in out["citations"]} == {"0001", "0002"}


def test_weak_evidence_still_refuses_even_with_extraction(monkeypatch):
    monkeypatch.setitem(generate.CFG["retrieval"], "evidence_threshold", 1.0)
    out = generate.answer("trabecular bone", retriever=FakeRetriever([hit(rerank=0.2)]), engine="synth")
    assert out["sufficient"] is False and out["reason"] == "gate"


def test_citation_quotes_every_sentence_that_used_the_passage():
    _, cites = compose([(0, "Spaceflight reduced bone volume"), (0, "Osteoclast activity increased in orbit")], [hit()])
    assert "trabecular" in cites[0]["quote"] and "Osteoclast" in cites[0]["quote"]


def test_citation_does_not_repeat_a_shared_quote():
    _, cites = compose([(0, "Spaceflight reduced bone volume"), (0, "Spaceflight reduced trabecular bone")], [hit()])
    assert cites[0]["quote"].count("Spaceflight") == 1


def test_select_drops_a_claim_that_adds_a_negation():
    assert select(["Spaceflight did not reduce trabecular bone volume in mice"], [hit()]) == []


def test_select_drops_a_claim_that_loses_a_negation():
    text = "Spaceflight did not reduce trabecular bone volume in mice."
    assert select(["Spaceflight reduced trabecular bone volume in mice"], [hit(text)]) == []


def test_select_keeps_a_claim_that_copies_a_negation():
    text = "Spaceflight did not reduce trabecular bone volume in mice."
    assert len(select([text], [hit(text)])) == 1


LONG_A = "Spaceflight reduced trabecular bone volume in the femur of adult mice after thirty days."
LONG_B = "Osteoclast activity increased sharply in the pelvis of flown mice during the same mission."
OTHER = "Plant roots grew noticeably slower under simulated gravity conditions in the growth chamber."


def scored(monkeypatch, table):
    monkeypatch.setattr(generate, "_claim_scores", lambda q, claims: [table.get(c, 0.0) for c in claims])


def test_rank_claims_puts_the_best_sentence_first(monkeypatch):
    scored(monkeypatch, {LONG_A: 5.0, OTHER: 9.0})
    hits = [hit(LONG_A), hit(OTHER, paper="0002")]
    assert generate.rank_claims("q", hits, 1.0) == [(1, OTHER), (0, LONG_A)]


def test_rank_claims_takes_one_sentence_per_paper(monkeypatch):
    scored(monkeypatch, {LONG_A: 9.0, LONG_B: 8.0, OTHER: 5.0})
    hits = [hit(LONG_A + " " + LONG_B), hit(OTHER, paper="0002")]
    assert [s for _, s in generate.rank_claims("q", hits, 1.0)] == [LONG_A, OTHER]


def test_rank_claims_drops_sentences_below_the_bar(monkeypatch):
    scored(monkeypatch, {LONG_A: 9.0, OTHER: 0.5})
    hits = [hit(LONG_A), hit(OTHER, paper="0002")]
    assert generate.rank_claims("q", hits, 1.0) == [(0, LONG_A)]


def test_rank_claims_skips_fragments_and_walls_of_text(monkeypatch):
    wall = " ".join(["word"] * 80) + "."
    scored(monkeypatch, {"Too short here.": 9.0, wall: 9.0, LONG_A: 5.0})
    assert generate.rank_claims("q", [hit("Too short here. " + wall + " " + LONG_A)], 1.0) == [(0, LONG_A)]


def test_rank_claims_stops_at_the_answer_limit(monkeypatch):
    scored(monkeypatch, {LONG_A: 9.0, LONG_B: 8.0, OTHER: 7.0})
    fourth = "Heart rate variability declined steadily across every crew member on the long flight."
    scored(monkeypatch, {LONG_A: 9.0, LONG_B: 8.0, OTHER: 7.0, fourth: 6.0})
    hits = [hit(s, paper=p) for s, p in ((LONG_A, "1"), (LONG_B, "2"), (OTHER, "3"), (fourth, "4"))]
    assert len(generate.rank_claims("q", hits, 1.0)) == generate.RANK_CLAIMS


def test_rank_claims_handles_no_usable_sentences():
    assert generate.rank_claims("q", [hit("Too short.")], 1.0) == []


def test_rank_engine_answers_without_any_language_model(monkeypatch):
    monkeypatch.setitem(generate.CFG["retrieval"], "evidence_threshold", 1.0)
    scored(monkeypatch, {LONG_A: 5.0})
    monkeypatch.setattr(generate.llm, "complete", lambda p: (_ for _ in ()).throw(AssertionError("model was called")))
    monkeypatch.setattr(generate, "draft", lambda *a: (_ for _ in ()).throw(AssertionError("model was called")))
    out = generate.answer("q", retriever=FakeRetriever([hit(LONG_A)]), engine="rank")
    assert out["sufficient"] and out["engine"] == "rank" and "[1]" in out["answer"] and out["citations"][0]["quote"] == LONG_A


def test_rank_claims_always_keeps_the_best_sentence(monkeypatch):
    scored(monkeypatch, {LONG_A: 0.2})
    assert generate.rank_claims("q", [hit(LONG_A)], 1.0) == [(0, LONG_A)]


def test_rank_engine_refuses_when_no_sentence_is_usable(monkeypatch):
    monkeypatch.setitem(generate.CFG["retrieval"], "evidence_threshold", 1.0)
    out = generate.answer("q", retriever=FakeRetriever([hit("Too short.")]), engine="rank")
    assert out["sufficient"] is False and out["reason"] == "filter"


def test_rank_engine_uses_the_sentence_bar_for_extra_sentences(monkeypatch):
    monkeypatch.setitem(generate.CFG["retrieval"], "evidence_threshold", 1.0)
    monkeypatch.setitem(generate.CFG["retrieval"], "sentence_threshold", 6.0)
    scored(monkeypatch, {LONG_A: 9.0, OTHER: 5.0})
    out = generate.answer("q", retriever=FakeRetriever([hit(LONG_A), hit(OTHER, paper="0002")]), engine="rank")
    assert [c["paper_id"] for c in out["citations"]] == ["0001"]


def test_rank_engine_still_obeys_the_evidence_gate(monkeypatch):
    monkeypatch.setitem(generate.CFG["retrieval"], "evidence_threshold", 1.0)
    scored(monkeypatch, {LONG_A: 9.0})
    out = generate.answer("q", retriever=FakeRetriever([hit(LONG_A, rerank=0.2)]), engine="rank")
    assert out["reason"] == "gate"



def test_tidy_closes_gaps_left_by_removed_citations():
    assert generate.tidy("Dose, and duration of exposure .") == "Dose, and duration of exposure."
    assert generate.tidy("Bone ( Table 1 ) fell , then rose .") == "Bone (Table 1) fell, then rose."


def test_tidy_removes_stray_reference_lists():
    assert generate.tidy("Shifts in metabolism 9 , 10 .") == "Shifts in metabolism."
    assert generate.tidy("Shifts in metabolism 9 , 10 , 12 .") == "Shifts in metabolism."


def test_tidy_keeps_real_numbers():
    assert generate.tidy("Mice received a dose of 10 .") == "Mice received a dose of 10."
    assert generate.tidy("Doses of 5, 10 and 20 Gy were used.") == "Doses of 5, 10 and 20 Gy were used."


def test_ranked_answers_and_quotes_are_tidy(monkeypatch):
    monkeypatch.setitem(generate.CFG["retrieval"], "evidence_threshold", 1.0)
    raw = "Spaceflight reduced trabecular bone volume in adult mice over thirty days 9 , 10 ."
    scored(monkeypatch, {raw: 5.0})
    out = generate.answer("trabecular bone", retriever=FakeRetriever([hit(raw)]), engine="rank")
    assert "9 ," not in out["answer"] and "9 ," not in out["citations"][0]["quote"]


def test_sentences_do_not_split_on_abbreviations():
    text = "Effects were strong (e.g., refs. 3 and 4) in mice (Supplementary Fig. 2). A second sentence follows."
    assert len(generate.sentences(text)) == 2


def test_whole_sentence_rejects_fragments():
    assert generate.whole_sentence("Bone volume fell sharply in flown mice.")
    assert generate.whole_sentence("16S sequencing showed a shift in gut bacteria.")
    assert not generate.whole_sentence("showed that a combined exposure reduced bone volume.")
    assert not generate.whole_sentence("Bone volume fell sharply (see the methods.")
    assert not generate.whole_sentence("Bone volume fell sharply in flown mice")


def test_rank_claims_skips_fragments(monkeypatch):
    bad = "showed that a combined exposure reduced bone volume across every flown mouse."
    scored(monkeypatch, {bad: 9.0, LONG_A: 5.0})
    assert generate.rank_claims("q", [hit(bad + " " + LONG_A)], 1.0) == [(0, LONG_A)]


def test_closest_papers_lists_each_paper_once_in_rank_order():
    hits = [hit(paper="0001", rerank=3.0), hit(paper="0001", rerank=2.0), hit(paper="0002", rerank=1.0)]
    assert [p["paper_id"] for p in generate.closest_papers(hits, 0.0)] == ["0001", "0002"]


def test_closest_papers_stop_at_the_floor_and_the_limit():
    hits = [hit(paper=str(i), rerank=5.0 - i * 0.5) for i in range(6)]
    assert len(generate.closest_papers(hits, 0.0)) == generate.CLOSEST_PAPERS
    assert generate.closest_papers([hit(rerank=-1.0)], 0.0) == []


def test_closest_papers_ignore_unscored_hits():
    assert generate.closest_papers([hit(rerank=None)], 0.0) == []


def test_refusal_offers_closest_papers_without_answering(monkeypatch):
    monkeypatch.setitem(generate.CFG["retrieval"], "evidence_threshold", 4.0)
    monkeypatch.setitem(generate.CFG["retrieval"], "closest_floor", 0.0)
    out = generate.answer("q", retriever=FakeRetriever([hit(rerank=3.0)]), engine="rank")
    assert out["sufficient"] is False and out["citations"] == [] and out["answer"] == generate.NO_EVIDENCE
    assert out["closest"][0]["paper_id"] == "0001"


def test_off_topic_refusals_offer_nothing(monkeypatch):
    monkeypatch.setitem(generate.CFG["retrieval"], "evidence_threshold", 4.0)
    monkeypatch.setitem(generate.CFG["retrieval"], "closest_floor", 0.0)
    assert generate.answer("q", retriever=FakeRetriever([hit(rerank=-5.0)]), engine="rank")["closest"] == []


def test_answers_offer_no_closest_list(monkeypatch):
    monkeypatch.setitem(generate.CFG["retrieval"], "evidence_threshold", 1.0)
    scored(monkeypatch, {LONG_A: 5.0})
    assert generate.answer("q", retriever=FakeRetriever([hit(LONG_A)]), engine="rank")["closest"] == []


def test_citations_carry_the_full_passage():
    _, cites = compose([(0, "Bone volume fell")], [hit()])
    assert cites[0]["passage"] == PASSAGE


def test_citation_spans_locate_the_quote_even_when_tidying_changed_it():
    text = "Intro sentence stands here first. Bone loss of 30% is noted 3 , 11 . Later text follows on."
    _, cites = compose([(0, "Bone loss of 30% is noted")], [hit(text)])
    (a, b), = cites[0]["spans"]
    assert cites[0]["quote"] == "Bone loss of 30% is noted."
    assert text[a:b] == "Bone loss of 30% is noted 3 , 11 ."


def test_citation_spans_follow_the_passage_order():
    text = "Bone volume fell sharply in mice. Muscle mass dropped in the same mice."
    _, cites = compose([(0, "Muscle mass dropped"), (0, "Bone volume fell")], [hit(text)])
    assert [text[a:b] for a, b in cites[0]["spans"]] == ["Bone volume fell sharply in mice.", "Muscle mass dropped in the same mice."]


def test_tidy_drops_a_comma_stranded_before_the_full_stop():
    assert generate.tidy("It affects the immune system , .") == "It affects the immune system."


BONE_Q = "How does microgravity affect bone in mice?"


@pytest.mark.parametrize("sentence", [
    "The purpose of this study was to measure bone loss in microgravity in mice.",
    "We subjected mice to microgravity and then performed scans of bone tissue.",
    "Little is known about how microgravity changes bone in older mice.",
    "It remains unclear how microgravity alters bone remodeling in mice.",
])
def test_aims_and_open_questions_do_not_count_as_findings(sentence):
    assert not generate.addresses(BONE_Q, sentence)


def test_a_generic_list_of_body_systems_does_not_answer():
    s = "In mice, microgravity can influence several body systems such as the cardiovascular, bone and immune system."
    assert not generate.addresses(BONE_Q, s)


def test_a_finding_on_the_asked_tissue_and_stressor_answers():
    assert generate.addresses(BONE_Q, "Microgravity reduced trabecular bone volume in the femur of flown mice.")


def test_a_sentence_about_another_tissue_does_not_answer():
    assert not generate.addresses(BONE_Q, "Microgravity reduced heart mass in the flown mice.")


def test_a_sentence_about_another_stressor_does_not_answer():
    assert not generate.addresses("How does radiation exposure affect bone in mice?", "Microgravity reduced bone volume in flown mice.")


def test_related_terms_count_as_the_tissue():
    assert generate.addresses("How does microgravity affect gut in mice?", "Microgravity altered the gastrointestinal tract of mice.")
    assert generate.addresses("How does radiation exposure affect immune in mice?", "Radiation lowered cytokines after low doses.")
    assert generate.addresses("How does radiation exposure affect brain in mice?", "Radiation damaged mice brains within days.")


def test_questions_naming_no_tissue_or_stressor_are_not_narrowed():
    assert generate.addresses("Which Arabidopsis mutants tolerate magnesium sulfate?", "Two mutants grew well on the salt-rich soil tested here.")


def test_rank_claims_passes_over_a_higher_scored_sentence_that_does_not_answer(monkeypatch):
    method = "We subjected mice to microgravity and then performed scans of bone tissue."
    finding = "Microgravity reduced trabecular bone volume in the femur of flown mice."
    scored(monkeypatch, {method: 9.0, finding: 5.0})
    kept = generate.rank_claims(BONE_Q, [hit(method + " " + finding)], 3.0)
    assert [s for _, s in kept] == [finding]


def test_refuses_when_no_sentence_answers_the_question(monkeypatch):
    monkeypatch.setitem(generate.CFG["retrieval"], "evidence_threshold", 1.0)
    monkeypatch.setitem(generate.CFG["retrieval"], "closest_floor", 0.0)
    method = "We subjected mice to microgravity and then performed scans of bone tissue."
    scored(monkeypatch, {method: 9.0})
    out = generate.answer(BONE_Q, retriever=FakeRetriever([hit(method)]), engine="rank")
    assert out["sufficient"] is False and out["reason"] == "filter" and out["closest"][0]["paper_id"] == "0001"


@pytest.mark.parametrize("sentence", [
    "This study explored the impact of microgravity on bone in mice.",
    "We chose to focus on bone data from mice in microgravity.",
    "It is, therefore, unclear how microgravity influences bone metabolism in mice.",
])
def test_more_ways_of_stating_an_aim_or_open_question(sentence):
    assert not generate.addresses(BONE_Q, sentence)


CELL_Q = "How does microgravity affect liver in cultured cells?"


def test_a_cell_question_needs_a_sentence_about_cells():
    assert not generate.addresses(CELL_Q, "Microgravity reduced lipoprotein synthesis in the liver of flown mice.")


@pytest.mark.parametrize("sentence", ["Microgravity reduced lipid storage in cultured liver cells.", "Hepatocytes grown in microgravity stored less lipid in the liver."])
def test_a_cell_question_accepts_cells_by_any_name(sentence):
    assert generate.addresses(CELL_Q, sentence)


@pytest.mark.parametrize("sentence", [
    "The mechanisms of bone loss in microgravity in mice are currently unknown.",
    "The effect of microgravity on bone growth in mice is presented in.",
    "We hypothesised that microgravity would change bone in mice.",
])
def test_unknowns_pointers_and_hypotheses_are_not_findings(sentence):
    assert not generate.addresses(BONE_Q, sentence)


def test_everyday_words_count_as_the_tissue_and_the_stressor():
    assert generate.addresses("How does microgravity affect the eye?", "Astronauts reported impaired visual acuity during their mission.")
    assert generate.addresses("How does spaceflight affect bone?", "Gravity changes altered bone growth in the tested animals.")
    assert generate.addresses("How does spaceflight affect the heart?", "Heart rate rose in-flight compared with the ground.")


def test_space_words_do_not_stand_in_for_radiation():
    assert not generate.addresses("How does radiation exposure affect bone in mice?", "Astronauts lost bone on long missions.")


@pytest.mark.parametrize("sentence", ["To determine how microgravity changes bone in mice, femurs were scanned.", "To explore whether microgravity alters bone in mice, we analyzed five datasets."])
def test_a_sentence_opening_with_its_aim_is_not_a_finding(sentence):
    assert not generate.addresses(BONE_Q, sentence)


@pytest.mark.parametrize("sentence", ["IR exposure alone reduced kidney weights in the treated mice.", "A 2 Gy dose reduced kidney weights in the treated mice."])
def test_common_shorthand_counts_as_radiation(sentence):
    assert generate.addresses("How does radiation exposure affect kidney in mice?", sentence)


def test_lowercase_ir_inside_words_is_not_radiation():
    assert not generate.addresses("How does radiation exposure affect kidney in mice?", "Kidney weights fell in the air-fed mice.")


def test_citations_carry_year_authors_and_journal():
    h = {**hit(), "year": 2014, "authors": ["Smith A"], "journal": "Bone"}
    _, cites = compose([(0, "Bone volume fell")], [h])
    assert (cites[0]["year"], cites[0]["authors"], cites[0]["journal"]) == (2014, ["Smith A"], "Bone")


HUMAN_Q = "How does microgravity affect bone in humans?"
MOUSE_S = "Microgravity reduced trabecular bone volume in the femur of flown mice."
HUMAN_S = "Microgravity reduced bone density in the hips of long-duration astronauts."
PLAIN_S = "Microgravity reduced bone density across the weight-bearing skeleton on long missions."


def test_a_study_of_another_organism_is_recognised():
    assert generate.other_organism(HUMAN_Q, MOUSE_S)
    assert generate.other_organism(HUMAN_Q, "Animal studies showed that microgravity reduced bone mass.")
    assert generate.other_organism("How does microgravity affect bone in mice?", HUMAN_S)


@pytest.mark.parametrize("sentence", [HUMAN_S, PLAIN_S, "Both mice and humans lost bone in microgravity."])
def test_matching_general_or_shared_statements_are_not_other_organisms(sentence):
    assert not generate.other_organism(HUMAN_Q, sentence)


def test_animal_studies_still_count_for_a_question_about_mice():
    assert not generate.other_organism("How does microgravity affect bone in mice?", "Animal studies showed that microgravity reduced bone mass.")


def test_questions_naming_no_organism_prefer_nothing():
    assert not generate.other_organism("How does microgravity affect bone?", MOUSE_S)


def test_an_on_organism_sentence_beats_a_higher_scored_other_organism(monkeypatch):
    scored(monkeypatch, {MOUSE_S: 9.0, HUMAN_S: 5.0})
    kept = generate.rank_claims(HUMAN_Q, [hit(MOUSE_S, paper="0001"), hit(HUMAN_S, paper="0002")], 3.0)
    assert [s for _, s in kept] == [HUMAN_S]


def test_other_organism_evidence_is_still_used_when_nothing_better_clears_the_bar(monkeypatch):
    scored(monkeypatch, {MOUSE_S: 9.0, HUMAN_S: 1.0})
    kept = generate.rank_claims(HUMAN_Q, [hit(MOUSE_S, paper="0001"), hit(HUMAN_S, paper="0002")], 3.0)
    assert kept[0][1] == MOUSE_S


def test_a_microbiome_is_not_a_different_organism():
    assert not generate.other_organism("How does microgravity affect gut in humans?", "Microgravity caused an imbalance in the gut microbiota.")


def test_rodents_count_as_mice():
    s = "Microgravity may induce hepatic lipotoxicity, previously documented in rodents but needing study in humans."
    assert not generate.other_organism("How does microgravity affect liver in mice?", s)

