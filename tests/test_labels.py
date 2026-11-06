import pytest
import labels

FAITH = [{"q": "Bone?", "decoding": "rank", "sentence": "Bone fell.", "chunk_id": "c1", "quote": "Bone fell.",
          "supported": None, "on_topic": None, "passage": "Bone fell in mice."},
         {"q": "Eye?", "decoding": "rank", "sentence": "Eyes changed.", "chunk_id": "c2", "quote": "Eyes changed.",
          "supported": None, "on_topic": None, "passage": "Eyes changed in orbit."}]
POOL = [{"qid": "q1", "question": "Old?", "pool": [{"paper_id": "0001", "title": "A"}]},
        {"qid": "q2", "question": "New?", "pool": [{"paper_id": "0002", "title": "B"}, {"paper_id": "0003", "title": "C"}]}]
QRELS = [{"qid": "q1", "question": "Old?", "judged": ["0001"], "relevant": ["0001"], "label_source": "claude"}]


@pytest.mark.parametrize("text,expected", [("yes", True), ("Y", True), (" no ", False), ("N", False), ("", None), ("  ", None)])
def test_parse_answer_reads_yes_no_and_blank(text, expected):
    assert labels.parse_answer(text) is expected


def test_parse_answer_rejects_anything_unclear():
    with pytest.raises(ValueError, match="maybe"):
        labels.parse_answer("maybe")


def test_faith_sheet_has_one_blank_row_per_sentence():
    rows = labels.faith_sheet(FAITH)
    assert [r["row"] for r in rows] == [1, 2]
    assert rows[0]["question"] == "Bone?" and rows[0]["passage"] == "Bone fell in mice."
    assert rows[0]["supported"] == "" and rows[0]["on_topic"] == ""


def test_apply_faith_writes_answers_to_matching_rows():
    sheet = [{"row": "2", "supported": "yes", "on_topic": "no"}]
    out = labels.apply_faith(FAITH, sheet)
    assert (out[1]["supported"], out[1]["on_topic"]) == (True, False)
    assert out[0]["supported"] is None


def test_apply_faith_leaves_the_input_unchanged():
    labels.apply_faith(FAITH, [{"row": "1", "supported": "yes", "on_topic": "yes"}])
    assert FAITH[0]["supported"] is None


def test_apply_faith_names_the_row_with_a_bad_answer():
    with pytest.raises(ValueError, match="row 2"):
        labels.apply_faith(FAITH, [{"row": "2", "supported": "perhaps", "on_topic": ""}])


def test_apply_faith_rejects_an_unknown_row():
    with pytest.raises(ValueError, match="row 9"):
        labels.apply_faith(FAITH, [{"row": "9", "supported": "yes", "on_topic": "yes"}])


def test_relevance_sheet_lists_only_unjudged_questions():
    rows = labels.relevance_sheet(POOL, QRELS, {"0002": "Abstract B"})
    assert [(r["qid"], r["paper_id"]) for r in rows] == [("q2", "0002"), ("q2", "0003")]
    assert rows[0]["abstract"] == "Abstract B" and rows[1]["abstract"] == "" and rows[0]["relevant"] == ""


def test_apply_relevance_adds_a_human_row_for_a_finished_question():
    sheet = [{"qid": "q2", "paper_id": "0002", "relevant": "yes"}, {"qid": "q2", "paper_id": "0003", "relevant": "no"}]
    out = labels.apply_relevance(QRELS, POOL, sheet)
    assert out[-1] == {"qid": "q2", "question": "New?", "judged": ["0002", "0003"], "relevant": ["0002"], "label_source": "human"}


def test_apply_relevance_skips_a_half_judged_question():
    sheet = [{"qid": "q2", "paper_id": "0002", "relevant": "yes"}, {"qid": "q2", "paper_id": "0003", "relevant": ""}]
    assert labels.apply_relevance(QRELS, POOL, sheet) == QRELS


def test_apply_relevance_keeps_existing_judgments():
    sheet = [{"qid": "q1", "paper_id": "0001", "relevant": "no"}]
    assert labels.apply_relevance(QRELS, POOL, sheet) == QRELS


def test_apply_relevance_names_the_paper_with_a_bad_answer():
    with pytest.raises(ValueError, match="0003"):
        labels.apply_relevance(QRELS, POOL, [{"qid": "q2", "paper_id": "0003", "relevant": "dunno"}])


def test_apply_faith_refuses_labels_for_a_different_sentence():
    with pytest.raises(ValueError, match="row 1"):
        labels.apply_faith(FAITH, [{"row": "1", "sentence": "Something else.", "supported": "yes", "on_topic": "yes"}])
