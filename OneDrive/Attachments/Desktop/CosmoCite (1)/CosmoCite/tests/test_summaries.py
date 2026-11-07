import summaries

ABSTRACT = ("Spaceflight reduces bone density in mice through increased osteoclast activity. "
            "Bone loss in flown mice was measured with micro computed tomography after thirty days. "
            "The authors thank the crew for their support of this work. "
            "Osteoclast activity and bone density changes were strongest in the femur of flown mice.")


def test_key_sentences_keeps_the_central_sentences():
    out = summaries.key_sentences(ABSTRACT, 2)
    assert "thank the crew" not in out and "Osteoclast activity and bone density" in out


def test_key_sentences_keeps_the_original_order():
    out = summaries.key_sentences(ABSTRACT, 3)
    assert out.index("Spaceflight reduces") < out.index("Osteoclast activity and")


def test_key_sentences_copies_text_exactly():
    for s in summaries.key_sentences(ABSTRACT, 2).split("  "):
        assert s in ABSTRACT


def test_key_sentences_returns_short_text_unchanged():
    assert summaries.key_sentences("One sentence only about bone density in mice.", 2) == "One sentence only about bone density in mice."


def test_key_sentences_handles_empty_text():
    assert summaries.key_sentences("", 2) == ""


def test_add_summaries_uses_the_abstract_then_the_first_section():
    papers = [{"paper_id": "1", "summary": ""}, {"paper_id": "2", "summary": ""}, {"paper_id": "3", "summary": ""}]
    corpus = [{"paper_id": "1", "abstract": ABSTRACT, "sections": []},
              {"paper_id": "2", "abstract": "", "sections": [{"text": ABSTRACT}]},
              {"paper_id": "3", "abstract": "", "sections": []}]
    out = summaries.add_summaries(papers, corpus)
    assert out[0]["summary"] and out[1]["summary"] and out[2]["summary"] == ""
