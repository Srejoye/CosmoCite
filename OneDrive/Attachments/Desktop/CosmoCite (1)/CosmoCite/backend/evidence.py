"""How much of the corpus backs an answer: every paper with a passage above the evidence threshold."""
from collections import Counter


def evidence_profile(hits, papers, threshold):
    by_id = {p["paper_id"]: p for p in papers}
    ids = {h["paper_id"] for h in hits if h["rerank"] is not None and h["rerank"] >= threshold}
    found = [by_id[i] for i in sorted(ids) if i in by_id]
    if not found:
        return None
    years = [p["year"] for p in found if p["year"]]
    count = lambda key: dict(Counter(v for p in found for v in p["entities"][key]).most_common())
    return {"papers": len(found), "year_min": min(years, default=None), "year_max": max(years, default=None),
            "organisms": count("organism"), "stressors": count("stressor"), "tissues": count("tissue")}
