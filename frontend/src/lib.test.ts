import { describe, expect, it } from "vitest";
import { agreement, answerLabel, bibtex, cellLabel, cellValue, citationText, filtersFrom, filtersTo, nextStar, shortCite, claims, columnLabel, engineHint, friendlyError, gapQuestion, headlines, highlightParts, imageCaption, isGap, pageTitle, paperUrl, ranked, refusalMessage } from "./lib";

describe("friendlyError", () => {
  it("explains a missing search index", () => {
    expect(friendlyError(503)).toMatch(/index/i);
  });
  it("explains a rejected request", () => {
    expect(friendlyError(422)).toMatch(/question/i);
  });
  it("hides server details on a crash", () => {
    expect(friendlyError(500)).not.toMatch(/500/);
  });
  it("asks people to wait when the server is busy", () => {
    expect(friendlyError(429)).toMatch(/busy/i);
  });
  it("explains an unreachable server", () => {
    expect(friendlyError(0)).toMatch(/reach/i);
  });
});

describe("claims", () => {
  it("gives each sentence its own citation number", () => {
    expect(claims("Bone fell. [1] Muscle shrank. [2]")).toEqual([{ text: "Bone fell", n: 1 }, { text: "Muscle shrank", n: 2 }]);
  });
  it("never changes the case of quoted text", () => {
    expect(claims("mRNA levels fell. [1]")[0].text).toBe("mRNA levels fell");
  });
  it("keeps text with no marks as one uncited claim", () => {
    expect(claims("no marks here")).toEqual([{ text: "no marks here", n: 0 }]);
  });
});

describe("paperUrl", () => {
  it("links to the PMC article", () => {
    expect(paperUrl("PMC123")).toBe("https://pmc.ncbi.nlm.nih.gov/articles/PMC123/");
  });
  it("has no link without an id", () => {
    expect(paperUrl("")).toBeNull();
  });
});

describe("answerLabel", () => {
  it("says ranked answers are direct quotes", () => {
    expect(answerLabel("rank", "greedy")).toBe("quoted from the cited papers");
  });
  it("does not call a passage-based answer generated", () => {
    expect(answerLabel("synth", "greedy")).toBe("drawn from the cited passages");
  });
  it("names the flan-T5 decoding", () => {
    expect(answerLabel("flan", "beam")).toBe("flan-T5 beam · generated");
  });
});

describe("pageTitle", () => {
  it("names each page", () => {
    expect(pageTitle("/ask")).toBe("Ask · CosmoCite");
    expect(pageTitle("/results")).toBe("Results · CosmoCite");
  });
  it("describes the site on the home page", () => {
    expect(pageTitle("/")).toBe("CosmoCite: cited answers from NASA space-biology papers");
  });
  it("marks unknown pages", () => {
    expect(pageTitle("/nope")).toBe("Page not found · CosmoCite");
  });
});

describe("citationText", () => {
  const cites = [{ title: "Bone in orbit", section: "Results", pmc_id: "PMC1" }, { title: "No link", section: "Intro", pmc_id: "" }];
  it("keeps the answer and lists numbered references", () => {
    const text = citationText("How?", "Bone fell. [1] Mass dropped. [2]", cites);
    expect(text).toContain("Q: How?");
    expect(text).toContain("Bone fell. [1] Mass dropped. [2]");
    expect(text).toContain("[1] Bone in orbit (Results). https://pmc.ncbi.nlm.nih.gov/articles/PMC1/");
  });
  it("leaves out a missing link", () => {
    expect(citationText("Q", "A [2]", cites)).toContain("[2] No link (Intro).\n");
  });
});

describe("gapQuestion", () => {
  it("asks about a gap cell in plain words", () => {
    expect(gapQuestion("mouse", "radiation")).toBe("How does radiation affect mice?");
    expect(gapQuestion("cells", "microgravity")).toBe("How does microgravity affect cultured cells?");
  });
  it("handles the catch-all stressor", () => {
    expect(gapQuestion("plant", "other")).toBe("How does spaceflight affect plants?");
  });
});

describe("highlightParts", () => {
  it("marks the quoted span inside the passage", () => {
    expect(highlightParts("One. Two here. Three.", [[5, 14]])).toEqual([{ text: "One. ", hit: false }, { text: "Two here.", hit: true }, { text: " Three.", hit: false }]);
  });
  it("marks each span", () => {
    expect(highlightParts("A one. B two. C three.", [[0, 6], [14, 22]]).filter((p) => p.hit).map((p) => p.text)).toEqual(["A one.", "C three."]);
  });
  it("shows the plain passage when there are no spans", () => {
    expect(highlightParts("Bone fell .", [])).toEqual([{ text: "Bone fell .", hit: false }]);
  });
});

describe("agreement", () => {
  it("counts how many retrievers found each passage", () => {
    const n = agreement({ bm25: [{ chunk_id: "a" }, { chunk_id: "b" }], dense: [{ chunk_id: "a" }] });
    expect(n.get("a")).toBe(2);
    expect(n.get("b")).toBe(1);
  });
  it("counts a retriever once even if it repeats a passage", () => {
    expect(agreement({ bm25: [{ chunk_id: "a" }, { chunk_id: "a" }] }).get("a")).toBe(1);
  });
});

describe("refusalMessage", () => {
  it("says the topic is outside the papers when nothing is close", () => {
    expect(refusalMessage("gate", false)).toMatch(/outside/);
  });
  it("suggests narrowing when close papers exist", () => {
    expect(refusalMessage("gate", true)).toMatch(/organism, tissue or stressor/);
  });
  it("explains a question no sentence answers, without jargon", () => {
    const m = refusalMessage("filter", true);
    expect(m).toMatch(/no sentence/i);
    expect(m).not.toMatch(/grounding|claim/i);
  });
});

describe("engineHint", () => {
  it("adds nothing for the default engine", () => expect(engineHint("rank")).toBe(""));
  it("warns that other engines write their own words", () => {
    expect(engineHint("synth")).toMatch(/Ollama/);
    expect(engineHint("flan")).toMatch(/fragments|comparison/i);
  });
});

describe("result labels", () => {
  it("names known columns in plain words", () => {
    expect(columnLabel("mean_answer_words")).toBe("Average words");
    expect(columnLabel("mrr")).toBe("MRR");
  });
  it("tidies unknown columns", () => expect(columnLabel("some_new_field")).toBe("Some new field"));
  it("keeps the usual spelling of metric names", () => {
    expect(columnLabel("ndcg@5")).toBe("nDCG@5");
    expect(columnLabel("macro_f1")).toBe("Macro F1");
    expect(columnLabel("rouge1")).toBe("ROUGE1");
  });
  it("shows shares as percentages", () => {
    expect(cellValue("correct", 14 / 15)).toBe("93%");
    expect(cellValue("multi_paper_rate", 0.5625)).toBe("56%");
    expect(cellValue("mrr", 0.8912)).toBe("0.891");
    expect(cellValue("questions", 15)).toBe("15");
  });
  it("words booleans and engine names", () => {
    expect(cellValue("stemmed", true)).toBe("Yes");
    expect(cellValue("name", "rank")).toBe("Sentence ranking (default)");
  });
  it("names the engines", () => {
    expect(cellLabel("rank")).toBe("Sentence ranking (default)");
    expect(cellLabel("unknown thing")).toBe("unknown thing");
  });
  it("does not call a bigger cost the best score", () => {
    expect(ranked("recall@10")).toBe(true);
    for (const c of ["p50_ms", "refused_by_gate", "build_s", "errors", "size"]) expect(ranked(c)).toBe(false);
  });
  it("captions the plots", () => {
    expect(imageCaption("cm_bert.png")).toMatch(/SciBERT/);
    expect(imageCaption("new_plot.png")).toBe("new plot");
  });
});

describe("headlines", () => {
  const r = {
    abstention: [{ set: "off-topic", questions: 10, correct: 1 }, { set: "test: unanswerable (should refuse)", questions: 15, correct: 1 },
      { set: "test: answerable (should answer)", questions: 15, correct: 14 / 15 }],
    generation: { rank: { abstention_rate: 0.5, refused_by_gate: 4, refused_by_filter: 1 } },
    relevance_check: { after: { on_topic: 49, sentences: 61 } },
  };
  it("adds up the refusal sets", () => {
    const h = headlines(r);
    expect(h[0].value).toBe("25 of 25");
    expect(h[1].value).toBe("14 of 15");
  });
  it("works out broad coverage from the refusal rate", () => expect(headlines(r)[2].value).toBe("5 of 10"));
  it("marks the relevance figure as an estimate", () => {
    const last = headlines(r)[3];
    expect(last.value).toBe("About 80%");
    expect(last.label).toMatch(/Claude/);
  });
  it("leaves out what the reports do not hold", () => expect(headlines({})).toEqual([]));
});

describe("isGap", () => {
  it("marks cells far below the busiest one", () => {
    expect(isGap(0, 180)).toBe(true);
    expect(isGap(17, 180)).toBe(true);
    expect(isGap(47, 180)).toBe(false);
  });
});

describe("shortCite", () => {
  it("names one, two or many authors", () => {
    expect(shortCite({ authors: ["Smith A"], year: 2014 })).toBe("Smith, 2014");
    expect(shortCite({ authors: ["Smith A", "Jones BC"], year: 2014 })).toBe("Smith & Jones, 2014");
    expect(shortCite({ authors: ["Andreev-Andrievskiy A", "B C", "D E"], year: 2014 })).toBe("Andreev-Andrievskiy et al., 2014");
  });
  it("keeps multi-word surnames", () => expect(shortCite({ authors: ["van der Berg J"], year: 2020 })).toBe("van der Berg, 2020"));
  it("falls back to what is known", () => {
    expect(shortCite({ authors: [], year: 2014 })).toBe("2014");
    expect(shortCite({})).toBe("");
  });
});

describe("citationText with authors", () => {
  it("opens the reference with authors and year", () => {
    const c = [{ title: "T", section: "S", pmc_id: "PMC1", authors: ["Smith A"], year: 2014 }];
    expect(citationText("Q", "A [1]", c)).toContain("[1] Smith, 2014. T (S).");
  });
});

describe("bibtex", () => {
  const c = { title: "Bone {loss}", section: "S", pmc_id: "PMC12", paper_id: "0001", authors: ["Smith A", "Jones B"], year: 2014, journal: "Bone" };
  it("writes one entry with the known fields", () => {
    const b = bibtex([c]);
    expect(b).toContain("@article{PMC12,");
    expect(b).toContain("author = {Smith A and Jones B}");
    expect(b).toContain("title = {Bone loss}");
    expect(b).toContain("year = {2014}");
    expect(b).toContain("url = {https://pmc.ncbi.nlm.nih.gov/articles/PMC12/}");
  });
  it("lists a paper once however many passages cite it", () => {
    expect(bibtex([c, c]).match(/@article/g)).toHaveLength(1);
  });
  it("leaves out unknown fields", () => {
    expect(bibtex([{ title: "T", pmc_id: "", paper_id: "0002" }])).not.toMatch(/author|year|url/);
  });
});

describe("Explore filters in links", () => {
  it("round-trips the filters", () => {
    const f = { organism: "mouse", stressor: "", tissue: "bone", year_min: 2015, q: "loss" };
    expect(filtersFrom(new URLSearchParams(filtersTo(f)))).toEqual(f);
  });
  it("keeps empty filters out of the link", () => {
    expect(filtersTo({ organism: "", stressor: "", tissue: "", year_min: 0, q: "" })).toEqual({});
  });
  it("ignores a bad year", () => expect(filtersFrom(new URLSearchParams("year_min=abc")).year_min).toBe(0));
});

describe("nextStar", () => {
  const pts = [{ x: 0, y: 0 }, { x: 5, y: 0 }, { x: 9, y: 1 }, { x: 0, y: 6 }];
  it("moves to the nearest star in that direction", () => {
    expect(nextStar(pts, 0, [1, 0])).toBe(1);
    expect(nextStar(pts, 1, [1, 0])).toBe(2);
    expect(nextStar(pts, 0, [0, 1])).toBe(3);
  });
  it("stays put at the edge", () => expect(nextStar(pts, 0, [-1, 0])).toBe(0));
});

