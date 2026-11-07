/** Plain-language message for a failed request. Status 0 means the server could not be reached. */
export function friendlyError(status: number): string {
  if (status === 0) return "Can't reach the CosmoCite server. Check that the backend is running, then try again.";
  if (status === 503) return "That isn't available on this server right now. The search index or answer engine may be missing.";
  if (status === 429) return "The server is busy answering other questions. Please try again in a moment.";
  if (status === 422) return "That request wasn't valid. Try a shorter question.";
  return "Something went wrong on the server. Please try again.";
}

/** "claim. [1] claim. [2]" -> [{text, n}] so each sentence can carry its own citation chip. */
export function claims(answer: string) {
  const found = [...answer.matchAll(/(.+?)\s*\[(\d+)\]/g)].map((m) => ({ text: m[1].trim().replace(/\.+$/, ""), n: Number(m[2]) }));
  return found.length ? found : [{ text: answer, n: 0 }]; // quoted text keeps its case: mRNA must not become MRNA
}

/** Says what produced an answer, so readers know whether they are seeing quotes or generated text. */
export function answerLabel(engine: string, decoding: string) {
  if (engine === "rank") return "quoted from the cited papers";
  return engine === "synth" ? "drawn from the cited passages" : `flan-T5 ${decoding} · generated`;
}

export const paperUrl = (pmcId: string) => (pmcId ? `https://pmc.ncbi.nlm.nih.gov/articles/${pmcId}/` : null);

const PAGES: Record<string, string> = { "/ask": "Ask", "/search": "Search", "/explore": "Explore", "/analyze": "Analyze", "/results": "Results" };

export function pageTitle(path: string) {
  if (path === "/") return "CosmoCite: cited answers from NASA space-biology papers";
  return `${PAGES[path] ?? "Page not found"} · CosmoCite`;
}

/** An answer with numbered references, ready to paste into notes. */
export function citationText(question: string, answer: string, citations: ({ title: string; section: string; pmc_id: string } & Byline)[]) {
  const refs = citations.map((c, i) => {
    const url = paperUrl(c.pmc_id), who = shortCite(c);
    return `[${i + 1}] ${who ? who + ". " : ""}${c.title} (${c.section}).${url ? " " + url : ""}\n`;
  });
  return `Q: ${question}\n\n${answer}\n\nReferences\n${refs.join("")}`;
}

const ORGANISMS: Record<string, string> = { mouse: "mice", human: "humans", plant: "plants", cells: "cultured cells" };

/** Turns a gap-map cell into a question for the Ask page. */
export const gapQuestion = (organism: string, stressor: string) =>
  `How does ${stressor === "other" ? "spaceflight" : stressor} affect ${ORGANISMS[organism] ?? organism}?`;

/** Splits a passage at the offsets the server gives for each quoted sentence. */
export function highlightParts(passage: string, spans: [number, number][]) {
  const parts: { text: string; hit: boolean }[] = [];
  let at = 0;
  for (const [start, end] of spans) {
    if (start > at) parts.push({ text: passage.slice(at, start), hit: false });
    parts.push({ text: passage.slice(start, end), hit: true });
    at = end;
  }
  if (at < passage.length) parts.push({ text: passage.slice(at), hit: false });
  return parts;
}

/** How many retrievers found each passage. Agreement is the useful part of comparing them, because their scores use different scales. */
export function agreement(res: Record<string, { chunk_id: string }[]>) {
  const found = new Map<string, number>();
  for (const hits of Object.values(res)) {
    for (const id of new Set(hits.map((h) => h.chunk_id))) found.set(id, (found.get(id) ?? 0) + 1);
  }
  return found;
}

/** Says why a question was refused, in terms of what the reader can do next. */
export function refusalMessage(reason: string | null, hasLeads: boolean) {
  if (reason === "filter") return "Papers on this topic exist, but no sentence in them states a finding about what you asked. Try rewording it, or read the closest papers.";
  if (hasLeads) return "Some papers come close, but none is strong enough to answer this. Try naming a specific organism, tissue or stressor.";
  return "This looks outside NASA's space-biology papers, so CosmoCite stopped rather than guess.";
}

export function engineHint(engine: string) {
  if (engine === "synth") return "Writes its own sentences from the passages, so check them against the sources. Needs Ollama running on this computer.";
  if (engine === "flan") return "A small model kept for comparison. It often returns fragments, so use it to compare decoding, not for answers.";
  return "";
}

const COLUMNS: Record<string, string> = {
  name: "Model", mode: "Retriever", set: "Question set", mrr: "MRR", correct: "Correct", questions: "Questions", kind: "Index", size: "Vectors", data: "Data",
  build_s: "Build (s)", p50_ms: "Median (ms)", p95_ms: "Slowest 5% (ms)", p50_s: "Median (s)", p95_s: "Slowest 5% (s)", per_second: "Answers per second",
  abstention_rate: "Refusal rate", refused_by_gate: "Refused: weak evidence", refused_by_filter: "Refused: no on-topic sentence", fallbacks: "Fallbacks",
  mean_answer_words: "Average words", mean_citations: "Average citations", multi_paper_rate: "Cites 2+ papers", cited_silver_paper: "Cites expected paper",
  stopwords_removed: "Stop words removed", stemmed: "Stemmed", people: "People at once", served: "Served", busy: "Turned away", errors: "Errors",
};
const VALUES: Record<string, string> = {
  rank: "Sentence ranking (default)", synth: "Local model", greedy: "flan-T5 greedy", beam: "flan-T5 beam", top_k: "flan-T5 top-k", nucleus: "flan-T5 nucleus",
  bm25: "Keyword (BM25)", dense: "Semantic", hybrid: "Hybrid", rerank: "Reranked", numpy: "Exact (numpy)", faiss: "FAISS", tfidf: "TF-IDF", lstm: "BiLSTM", bert: "SciBERT",
  rules: "Rules", biobert: "BioBERT",
};
const COSTS = /perplexity|questions|abstained|abstention|sentences|words|citations|refused|fallback|people|served|busy|errors|size|_ms$|_s$/i;

export const columnLabel = (key: string) => COLUMNS[key] ?? (key.charAt(0).toUpperCase() + key.slice(1)).replace(/_/g, " ")
  .replace(/ndcg/i, "nDCG").replace(/\bf1\b/i, "F1").replace(/rouge/i, "ROUGE").replace(/bleu/i, "BLEU");
export const cellLabel = (value: string) => VALUES[value] ?? value;
const SHARES = /^correct$|_rate$|^cited_/;

export function cellValue(column: string, v: any) {
  if (typeof v === "boolean") return v ? "Yes" : "No";
  if (typeof v !== "number") return cellLabel(String(v));
  if (SHARES.test(column)) return `${Math.round(v * 100)}%`;
  return Number.isInteger(v) ? String(v) : v.toFixed(3);
}
/** Whether a higher number in this column is simply better. */
export const ranked = (column: string) => !COSTS.test(column);

const CAPTIONS: Record<string, string> = {
  "cm_bert.png": "SciBERT section classifier: predicted against true labels", "cm_lstm.png": "BiLSTM section classifier: predicted against true labels",
  "cm_tfidf.png": "TF-IDF section classifier: predicted against true labels", "attn_bahdanau.png": "Bahdanau attention: where the title model looks",
  "attn_luong.png": "Luong attention: where the title model looks", "w2v_tsne.png": "Word2Vec: related words sit close together",
};
export const imageCaption = (file: string) => CAPTIONS[file] ?? file.replace(/\.\w+$/, "").replace(/_/g, " ");

type Row = { set: string; questions: number; correct: number };

/** The few numbers that say whether the site does its job, from the same reports as the tables. */
export function headlines(r: Record<string, any>) {
  const out: { value: string; label: string }[] = [];
  const rows: Row[] = r.abstention ?? [];
  const total = (want: boolean) => {
    const set = rows.filter((x) => x.set.includes("should answer") === want);
    return [Math.round(set.reduce((n, x) => n + x.correct * x.questions, 0)), set.reduce((n, x) => n + x.questions, 0)];
  };
  if (rows.length) {
    const [refused, unanswerable] = total(false), [answered, answerable] = total(true);
    out.push({ value: `${refused} of ${unanswerable}`, label: "Unanswerable questions refused" }, { value: `${answered} of ${answerable}`, label: "Answerable questions answered" });
  }
  const g = r.generation?.rank;
  if (g?.abstention_rate) {
    const stopped = g.refused_by_gate + g.refused_by_filter, all = Math.round(stopped / g.abstention_rate);
    out.push({ value: `${all - stopped} of ${all}`, label: "Broad questions answered" });
  }
  const c = r.relevance_check?.after;
  if (c) out.push({ value: `About ${Math.round((c.on_topic / c.sentences) * 100)}%`, label: "Quotes on topic" });
  return out;
}

/** Cells this far below the busiest one are the under-studied combinations. */
export const isGap = (count: number, max: number) => count <= max * 0.1;

type Byline = { authors?: string[]; year?: number | null; journal?: string };

const surname = (name: string) => name.replace(/\s+\S+$/, "");

/** "Smith et al., 2014", from whatever is known. */
export function shortCite({ authors = [], year }: Byline) {
  const who = authors.length > 2 ? `${surname(authors[0])} et al.` : authors.map(surname).join(" & ");
  return [who, year].filter(Boolean).join(", ");
}

/** One BibTeX entry per paper, for reference managers. */
export function bibtex(citations: ({ title: string; pmc_id: string; paper_id: string } & Byline)[]) {
  const seen = new Set<string>();
  const clean = (s: string) => s.replace(/[{}]/g, "");
  return citations.flatMap((c) => {
    const key = c.pmc_id || `paper${c.paper_id}`;
    if (seen.has(key)) return [];
    seen.add(key);
    const url = paperUrl(c.pmc_id);
    const fields = [["title", clean(c.title)], ["author", (c.authors ?? []).map(clean).join(" and ")], ["journal", clean(c.journal ?? "")], ["year", c.year ? String(c.year) : ""], ["url", url ?? ""]];
    return [`@article{${key},\n${fields.filter(([, v]) => v).map(([k, v]) => `  ${k} = {${v}}`).join(",\n")}\n}`];
  }).join("\n\n") + "\n";
}

export type Filters = { organism: string; stressor: string; tissue: string; year_min: number; q: string };

export function filtersFrom(p: URLSearchParams): Filters {
  return { organism: p.get("organism") ?? "", stressor: p.get("stressor") ?? "", tissue: p.get("tissue") ?? "", year_min: Number(p.get("year_min")) || 0, q: p.get("q") ?? "" };
}

export function filtersTo(f: Filters) {
  return Object.fromEntries(Object.entries(f).filter(([, v]) => v).map(([k, v]) => [k, String(v)]));
}

/** The closest star in an arrow key's direction, so the map works without a mouse. */
export function nextStar(pts: { x: number; y: number }[], from: number, [dx, dy]: number[]) {
  let best = from, near = Infinity;
  pts.forEach((p, i) => {
    const ax = p.x - pts[from].x, ay = p.y - pts[from].y, along = ax * dx + ay * dy, side = Math.abs(ax * dy) + Math.abs(ay * dx);
    if (along > 0 && side <= along && along + side < near) { near = along + side; best = i; }
  });
  return best;
}

