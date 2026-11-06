"""Evidence check + cited answers from two engines. Run: python backend/generate.py "question"

synth: a local Ollama model reads all passages together and marks each sentence with its source.
rank:  no generation. The cross-encoder scores each passage's sentences and the best are quoted, one per paper.
flan:  flan-T5 answers each passage on its own, with 4 decoding strategies to compare.
Either way, sentences that drift from their passage are dropped, and only passages behind a kept sentence are cited."""
import functools, json, re, sys
import llm
from data import CFG, PROC
from evidence import evidence_profile
from features import tokenize
from ner import extract
from retrieval import Retriever, _cross_encoder

DECODING = {"greedy": dict(do_sample=False, num_beams=1),
            "beam": dict(do_sample=False, num_beams=4, early_stopping=True),
            "top_k": dict(do_sample=True, top_k=50),
            "nucleus": dict(do_sample=True, top_p=0.9)}
ENGINES = ("rank", "synth", "flan")
NO_EVIDENCE = "Insufficient evidence in the corpus to answer this question."
MIN_SUPPORT = 0.7   # share of a sentence's content words that must appear in its own passage
MIN_WORDS = 3       # shorter outputs are fragments, not claims
MAX_OVERLAP = 0.8   # a sentence this close to an earlier one adds nothing
MAX_CLAIMS = 4      # longer answers from a small model mostly repeat themselves
MAX_PER_PASSAGE = 2 # leaves room for other papers in the answer
RANK_CLAIMS = 3     # quoted sentences per answer
CLOSEST_PAPERS = 3  # suggestions shown with a refusal
RANK_PASSAGES = 8   # passages searched for sentences
RANK_WORDS = (8, 60)  # shorter is a fragment, longer is a splitting failure
ABBREVIATION = re.compile(r"\b(?:e\.g|i\.e|cf|vs|al|fig|figs|ref|refs|approx|ca|eq|eqs|sect|suppl|spp)\.$", re.I)
NON_FINDING = re.compile(r"^To (?:determine|explore|examine|investigate|assess|evaluate|test)\b|\b(?:the (?:purpose|aim|goal|objective) of|this study (?:is|was|aims?|aimed|explored?|explores)|"
                         r"we (?:subjected|performed|aimed|sought|set out|queried|chose|hypothesi[sz]ed)|led us to|(?:little|less) is known|"
                         r"largely unknown|(?:is|are|remains?) (?:currently )?(?:unclear|unknown)|unclear (?:how|whether|if)|"
                         r"implications for|(?:is|are) presented in)\b", re.I)
MICROBES = re.compile(r"\bmicrob\w*", re.I)
RODENTS = re.compile(r"\brodents?\b", re.I)
ANIMAL_WORDS = re.compile(r"\b(?:rodents?|animals?|flies|fly)\b", re.I)
CELL_WORDS = re.compile(r"\b(?:cells?|cultur\w*|\w+cytes?)\b", re.I)
GENERIC_LIST = re.compile(r"\b(?:several|various|multiple|many|numerous) (?:body |organ |physiological )?systems\b", re.I)
# Words for a tissue that the entity lexicon misses. They live here so its reported scores stay as they are.
TISSUE_TERMS = {"gut": re.compile(r"\b(?:gastrointestinal|microbiota|colonic)\b", re.I),
                "immune": re.compile(r"\b(?:cytokines?|interleukin\w*|inflamm\w*|il-\d+)\b", re.I),
                "brain": re.compile(r"\b(?:brains|cognit\w*|memory|cns|central nervous system)\b", re.I),
                "muscle": re.compile(r"\b(?:sarcoplasmic|sarcopeni\w*)\b", re.I),
                "eye": re.compile(r"\bvisual\w*\b", re.I)}
# Sentences in these papers often imply spaceflight without the lexicon's words for it.
RADIATION_TERMS = re.compile(r"\b(?:IR|GCR|HZE|c?Gy|m?Sv)\b")  # case matters: "ir" and "gy" are not units
SPACE_TERMS = re.compile(r"\b(?:gravity|in-?flight|flight|flown|orbit\w*|astronauts?|space|iss)\b", re.I)
NEGATIONS = {"no", "not", "never", "without", "neither", "nor", "none", "cannot", "lacks", "lacking", "absent"}


@functools.lru_cache(maxsize=1)
def _lm():
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    tok = AutoTokenizer.from_pretrained(CFG["gen_model"])
    return tok, AutoModelForSeq2SeqLM.from_pretrained(CFG["gen_model"]).eval()


@functools.lru_cache(maxsize=1)
def _papers():
    p = PROC / "papers.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []


def support(sentence, passage):
    """Share of the sentence's content words found in the passage (0 when it has none)."""
    w = tokenize(sentence)
    ctx = set(tokenize(passage))
    return sum(t in ctx for t in w) / len(w) if w else 0.0


def sentences(text):
    out = []
    for part in re.split(r"(?<=[.!?])\s+", text):
        part = part.strip()
        if part and out and ABBREVIATION.search(out[-1]):
            out[-1] += " " + part  # "Fig. 2" is one sentence, not two
        elif part:
            out.append(part)
    return out


def whole_sentence(s):
    """Rejects fragments: an opening lowercase word, unbalanced brackets, or no closing punctuation."""
    return (s[-1:] in (".", "!", "?") and not ABBREVIATION.search(s) and s.count("(") == s.count(")")
            and not re.match(r"[a-z]+\b", s))


def tidy(sentence):
    """Closes gaps left where citation marks were stripped. Only numbers spaced like "9 , 10" go, because real
    prose writes "9, 10", so a real value such as "a dose of 10" is never removed."""
    s = re.sub(r"\s+\d{1,3}(?:\s+,\s+\d{1,3})+(?=\s*[.!?]$)", "", sentence)
    s = re.sub(r"\s+([.,;:!?)])", lambda m: m.group(1), s)
    s = re.sub(r"[,;:]+(?=[.!?]$)", "", s)  # "system , ." once its citation is gone
    return re.sub(r"\(\s+", "(", s)


def best_quote(claim, passage):
    """The passage sentence sharing the most content words with the claim, shown so readers can check it."""
    w = set(tokenize(claim))
    return max(sentences(passage), key=lambda s: len(w & set(tokenize(s))), default="")


def negated(text):
    words = re.findall(r"[a-z']+", text.lower())
    return any(w in NEGATIONS or w.endswith("n't") for w in words)


def flips_negation(sentence, passage):
    """Word overlap cannot see "did not increase" against "increased", so compare with the closest source sentence."""
    return negated(sentence) != negated(best_quote(sentence, passage))


def select_claims(claims, hits):
    """[(hit index, sentence)] -> the ones that are grounded in their own passage, whole and not repeats."""
    kept = []
    for i, d in claims:
        d = re.sub(r"\s*\[\d+\]", "", d).strip()
        w = set(tokenize(d))
        if len(d.split()) < MIN_WORDS or support(d, hits[i]["text"]) < MIN_SUPPORT or flips_negation(d, hits[i]["text"]):
            continue
        if any(len(w & set(tokenize(s))) / max(len(w), 1) >= MAX_OVERLAP for _, s in kept):
            continue
        kept.append((i, d))
        if len(kept) == MAX_CLAIMS:
            break
    return kept


def select(drafts, hits):
    """drafts[i] was generated from hits[i]."""
    return select_claims(list(enumerate(drafts)), hits)


def parse_claims(text, n_hits):
    """Model text -> [(hit index, sentence)]. Accepts marks before or after sentences; other text is dropped."""
    parts = re.split(r"\[(\d+)\]", text)
    marks = [int(m) for m in parts[1::2]]
    segments = parts[2::2] if not parts[0].strip() else parts[0::2][:len(marks)]
    return [(n - 1, s) for n, seg in zip(marks, segments) if 1 <= n <= n_hits for s in sentences(seg)]


def diverse(hits, k, threshold):
    """Up to k passages that clear the evidence bar, taking one per paper before repeating a paper."""
    ok = [h for h in hits if h["rerank"] is None or h["rerank"] >= threshold]
    seen, first, rest = set(), [], []
    for h in ok:
        (rest if h["paper_id"] in seen else first).append(h)
        seen.add(h["paper_id"])
    return (first + rest)[:k]


def extract_claim(question, hit):
    """The passage's own sentence closest to the question. It cannot misquote, because it is copied."""
    words = set(tokenize(question))
    s = best_quote(question, hit["text"])
    return s if words & set(tokenize(s)) else None


def balance(question, claims, hits):
    """Spreads the answer across passages, and fills any passage the model ignored with its closest sentence."""
    count, kept = {}, []
    for i, s in claims:
        if count.get(i, 0) < MAX_PER_PASSAGE:
            kept.append((count.get(i, 0), i, s))
            count[i] = count.get(i, 0) + 1
    for i, h in enumerate(hits):
        s = None if i in count else extract_claim(question, h)
        if s:
            kept.append((0, i, s))
    ranked = [(i, s) for _, i, s in sorted(kept, key=lambda c: c[0])]  # every passage's first sentence goes first
    return sorted(select_claims(ranked, hits), key=lambda c: c[0])


def _claim_scores(question, claims):
    return [float(x) for x in _cross_encoder().predict([(question, c) for c in claims], show_progress_bar=False)]


def named(text, kind):
    return {x["cat"] for x in extract(text) if x["type"] == kind}


def tissues(text):
    return named(text, "tissue") | {t for t, rx in TISSUE_TERMS.items() if rx.search(text)}


def stressors(text):
    return (named(text, "stressor") | ({"microgravity"} if SPACE_TERMS.search(text) else set())
            | ({"radiation"} if RADIATION_TERMS.search(text) else set()))


def addresses(question, sentence):
    """A quote answers only if it reports something about what was asked. A question naming no tissue or stressor
    is not narrowed. Whole organisms are not compared, because that rejected too many fair general statements."""
    if NON_FINDING.search(sentence) or GENERIC_LIST.search(sentence):
        return False
    if "cells" in named(question, "organism") and not CELL_WORDS.search(sentence):
        return False
    tissue, stressor = tissues(question), named(question, "stressor")
    return (not tissue or bool(tissue & tissues(sentence))) and (not stressor or bool(stressor & stressors(sentence)))


def other_organism(question, sentence):
    """True when a sentence is explicitly about a different organism from the one asked about."""
    want = named(question, "organism") - {"cells"}
    got = named(MICROBES.sub("", sentence), "organism") - {"cells"}  # a gut microbiome is not a separate study organism
    if RODENTS.search(sentence):
        got |= {"mouse"}
    if "human" in want and ANIMAL_WORDS.search(sentence):
        got |= {"animal"}
    return bool(want and got and not want & got)


def rank_claims(question, hits, bar):
    """Best sentences across passages, one per paper. The top one always stays, because its passage already
    passed the evidence gate; extra sentences must clear the bar so weak filler stays out."""
    lo, hi = RANK_WORDS
    cand = [(i, s) for i, h in enumerate(hits) for s in sentences(h["text"])
            if lo <= len(s.split()) <= hi and whole_sentence(s) and addresses(question, s)]
    if not cand:
        return []
    scores = _claim_scores(question, [s for _, s in cand])
    ranked = sorted(zip(scores, cand), key=lambda x: -x[0])
    # Evidence from the organism asked about wins, but other organisms still answer when nothing else clears the bar.
    matching = [x for x in ranked if x[0] >= bar and not other_organism(question, x[1][1])]
    kept, papers = [], set()
    for score, (i, s) in matching or ranked:
        if (kept and score < bar) or len(kept) == RANK_CLAIMS:
            break
        if hits[i]["paper_id"] not in papers:
            papers.add(hits[i]["paper_id"])
            kept.append((i, tidy(s)))
    return kept


def closest_papers(hits, floor):
    """Papers to read after a refusal. These are leads, not evidence, so they never enter the answer."""
    out, seen = [], set()
    for h in hits:
        if h["rerank"] is None or h["rerank"] < floor or h["paper_id"] in seen:
            continue
        seen.add(h["paper_id"])
        out.append({"paper_id": h["paper_id"], "pmc_id": h.get("pmc_id", ""), "title": h["title"], "section": h["section"]})
        if len(out) == CLOSEST_PAPERS:
            break
    return out


def on_topic(question, kept):
    """Drops sentences that do not address the question. Off until claim_threshold is set from labeled data."""
    t = CFG["retrieval"]["claim_threshold"]
    if t is None or not kept:
        return kept
    scores = _claim_scores(question, [s for _, s in kept])
    return [k for k, v in zip(kept, scores) if v >= t]


def compose(kept, hits):
    """-> (answer text, citations). Citations are renumbered so [n] always indexes the returned list."""
    parts, cites, number, quotes, spans = [], [], {}, {}, {}
    for i, s in kept:
        h = hits[i]
        if i not in number:
            number[i] = len(number) + 1
            quotes[i], spans[i] = [], []
            cites.append({"chunk_id": h["chunk_id"], "paper_id": h["paper_id"], "pmc_id": h.get("pmc_id", ""), "title": h["title"],
                          "section": h["section"], "passage": h["text"],
                          "year": h.get("year"), "authors": h.get("authors", []), "journal": h.get("journal", "")})
        raw = best_quote(s, h["text"])
        q, at = tidy(raw), h["text"].find(raw)
        if q not in quotes[i]:
            quotes[i].append(q)
            if at >= 0:  # offsets into the passage, because tidying means the quote may not match it exactly
                spans[i].append([at, at + len(raw)])
        parts.append(f"{s.rstrip('.')}. [{number[i]}]")
    for i, n in number.items():
        cites[n - 1].update(quote=" … ".join(quotes[i]), spans=sorted(spans[i]))
    return " ".join(parts), cites


def sufficient(hits):
    """Evidence gate: the best passage must clear the calibrated cross-encoder threshold."""
    scores = [h["rerank"] for h in hits if h["rerank"] is not None]
    return bool(scores) and max(scores) >= CFG["retrieval"]["evidence_threshold"]


def draft(question, hits, decoding):
    import torch
    from transformers import set_seed
    tok, lm = _lm()
    prompts = [f"Read the passage and answer the question in one complete sentence.\n\nPassage: {h['text']}\n\n"
               f"Question: {question}\nAnswer:" for h in hits]
    enc = tok(prompts, return_tensors="pt", truncation=True, max_length=512, padding=True)
    set_seed(CFG["seed"])
    with torch.no_grad():
        ids = lm.generate(**enc, max_new_tokens=80, no_repeat_ngram_size=3, **DECODING[decoding])
    return tok.batch_decode(ids, skip_special_tokens=True)


def synth_prompt(question, hits):
    passages = "\n\n".join(f"[{i + 1}] {h['text']}" for i, h in enumerate(hits))
    return (f"Passages:\n{passages}\n\nQuestion: {question}\n\n"
            "Answer in two or three sentences using only the passages. "
            "End each sentence with the passage number, like [1].")


def answer(question, decoding="greedy", retriever=None, mode=None, engine="flan"):
    if decoding not in DECODING:
        raise ValueError(f"decoding must be one of {list(DECODING)}")
    if engine not in ENGINES:
        raise ValueError(f"engine must be one of {list(ENGINES)}")
    r = retriever or Retriever()
    mode = mode or CFG["retrieval"]["default_mode"]
    # The gate needs cross-encoder scores whichever mode ranks the passages shown to the model.
    gate = r.search(question, "rerank", CFG["retrieval"]["evidence_k"])
    out = {"answer": NO_EVIDENCE, "sufficient": False, "citations": [], "decoding": decoding, "engine": engine, "reason": None, "evidence": None, "closest": []}
    leads = closest_papers(gate, CFG["retrieval"]["closest_floor"])
    if not sufficient(gate):
        return {**out, "reason": "gate", "closest": leads}
    bar = CFG["retrieval"]["evidence_threshold"]
    if engine == "rank":
        hits = diverse(gate, RANK_PASSAGES, bar)
        kept = rank_claims(question, hits, CFG["retrieval"]["sentence_threshold"])
        if not kept:
            return {**out, "reason": "filter", "closest": leads}
        text, cites = compose(kept, hits)
        return {**out, "answer": text, "sufficient": True, "citations": cites, "evidence": evidence_profile(gate, _papers(), bar)}
    ranked, k = (gate if mode == "rerank" else r.search(question, mode)), CFG["retrieval"]["ctx_k"]
    hits = diverse(ranked, k, CFG["retrieval"]["evidence_threshold"]) if engine == "synth" else ranked[:k]
    try:
        if engine != "synth":
            raise llm.LLMUnavailable("flan requested")
        drafted = select_claims(parse_claims(llm.complete(synth_prompt(question, hits)), len(hits)), hits)
        kept = balance(question, drafted, hits)
    except llm.LLMUnavailable:
        out["engine"] = "flan"
        kept = select(draft(question, hits, decoding), hits)
    kept = on_topic(question, kept)
    if not kept:
        return {**out, "reason": "filter", "closest": leads}
    text, cites = compose(kept, hits)
    evidence = evidence_profile(gate, _papers(), CFG["retrieval"]["evidence_threshold"])
    return {**out, "answer": text, "sufficient": True, "citations": cites, "evidence": evidence}


if __name__ == "__main__":
    print(answer(sys.argv[1] if len(sys.argv) > 1 else "How does microgravity affect bone density in mice?", engine="rank"))
