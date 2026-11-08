import { FormEvent, useEffect, useRef, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { AnimatePresence, motion } from "motion/react";
import { api, Answer, Evidence } from "../api";
import CitationCard from "../components/CitationCard";
import { Arrow, EASE, PageHead, Skeleton } from "../components/ui";
import { answerLabel, bibtex, citationText, claims, engineHint, paperUrl, refusalMessage, shortCite } from "../lib";
import { useStats } from "../useStats";

const SAMPLES = ["How does microgravity affect bone density in mice?", "What happens to plant roots in microgravity?",
  "What countermeasures reduce bone loss in space?", "Does spaceflight increase dandruff in astronauts?"];

const LIMIT_MS = 180_000; // the local model's timeout plus the flan-T5 fallback

function EvidenceStrip({ e }: { e: Evidence }) {
  const years = e.year_min === null ? "" : e.year_min === e.year_max ? `${e.year_min}` : `${e.year_min}–${e.year_max}`;
  const top = Object.keys(e.organisms).slice(0, 3);
  const parts = [`${e.papers} ${e.papers === 1 ? "paper clears" : "papers clear"} the evidence check`, years, top.join(", ")].filter(Boolean);
  return <p className="evidence-strip mut">{parts.join(" · ")}</p>;
}

export default function Ask() {
  const [params, setParams] = useSearchParams();
  const stats = useStats();
  const asked = params.get("q");
  const [q, setQ] = useState(asked ?? "");
  const [decoding, setDecoding] = useState("greedy");
  const [engine, setEngine] = useState("rank");
  const [askedWith, setAskedWith] = useState("rank");
  const [res, setRes] = useState<Answer | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const [active, setActive] = useState(0);
  const [secs, setSecs] = useState(0);
  const [copied, setCopied] = useState("");
  const [askedQ, setAskedQ] = useState("");
  const run = useRef(0);
  const abort = useRef<AbortController | null>(null);
  const timedOut = useRef(false);

  useEffect(() => {
    document.documentElement.dataset.energy = busy ? "1" : "0";
    return () => { document.documentElement.dataset.energy = "0"; };
  }, [busy]);

  useEffect(() => {
    if (!busy) return;
    setSecs(0);
    const t = setInterval(() => setSecs((s) => s + 1), 1000);
    return () => clearInterval(t);
  }, [busy]);
  useEffect(() => () => abort.current?.abort(), []);

  const go = async (question: string) => {
    if (!question.trim()) return;
    const id = ++run.current;
    abort.current?.abort();
    const ctl = new AbortController();
    abort.current = ctl;
    timedOut.current = false;
    const timer = setTimeout(() => { timedOut.current = true; ctl.abort(); }, LIMIT_MS);
    setBusy(true); setErr(""); setRes(null); setActive(0); setAskedWith(engine); setAskedQ(question); setCopied("");
    setParams({ q: question }, { replace: true });
    try {
      const r = await api.ask(question, decoding, engine, ctl.signal);
      if (id === run.current) setRes(r);
    } catch (e: any) {
      if (id === run.current) {
        if (e?.name !== "AbortError") setErr(e.message);
        else if (timedOut.current) setErr("This took too long. Try again, or ask a narrower question.");
      }
    }
    clearTimeout(timer);
    if (id === run.current) setBusy(false);
  };

  useEffect(() => { if (asked) go(asked); }, []);

  const submit = (e: FormEvent) => { e.preventDefault(); go(q); };
  const pick = (s: string) => { setQ(s); go(s); };
  const copy = async (what: string, text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(what);
    } catch {
      setCopied("failed");
    }
    setTimeout(() => setCopied(""), 2000);
  };
  const toSource = (n: number) => {
    setActive(n);
    document.getElementById(`source-${n}`)?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  };
  const searchLink = "/search?q=" + encodeURIComponent(askedQ);

  return (
    <>
      <PageHead title="Ask the corpus">Answers come only from retrieved passages, and every sentence links to its source.</PageHead>
      <form className="composer" onSubmit={submit}>
        <label className="sr" htmlFor="ask-q">Your question</label>
        <input id="ask-q" value={q} onChange={(e) => setQ(e.target.value)} placeholder="Ask about space biology…" autoComplete="off" />
        <label className="sr" htmlFor="ask-e">Answer engine</label>
        <select id="ask-e" value={engine} onChange={(e) => setEngine(e.target.value)}>
          <option value="rank">Exact quotes</option>
          <option value="synth">Local AI wording</option>
          <option value="flan">flan-T5 comparison</option>
        </select>
        {engine === "flan" && (
          <>
            <label className="sr" htmlFor="ask-d">Decoding strategy</label>
            <select id="ask-d" value={decoding} onChange={(e) => setDecoding(e.target.value)}>
              {["greedy", "beam", "top_k", "nucleus"].map((d) => <option key={d}>{d}</option>)}
            </select>
          </>
        )}
        <button className="send" type="submit" disabled={busy || !q.trim()} aria-label="Ask"><Arrow /></button>
      </form>
      {engineHint(engine) && <p className="mut hint">{engineHint(engine)}</p>}
      {!res && !busy && !err && (
        <div className="chips">
          {SAMPLES.map((s) => <button key={s} type="button" className="chip" onClick={() => pick(s)}>{s}</button>)}
        </div>
      )}
      <AnimatePresence mode="wait">
        {busy && (
          <motion.div key="busy" className="card" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} transition={{ duration: 0.3 }}>
            <p className="mut shimmer" role="status">Searching {stats ? stats.passages.toLocaleString("en-US") + " " : ""}passages and checking the evidence.</p>
            <Skeleton />
            <div className="busy-row">
              <span className="mut num">{secs}s{secs >= 10 ? " · this can take up to a minute on CPU" : ""}</span>
              <button type="button" className="chip" onClick={() => abort.current?.abort()}>Cancel</button>
            </div>
          </motion.div>
        )}
        {err && <motion.p key="err" className="card warn" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>{err}</motion.p>}
        {res && !res.sufficient && (
          <motion.section key="refused" className="card refusal" initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} transition={{ duration: 0.5, ease: EASE }}>
            <p className="eyebrow">Not enough evidence</p>
            <p className="claim">{res.answer}</p>
            <p className="mut">{refusalMessage(res.reason, res.closest.length > 0)}</p>
            {res.closest.length > 0 && (
              <div className="leads">
                <h2 className="h3">Closest papers</h2>
                <p className="mut">Worth reading, but not evidence for an answer.</p>
                <ul>
                  {res.closest.map((p) => {
                    const url = paperUrl(p.pmc_id);
                    return <li key={p.paper_id}>{url ? <a href={url} target="_blank" rel="noopener noreferrer">{p.title}</a> : p.title}</li>;
                  })}
                </ul>
                <div className="chips"><Link className="chip" to={searchLink}>Search the passages yourself</Link></div>
              </div>
            )}
            {!res.closest.length && (
              <div className="chips">
                {SAMPLES.slice(0, 3).map((s) => <button key={s} type="button" className="chip" onClick={() => pick(s)}>{s}</button>)}
              </div>
            )}
          </motion.section>
        )}
        {res?.sufficient && (
          <motion.section key="answer" className="answer" initial="hide" animate="show" exit={{ opacity: 0 }}>
            <div className="card">
              <p className="mut">{res.citations.length} {res.citations.length === 1 ? "source" : "sources"} · <span className="gen">{answerLabel(res.engine, res.decoding)}</span></p>
              {res.evidence && <EvidenceStrip e={res.evidence} />}
              {res.engine !== askedWith && <p className="mut">The local model is unavailable, so this answer came from flan-T5.</p>}
              <div className="claims" aria-live="polite">
                {claims(res.answer).map((c, i) => (
                  <motion.p key={i} className="claim" variants={{ hide: { opacity: 0, y: 12 }, show: { opacity: 1, y: 0 } }} transition={{ duration: 0.55, ease: EASE, delay: i * 0.16 }}>
                    {c.text}.
                    {c.n > 0 && (
                      <button type="button" className={"cite" + (active === c.n ? " on" : "")} aria-label={`Show source ${c.n}`}
                        onClick={() => toSource(c.n)} onMouseEnter={() => setActive(c.n)} onMouseLeave={() => setActive(0)} onFocus={() => setActive(c.n)} onBlur={() => setActive(0)}>{c.n}</button>
                    )}
                  </motion.p>
                ))}
              </div>
              <div className="chips">
                <button type="button" className="chip" onClick={() => copy("text", citationText(askedQ, res.answer, res.citations))}>{copied === "text" ? "Copied" : "Copy with references"}</button>
                <button type="button" className="chip" onClick={() => copy("bib", bibtex(res.citations))}>{copied === "bib" ? "Copied" : "Copy BibTeX"}</button>
                <Link className="chip" to={searchLink}>See more passages</Link>
              </div>
              {copied === "failed" && <p className="mut" role="status">Your browser blocked copying.</p>}
            </div>
            <aside aria-label="Sources">
              <h2 className="eyebrow">Sources</h2>
              {res.citations.map((c, i) => (
                <motion.div key={c.chunk_id} variants={{ hide: { opacity: 0, x: 16 }, show: { opacity: 1, x: 0 } }} transition={{ duration: 0.55, ease: EASE, delay: 0.2 + i * 0.1 }}>
                  <CitationCard id={`source-${i + 1}`} c={c} n={i + 1} text={c.quote} note={[shortCite(c), c.journal].filter(Boolean).join(" · ")} limit={700} active={active === i + 1} />
                </motion.div>
              ))}
            </aside>
          </motion.section>
        )}
      </AnimatePresence>
    </>
  );
}
