import { FormEvent, useEffect, useMemo, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { AnimatePresence, motion } from "motion/react";
import { api, Hit, SearchRes } from "../api";
import CitationCard from "../components/CitationCard";
import { Arrow, EASE, PageHead, Segmented, Skeleton } from "../components/ui";
import { agreement, shortCite } from "../lib";
import { useStats } from "../useStats";

type Mode = keyof SearchRes;
const MODES: [Mode, string, string][] = [
  ["bm25", "Keyword", "BM25 matches your exact terms."],
  ["dense", "Semantic", "Embeddings match meaning. Strongest on broad questions."],
  ["hybrid", "Hybrid", "Fuses keyword and semantic rankings."],
  ["rerank", "Reranked", "A cross-encoder re-scores the top passages. Ask uses this one."],
];
const SAMPLES = ["osteoclast bone loss spaceflight", "root gravitropism microgravity", "radiation DNA damage repair"];

const onePerPaper = (hits: Hit[]) => hits.filter((h, i) => hits.findIndex((x) => x.paper_id === h.paper_id) === i);

export default function Search() {
  const stats = useStats();
  const [params, setParams] = useSearchParams();
  const [q, setQ] = useState(params.get("q") ?? "");
  const [res, setRes] = useState<SearchRes | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const [view, setView] = useState<"all" | Mode>("all");
  const [unique, setUnique] = useState(true);

  const go = async (text: string) => {
    if (!text.trim()) return;
    setBusy(true); setErr("");
    setParams({ q: text }, { replace: true });
    try { setRes(await api.search(text)); } catch (e: any) { setErr(e.message); }
    setBusy(false);
  };
  const submit = (e: FormEvent) => { e.preventDefault(); go(q); };
  useEffect(() => { if (q) go(q); }, []);

  const cols = useMemo(() => MODES.filter(([m]) => view === "all" || view === m), [view]);
  const found = useMemo(() => (res ? agreement(res) : new Map<string, number>()), [res]);
  const agreed = (id: string) => {
    const n = found.get(id) ?? 1;
    return n === 1 ? "Only this retriever" : `Found by ${n} of ${MODES.length}`;
  };

  return (
    <>
      <PageHead title="Four ways to search">Run one query through every retriever and see how the results differ.</PageHead>
      <form className="composer" onSubmit={submit}>
        <label className="sr" htmlFor="s-q">Search passages</label>
        <input id="s-q" value={q} onChange={(e) => setQ(e.target.value)} placeholder={`Search ${stats ? stats.passages.toLocaleString("en-US") + " " : ""}passages…`} autoComplete="off" />
        <button className="send" type="submit" disabled={busy || !q.trim()} aria-label="Search"><Arrow /></button>
      </form>
      {!res && !busy && (
        <div className="chips">
          {SAMPLES.map((s) => <button key={s} type="button" className="chip" onClick={() => { setQ(s); go(s); }}>{s}</button>)}
        </div>
      )}
      {err && <p className="card warn">{err}</p>}
      {(res || busy) && (
        <div className="toolbar">
          <Segmented id="search-view" label="Retriever" value={view} onChange={setView}
            options={[["all", "Compare all"], ...MODES.map(([m, l]) => [m, l] as [Mode, string])]} />
          <button type="button" className="chip" aria-pressed={unique} onClick={() => setUnique(!unique)}>One result per paper</button>
          {q.trim() && <Link className="chip" to={"/ask?q=" + encodeURIComponent(q)}>Ask this as a question</Link>}
        </div>
      )}
      <div className="columns" style={{ gridTemplateColumns: `repeat(auto-fit, minmax(min(100%, ${view === "all" ? 250 : 420}px), 1fr))` }}>
        <AnimatePresence mode="popLayout">
          {(res || busy) && cols.map(([m, label, hint], i) => (
            <motion.section key={m} layout initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
              transition={{ duration: 0.45, ease: EASE, delay: i * 0.05 }} aria-label={label}>
              <h2 className="colhead">{label}</h2>
              <p className="mut colhint">{hint}</p>
              {busy ? <div className="card"><Skeleton /></div>
                : (unique ? onePerPaper(res![m]) : res![m]).map((h, n) => <CitationCard key={h.chunk_id} c={h} n={n + 1} text={h.text} note={[shortCite(h), agreed(h.chunk_id)].filter(Boolean).join(" · ")} />)}
            </motion.section>
          ))}
        </AnimatePresence>
      </div>
    </>
  );
}
