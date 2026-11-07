import { useEffect, useMemo, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { api, Results as R } from "../api";
import MetricTable from "../components/MetricTable";
import { PageHead, Reveal, Skeleton } from "../components/ui";
import { headlines, imageCaption } from "../lib";

const isScalar = (v: any) => ["number", "string", "boolean"].includes(typeof v);

// dict-of-dicts -> rows; recurses until a node holds scalar metrics
function rowsOf(obj: any, prefix = ""): Record<string, any>[] {
  if (Array.isArray(obj)) return obj;
  if (!obj || typeof obj !== "object") return [];
  if (Object.values(obj).some(isScalar)) return [{ name: prefix || "–", ...obj }];
  return Object.entries(obj).flatMap(([k, v]) => rowsOf(v, prefix ? `${prefix}/${k}` : k));
}

type Section = [string, string, string];
const GROUPS: { id: string; title: string; lede: string; sections: Section[] }[] = [
  { id: "safety", title: "Does it answer safely?", lede: "The main test: answer what the papers support, and refuse the rest.", sections: [
    ["abstention", "Refusal", "Share handled correctly in each set. Unanswerable questions must be refused, and answerable ones answered."],
    ["generation", "Answer engines", "On 55 broad questions. Agreement with the rule tagger, not human judgment."],
    ["natural", "Everyday questions", "30 questions in natural wording, kept as a regression check."],
    ["faithfulness", "Answer faithfulness", ""],
  ] },
  { id: "retrieval", title: "Does it find the right papers?", lede: "Search quality comes first, because answers can only quote what retrieval finds.", sections: [
    ["retrieval_judged", "Retrieval on judged relevance", "Broad questions. Relevance labels are model-judged, not human."],
    ["retrieval_known", "Known-item retrieval", "Narrow questions written from one source paper."],
    ["retrieval", "Retrieval on silver labels", "Gold papers come from the rule tagger, so this measures agreement with it."],
    ["ablation", "Preprocessing ablation", "BM25 under each cleaning choice."],
  ] },
  { id: "models", title: "How did the trained models do?", lede: "From an earlier run. The weights are not stored here, so these were not reproduced.", sections: [
    ["classify", "Section classifier", "SciBERT trained on 2,000 chunks, the others on all of them."],
    ["ner", "Named entities", "BioBERT trained on the rules' own labels, and did not beat them."],
    ["seq2seq", "Titles, summaries, language model", "ROUGE and BLEU compare generated text with the original. Lower perplexity is better."],
  ] },
  { id: "serving", title: "Can it run and grow?", lede: "Measured on one laptop.", sections: [
    ["load", "Load test", "Several people asking at once, with one answer running at a time."],
    ["scale", "Vector search at scale", "Exact search won every row measured, so it stays the default. The 500,000 run did not finish."],
  ] },
];

export default function Results() {
  const [r, setR] = useState<R | null>(null);
  const [err, setErr] = useState("");
  const [zoom, setZoom] = useState<string | null>(null);
  useEffect(() => { api.results().then(setR).catch((e) => setErr(e.message)); }, []);
  useEffect(() => {
    if (!zoom) return;
    const close = (e: KeyboardEvent) => { if (e.key === "Escape") setZoom(null); };
    addEventListener("keydown", close);
    return () => removeEventListener("keydown", close);
  }, [zoom]);
  const top = useMemo(() => (r ? headlines(r) : []), [r]);
  const groups = GROUPS.filter((g) => r && g.sections.some(([k]) => r[k]));
  const jump = (id: string) => document.getElementById(id)?.scrollIntoView({ behavior: "smooth" });
  return (
    <>
      <PageHead title="The numbers, including the losses">Where the project does well, and where a simpler method won. The best score in each column is highlighted.</PageHead>
      {err && <p className="card warn">{err}</p>}
      {!r && !err && <div className="card"><Skeleton lines={5} /></div>}
      {top.length > 0 && (
        <div className="headlines">
          {top.map((h) => <div key={h.label} className="card headline"><b>{h.value}</b><span className="mut">{h.label}</span></div>)}
        </div>
      )}
      {groups.length > 0 && (
        <nav className="chips" aria-label="Sections">
          {groups.map((g) => <button key={g.id} type="button" className="chip" onClick={() => jump(g.id)}>{g.title}</button>)}
        </nav>
      )}
      {r && groups.map((g) => (
        <section key={g.id} id={g.id} className="group">
          <h2 className="grouphead">{g.title}</h2>
          <p className="lede">{g.lede}</p>
          {g.sections.map(([k, title, note]) => (r[k] ? <Reveal key={k}><MetricTable title={title} note={note} rows={rowsOf(r[k])} /></Reveal> : null))}
        </section>
      ))}
      {r && <h2 id="plots" className="grouphead group">Plots</h2>}
      {r?.w2v && (
        <Reveal>
          <div className="card">
            <h2 className="h3">Word2Vec nearest neighbours</h2>
            {Object.entries(r.w2v as Record<string, string[]>).map(([w, n]) => (
              <p key={w} className="neighbour"><b>{w}</b> <span className="mut">{n.join(", ")}</span></p>
            ))}
          </div>
        </Reveal>
      )}
      <div className="grid gallery">
        {(r?.images ?? []).map((f) => (
          <Reveal key={f}>
            <figure className="plot">
              <button type="button" className="shot" onClick={() => setZoom(f)} aria-label={`Enlarge: ${imageCaption(f)}`}>
                <img src={api.img(f)} alt={imageCaption(f)} loading="lazy" />
              </button>
              <figcaption className="mut">{imageCaption(f)}</figcaption>
            </figure>
          </Reveal>
        ))}
      </div>
      <AnimatePresence>
        {zoom && (
          <motion.button type="button" className="lightbox" aria-label="Close image" onClick={() => setZoom(null)}
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.25 }}>
            <motion.img src={api.img(zoom)} alt={imageCaption(zoom)} initial={{ scale: 0.94 }} animate={{ scale: 1 }} exit={{ scale: 0.96 }} transition={{ type: "spring", stiffness: 320, damping: 30 }} />
          </motion.button>
        )}
      </AnimatePresence>
    </>
  );
}
