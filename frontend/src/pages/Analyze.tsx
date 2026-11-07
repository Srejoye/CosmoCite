import { useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { api, Analysis } from "../api";
import AttentionText from "../components/AttentionText";
import EntityHighlighter from "../components/EntityHighlighter";
import { EASE, PageHead, Skeleton } from "../components/ui";

const SAMPLE = "The average weight loss of space-flown mice was significantly lower than pre-flight values, consistent with tissue atrophy in microgravity. Osteoclast activity increased in the pelvis, and trabecular bone volume fell after thirty days of spaceflight.";

export default function Analyze() {
  const [text, setText] = useState("");
  const [shown, setShown] = useState("");
  const [res, setRes] = useState<Analysis | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");

  const go = async (passage: string) => {
    setBusy(true); setErr("");
    try { setRes(await api.analyze(passage)); setShown(passage); } catch (e: any) { setErr(e.message); }
    setBusy(false);
  };
  const sample = () => { setText(SAMPLE); go(SAMPLE); };

  return (
    <>
      <PageHead title="Read a passage closely">Paste text from a paper to tag its organisms, stressors, tissues and outcomes.</PageHead>
      <div className="card">
        <label className="sr" htmlFor="a-text">Passage</label>
        <textarea id="a-text" placeholder="Paste a passage from a paper…" value={text} onChange={(e) => setText(e.target.value)} />
        <div className="row" style={{ marginTop: 12 }}>
          <button type="button" disabled={busy || !text.trim()} onClick={() => go(text)}>{busy ? "Tagging…" : "Tag this passage"}</button>
          <button type="button" className="chip" disabled={busy} onClick={sample}>Try a sample passage</button>
        </div>
      </div>
      {err && <p className="card warn">{err}</p>}
      {busy && <div className="card"><Skeleton /></div>}
      <AnimatePresence>
        {res && !busy && (
          <motion.div initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} transition={{ duration: 0.5, ease: EASE }}>
            <div className="card">
              <h2 className="h3">What the passage mentions</h2>
              <EntityHighlighter text={shown} spans={res.entities} />
            </div>
            {res.label ? (
              <div className="grid">
                <div className="card">
                  <h2 className="h3">Section: {res.label}</h2>
                  <p className="mut">Predicted by {res.model}</p>
                  <div className="tags">{Object.entries(res.probs).map(([k, v]) => <span key={k} className="tag">{k} {(v * 100).toFixed(0)}%</span>)}</div>
                </div>
                <div className="card">
                  <h2 className="h3">Words the {res.attention_model} focused on</h2>
                  <p className="mut">These weights belong to the {res.attention_model}, which predicted "{res.attention_label}".</p>
                  <AttentionText words={res.attention} />
                </div>
              </div>
            ) : (
              <p className="card mut">Section prediction needs the trained classifier, which this copy of CosmoCite does not include. The tags above work without it.</p>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
