import { useState } from "react";
import { motion } from "motion/react";
import { api, Lead, Paper } from "../api";
import { paperUrl } from "../lib";
import { EASE } from "./ui";

function Related({ id }: { id: string }) {
  const [leads, setLeads] = useState<Lead[] | null>(null);
  const [failed, setFailed] = useState(false);
  const load = () => { setFailed(false); api.related(id).then(setLeads).catch(() => setFailed(true)); };
  if (leads) {
    return (
      <ul className="related">
        {leads.map((p) => {
          const url = paperUrl(p.pmc_id);
          return <li key={p.paper_id}>{url ? <a href={url} target="_blank" rel="noopener noreferrer">{p.title}</a> : p.title}</li>;
        })}
      </ul>
    );
  }
  return <button type="button" className="chip related-btn" onClick={load}>{failed ? "Could not load, try again" : "Related papers"}</button>;
}

type Facet = "organism" | "stressor" | "tissue";

export default function PaperTable({ papers, selected, onTag, onClear }: { papers: Paper[]; selected?: string | null; onTag: (facet: Facet, value: string) => void; onClear: () => void }) {
  return (
    <>
      {papers.map((p, i) => (
        <motion.article key={p.paper_id} id={`paper-${p.paper_id}`} className={"card paper" + (p.paper_id === selected ? " on" : "")} initial={{ opacity: 0, y: 14 }}
          whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true, margin: "-20px" }} transition={{ duration: 0.45, ease: EASE, delay: (i % 4) * 0.04 }}>
          <h3>
            {p.pmc_id ? <a href={paperUrl(p.pmc_id)!} target="_blank" rel="noopener noreferrer">{p.title}<span className="sr"> (opens in a new tab)</span></a> : p.title}
          </h3>
          <p className="mut">{p.year ?? "Year unknown"}</p>
          <div className="tags">
            {(["organism", "stressor", "tissue"] as const).flatMap((t) =>
              p.entities[t].map((v) => <button key={t + v} type="button" className="tag" title={`Show only ${v} papers`} aria-label={`Filter by ${t}: ${v}`} onClick={() => onTag(t, v)}>{v}</button>)
            )}
          </div>
          {p.summary && <p className="quote">{p.summary} <span className="gen">key sentences</span></p>}
          <Related id={p.paper_id} />
        </motion.article>
      ))}
      {!papers.length && (
        <div className="card">
          <p className="mut">No papers match these filters. This combination may be a real gap in the research.</p>
          <div className="chips"><button type="button" className="chip" onClick={onClear}>Clear filters</button></div>
        </div>
      )}
    </>
  );
}
