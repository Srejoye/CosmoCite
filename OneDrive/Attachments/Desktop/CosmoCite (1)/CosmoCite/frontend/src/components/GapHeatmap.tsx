import { Link } from "react-router-dom";
import type { Gap } from "../api";
import { gapQuestion, isGap } from "../lib";

export default function GapHeatmap({ gaps, organism, stressor, onPick }:
  { gaps: Gap[]; organism: string; stressor: string; onPick: (organism: string, stressor: string) => void }) {
  const orgs = [...new Set(gaps.map((g) => g.organism))];
  const strs = [...new Set(gaps.map((g) => g.stressor))];
  const max = Math.max(...gaps.map((g) => g.count), 1);
  const get = (o: string, s: string) => gaps.find((g) => g.organism === o && g.stressor === s)?.count ?? 0;
  return (
    <div className="card">
      <h2 className="h3">Research gap map</h2>
      <p className="mut">Papers per organism and stressor. Dashed cells are the biggest gaps. Click one to filter, and again to clear.</p>
      <div className="heat" style={{ gridTemplateColumns: `72px repeat(${strs.length}, minmax(0, 1fr))` }}>
        <span />
        {strs.map((s) => <span key={s} className="heat-head">{s}</span>)}
        {orgs.map((o) => [
          <span key={o} className="heat-row">{o}</span>,
          ...strs.map((s) => {
            const c = get(o, s);
            return (
              <button key={o + s} type="button" className={"heat-cell" + (isGap(c, max) ? " gap" : "")} aria-label={`${o}, ${s}: ${c} papers${isGap(c, max) ? ", a research gap" : ""}`} aria-pressed={organism === o && stressor === s}
                onClick={() => onPick(o, s)} style={{ background: `rgba(255,150,40,${(0.08 + 0.62 * (c / max)).toFixed(2)})` }}>{c}</button>
            );
          }),
        ])}
      </div>
      {organism && stressor && (
        <div className="heat-pick">
          <span className="mut">{get(organism, stressor)} papers on {organism} and {stressor}</span>
          <Link className="chip" to={"/ask?q=" + encodeURIComponent(gapQuestion(organism, stressor))}>Ask what they found</Link>
        </div>
      )}
    </div>
  );
}
