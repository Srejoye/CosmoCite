import { KeyboardEvent, useState } from "react";
import type { Paper } from "../api";
import { nextStar } from "../lib";

const ARROWS: Record<string, number[]> = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] };

const toPaper = (id: string) => document.getElementById(`paper-${id}`)?.scrollIntoView({ block: "center", behavior: "smooth" });

export default function TsneMap({ papers, active, selected, onPick }: { papers: Paper[]; active: Set<string>; selected?: string | null; onPick: (id: string) => void }) {
  const [hover, setHover] = useState<Paper | null>(null);
  const [cursor, setCursor] = useState(-1);
  const pts = papers.filter((p) => p.tsne);
  const pinned = papers.find((p) => p.paper_id === selected);
  if (!pts.length) return <p className="mut">No t-SNE coordinates yet.</p>;
  const xs = pts.map((p) => p.tsne![0]), ys = pts.map((p) => p.tsne![1]);
  const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  const W = 560, H = 340, pad = 14;
  const stars = pts.map((p) => ({ x: p.tsne![0], y: p.tsne![1] }));
  const keys = (e: KeyboardEvent) => {
    if (e.key in ARROWS) {
      e.preventDefault();
      const n = cursor < 0 ? 0 : nextStar(stars, cursor, ARROWS[e.key]);
      setCursor(n);
      setHover(pts[n]);
    } else if ((e.key === "Enter" || e.key === " ") && cursor >= 0) {
      e.preventDefault();
      onPick(pts[cursor].paper_id);
    }
  };
  const sx = (x: number) => pad + ((x - x0) / (x1 - x0 || 1)) * (W - 2 * pad);
  const sy = (y: number) => pad + ((y - y0) / (y1 - y0 || 1)) * (H - 2 * pad);
  return (
    <div className="card">
      <h2 className="h3">Paper map</h2>
      <p className="mut hovertitle" aria-live="polite">{hover ? hover.title : pinned ? pinned.title : "Papers placed by similarity. Hover a star to read its title, and click to pin it. On a keyboard, use the arrow keys and Enter."}</p>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", display: "block" }} role="application" aria-label="Map of papers by similarity. Use the arrow keys to move between papers, and Enter to pin one." tabIndex={0} onKeyDown={keys} onBlur={() => setHover(null)}>
        {pts.map((p, i) => {
          const on = active.has(p.paper_id), picked = p.paper_id === selected;
          return (
            <circle key={p.paper_id} className={"star" + (picked ? " picked" : on ? "" : " off") + (i === cursor ? " cursor" : "")} cx={sx(p.tsne![0])} cy={sy(p.tsne![1])} r={picked ? 6 : on ? 3 : 2}
              onMouseEnter={() => setHover(p)} onMouseLeave={() => setHover(null)} onClick={() => onPick(p.paper_id)} />
          );
        })}
      </svg>
      {pinned && (
        <div className="heat-pick">
          <button type="button" className="chip" onClick={() => toPaper(pinned.paper_id)}>Go to this paper</button>
          <button type="button" className="chip" onClick={() => onPick(pinned.paper_id)}>Unpin</button>
        </div>
      )}
    </div>
  );
}
