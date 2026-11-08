import type { Span } from "../api";

export const ENT_COLORS: Record<string, string> = { organism: "#ffffff", stressor: "#ffb340", tissue: "#8fb6ff", outcome: "#ffe0ad" };

export default function EntityHighlighter({ text, spans }: { text: string; spans: Span[] }) {
  const sorted = [...spans].sort((a, b) => a.start - b.start);
  const parts: JSX.Element[] = [];
  let pos = 0;
  sorted.forEach((s, i) => {
    if (s.start < pos) return;
    if (s.start > pos) parts.push(<span key={`t${i}`}>{text.slice(pos, s.start)}</span>);
    parts.push(
      <mark key={`m${i}`} className="ent" style={{ background: ENT_COLORS[s.type] ?? "#ccc" }} title={s.type}>
        {text.slice(s.start, s.end)}
      </mark>
    );
    pos = s.end;
  });
  if (pos < text.length) parts.push(<span key="end">{text.slice(pos)}</span>);
  return (
    <>
      <div className="row" style={{ margin: "8px 0 12px" }}>
        {Object.entries(ENT_COLORS).map(([k, c]) => (
          <mark key={k} className="ent" style={{ background: c }}>{k}</mark>
        ))}
      </div>
      <p style={{ whiteSpace: "pre-wrap" }}>{parts}</p>
    </>
  );
}