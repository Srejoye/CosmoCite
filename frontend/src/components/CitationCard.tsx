import { highlightParts, paperUrl } from "../lib";

type C = { title: string; section: string; pmc_id?: string; chunk_id?: string; passage?: string; spans?: [number, number][] };

export default function CitationCard({ c, n, text, note, active, limit = 300, id }: { c: C; n?: number; text?: string; note?: string; active?: boolean; limit?: number; id?: string }) {
  const url = paperUrl(c.pmc_id ?? "");
  const cut = !!text && text.length > limit;
  return (
    <div id={id} className={"card source" + (active ? " on" : "")}>
      <h3>{n !== undefined && <span className="cite static">{n}</span>}{c.title}</h3>
      <p className="mut">{c.section}{note && <> · {note}</>}</p>
      {text && <p className="quote">{cut ? text.slice(0, limit) + "…" : text}</p>}
      {c.passage && c.spans && (
        <details className="context">
          <summary>Show in context</summary>
          <p className="quote">{highlightParts(c.passage, c.spans).map((p, i) => (p.hit ? <mark key={i}>{p.text}</mark> : <span key={i}>{p.text}</span>))}</p>
        </details>
      )}
      {cut && !c.passage && (
        <details className="context">
          <summary>Read the full passage</summary>
          <p className="quote">{text}</p>
        </details>
      )}
      {url && <a className="paper-link" href={url} target="_blank" rel="noopener noreferrer">Read the paper<span className="sr"> (opens in a new tab)</span></a>}
    </div>
  );
}
