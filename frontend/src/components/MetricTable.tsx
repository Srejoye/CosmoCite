import { cellValue, columnLabel, ranked } from "../lib";

type Row = Record<string, any>;
const isScalar = (v: any) => ["number", "string", "boolean"].includes(typeof v);

export default function MetricTable({ title, note, rows }: { title: string; note?: string; rows: Row[] }) {
  if (!rows.length) return null;
  const cols = [...new Set(rows.flatMap((r) => Object.keys(r).filter((k) => isScalar(r[k]))))];
  const best = (c: string) => {
    const nums = rows.map((r) => r[c]).filter((v) => typeof v === "number");
    return rows.length > 1 && nums.length > 1 && ranked(c) ? Math.max(...nums) : null;
  };
  const top = Object.fromEntries(cols.map((c) => [c, best(c)]));
  return (
    <div className="card">
      <h2 className="h3">{title}</h2>
      {note && <p className="mut">{note}</p>}
      <div className="tablewrap" tabIndex={0} role="region" aria-label={title}>
        <table>
          <thead><tr>{cols.map((c) => <th key={c} scope="col">{columnLabel(c)}</th>)}</tr></thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={i}>{cols.map((c) => <td key={c} className={r[c] === top[c] ? "best" : undefined}>{r[c] === undefined ? "–" : cellValue(c, r[c])}</td>)}</tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
