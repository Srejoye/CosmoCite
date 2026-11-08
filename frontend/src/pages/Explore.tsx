import { useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { api, Gap, Paper } from "../api";
import { filtersFrom, filtersTo } from "../lib";
import PaperTable from "../components/PaperTable";
import TsneMap from "../components/TsneMap";
import GapHeatmap from "../components/GapHeatmap";
import { PageHead, Reveal, Skeleton } from "../components/ui";

const ORG = ["mouse", "human", "plant", "cells", "other"];
const STR = ["microgravity", "radiation", "other"];
const EMPTY = { organism: "", stressor: "", tissue: "", year_min: 0, q: "" };
const PAGE = 24;

export default function Explore() {
  const [all, setAll] = useState<Paper[]>([]);
  const [shown, setShown] = useState<Paper[]>([]);
  const [gaps, setGaps] = useState<Gap[]>([]);
  const [params, setParams] = useSearchParams();
  const [f, setF] = useState(() => filtersFrom(params));
  const [sel, setSel] = useState<string | null>(null);
  const [limit, setLimit] = useState(PAGE);
  const [err, setErr] = useState("");

  useEffect(() => {
    api.papers({}).then(setAll).catch((e) => setErr(e.message));
    api.gaps().then(setGaps).catch(() => {});
  }, []);
  useEffect(() => {
    setLimit(PAGE);
    setParams(filtersTo(f), { replace: true }); // a filtered view can be shared as a link
    api.papers(f).then(setShown).catch((e) => setErr(e.message));
  }, [f]);

  const tissues = useMemo(() => [...new Set(all.flatMap((p) => p.entities.tissue))].sort(), [all]);
  const active = useMemo(() => new Set(shown.map((p) => p.paper_id)), [shown]);
  const list = sel ? [...shown].sort((a, b) => (a.paper_id === sel ? -1 : b.paper_id === sel ? 1 : 0)) : shown;
  const set = (k: string, v: string | number) => setF((s) => ({ ...s, [k]: v }));
  const filtered = JSON.stringify(f) !== JSON.stringify(EMPTY) || sel !== null;
  const clear = () => { setF(EMPTY); setSel(null); };
  const pin = (id: string) => setSel((s) => (s === id ? null : id));
  const pickCell = (o: string, s: string) => setF((x) => (x.organism === o && x.stressor === s ? { ...x, organism: "", stressor: "" } : { ...x, organism: o, stressor: s }));
  const pickList = (k: "organism" | "stressor" | "tissue", opts: string[]) => (
    <select aria-label={k} value={f[k]} onChange={(e) => set(k, e.target.value)}>
      <option value="">Any {k}</option>
      {opts.map((o) => <option key={o}>{o}</option>)}
    </select>
  );

  return (
    <>
      <PageHead title="The paper constellation">Filter {all.length ? all.length.toLocaleString("en-US") : "the"} papers, or click the map and the gap grid to find what is under-studied.</PageHead>
      {err && <p className="card warn">{err}</p>}
      <div className="toolbar">
        <input className="wide" aria-label="Title contains" placeholder="Title contains…" value={f.q} onChange={(e) => set("q", e.target.value)} />
        {pickList("organism", ORG)}{pickList("stressor", STR)}{pickList("tissue", tissues)}
        <input type="number" aria-label="From year" placeholder="From year" style={{ width: 130 }} value={f.year_min || ""} onChange={(e) => set("year_min", Number(e.target.value) || 0)} />
        {filtered && <button type="button" className="chip" onClick={clear}>Clear filters</button>}
      </div>
      <Reveal className="grid">
        {gaps.length ? <GapHeatmap gaps={gaps} organism={f.organism} stressor={f.stressor} onPick={pickCell} /> : <div className="card"><Skeleton lines={5} /></div>}
        {all.length ? <TsneMap papers={all} active={active} selected={sel} onPick={pin} /> : <div className="card"><Skeleton lines={5} /></div>}
      </Reveal>
      <p className="mut count">Showing {Math.min(limit, list.length)} of {shown.length === all.length ? `${all.length} papers` : `${shown.length} matching papers`}. Click a tag to filter by it.</p>
      <PaperTable papers={list.slice(0, limit)} selected={sel} onTag={set} onClear={clear} />
      {limit < list.length && <div className="center"><button type="button" className="chip" onClick={() => setLimit(limit + PAGE)}>Show {Math.min(PAGE, list.length - limit)} more</button></div>}
    </>
  );
}
