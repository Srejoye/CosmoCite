import { friendlyError } from "./lib";

const BASE = (import.meta as any).env?.VITE_API ?? "/api";

export type Entities = { organism: string[]; stressor: string[]; tissue: string[]; outcome: string[] };
export type Paper = { paper_id: string; pmc_id: string; title: string; year: number | null; summary: string; entities: Entities; tsne: [number, number] | null };
export type Gap = { organism: string; stressor: string; count: number; paper_ids: string[] };
export type Stats = { papers: number; passages: number };
export type Citation = { chunk_id: string; paper_id: string; pmc_id: string; title: string; section: string; quote: string; passage: string; spans: [number, number][]; year: number | null; authors: string[]; journal: string };
export type Lead = { paper_id: string; pmc_id: string; title: string; section?: string };
export type Evidence = { papers: number; year_min: number | null; year_max: number | null; organisms: Record<string, number>; stressors: Record<string, number>; tissues: Record<string, number> };
export type Answer = { answer: string; sufficient: boolean; citations: Citation[]; decoding: string; engine: "rank" | "synth" | "flan"; reason: "gate" | "filter" | null; evidence: Evidence | null; closest: Lead[] };
export type Hit = { chunk_id: string; paper_id: string; pmc_id: string; title: string; section: string; text: string; score: number; dense: number; year: number | null; authors: string[]; journal: string };
export type SearchRes = { bm25: Hit[]; dense: Hit[]; hybrid: Hit[]; rerank: Hit[] };
export type Span = { start: number; end: number; type: string; text: string };
export type Analysis = { label: string | null; model: string | null; probs: Record<string, number>; attention: { word: string; weight: number }[]; attention_model: string; attention_label: string; entities: Span[] };
export type Results = Record<string, any> & { images?: string[] };

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  let r: Response;
  try {
    r = await fetch(BASE + path, init);
  } catch (e: any) {
    if (e?.name === "AbortError") throw e;
    throw new Error(friendlyError(0));
  }
  if (!r.ok) throw new Error(friendlyError(r.status));
  return r.json();
}
const post = (body: unknown): RequestInit => ({ method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

export const api = {
  papers: (p: Record<string, string | number>) =>
    req<Paper[]>("/papers?" + new URLSearchParams(Object.entries(p).filter(([, v]) => v !== "" && v !== 0).map(([k, v]) => [k, String(v)]))),
  related: (id: string) => req<Lead[]>(`/papers/${id}/related?k=4`),
  gaps: () => req<Gap[]>("/gaps"),
  stats: () => req<Stats>("/stats"),
  ask: (question: string, decoding: string, engine: string, signal?: AbortSignal) => req<Answer>("/ask", { ...post({ question, decoding, engine }), signal }),
  search: (q: string) => req<SearchRes>("/search?" + new URLSearchParams({ q, k: "5" })),
  analyze: (text: string) => req<Analysis>("/analyze", post({ text })),
  results: () => req<Results>("/results"),
  img: (name: string) => `${BASE}/reports/${name}`,
};