/** The site is one camera move: depth 0 is low Earth orbit, depth 1 is the event horizon. */
const PAGES: Record<string, number> = { "/explore": 0.34, "/search": 0.48, "/analyze": 0.58, "/results": 0.7, "/ask": 0.9 };

/** Fixed camera depth for a page. Null means the page drives depth with its scroll position. */
export const depthFor = (path: string): number | null => (path === "/" ? null : PAGES[path] ?? 0.4);

export const scrollDepth = (y: number, max: number) => (max > 0 ? Math.min(1, Math.max(0, y / max)) : 0);
