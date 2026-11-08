import { lazy, Suspense, useEffect, useRef } from "react";
import { Link, NavLink, Route, Routes, useLocation } from "react-router-dom";
import { AnimatePresence, MotionConfig, motion } from "motion/react";
import Lenis from "lenis";
import "lenis/dist/lenis.css";
import Cosmos from "./components/Cosmos";
import ErrorBoundary from "./components/ErrorBoundary";
import { depthFor } from "./depth";
import { pageTitle } from "./lib";
import { EASE, PageHead, Skeleton, SPRING } from "./components/ui";
import Home from "./pages/Home";

const Explore = lazy(() => import("./pages/Explore"));
const Ask = lazy(() => import("./pages/Ask"));
const Search = lazy(() => import("./pages/Search"));
const Analyze = lazy(() => import("./pages/Analyze"));
const Results = lazy(() => import("./pages/Results"));

const links: [string, string][] = [["/ask", "Ask"], ["/search", "Search"], ["/explore", "Explore"], ["/analyze", "Analyze"], ["/results", "Results"]];
const DIM: Record<string, string> = { "/": "0", "/ask": "0.8" };

export default function App() {
  const location = useLocation();
  const lenis = useRef<Lenis | null>(null);

  useEffect(() => {
    if (matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    lenis.current = new Lenis({ lerp: 0.1, autoRaf: true });
    return () => { lenis.current?.destroy(); lenis.current = null; };
  }, []);

  // One listener feeds the cursor spotlight on whichever card is under the pointer.
  useEffect(() => {
    const move = (e: PointerEvent) => {
      const card = (e.target as Element | null)?.closest?.<HTMLElement>(".card");
      if (!card) return;
      const r = card.getBoundingClientRect();
      card.style.setProperty("--mx", `${e.clientX - r.left}px`);
      card.style.setProperty("--my", `${e.clientY - r.top}px`);
    };
    addEventListener("pointermove", move, { passive: true });
    return () => removeEventListener("pointermove", move);
  }, []);

  useEffect(() => {
    const root = document.documentElement;
    root.dataset.dim = DIM[location.pathname] ?? "0.6";
    root.dataset.depth = String(depthFor(location.pathname) ?? "scroll");
    root.dataset.side = location.pathname === "/" ? "1" : "0";
    document.title = pageTitle(location.pathname);
  }, [location.pathname]);

  const focusMain = () => {
    const main = document.querySelector<HTMLElement>("main");
    main?.setAttribute("tabindex", "-1");
    main?.focus({ preventScroll: true });
  };
  const toTop = () => {
    lenis.current ? lenis.current.scrollTo(0, { immediate: true, force: true }) : scrollTo(0, 0);
    requestAnimationFrame(focusMain); // screen reader and keyboard users land on the new page
  };

  return (
    <MotionConfig reducedMotion="user">
      <Cosmos />
      <div className="shell">
        <button type="button" className="skip" onClick={focusMain}>Skip to content</button>
        <nav className="pillnav" aria-label="Main">
          <Link className="brand" to="/"><i aria-hidden="true" /><span className="brand-name">CosmoCite</span></Link>
          {links.map(([to, label]) => (
            <NavLink key={to} to={to} className="link">
              {({ isActive }) => (
                <>
                  {isActive && <motion.span layoutId="nav-pill" className="nav-pill" transition={SPRING} />}
                  <span className="link-text">{label}</span>
                </>
              )}
            </NavLink>
          ))}
        </nav>
        <AnimatePresence mode="wait" onExitComplete={toTop}>
          <motion.div key={location.pathname} className="route" initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }} transition={{ duration: 0.32, ease: EASE }}>
            <ErrorBoundary key={location.pathname}>
            <Suspense fallback={<main><div className="card"><Skeleton /></div></main>}>
            <Routes location={location}>
              <Route path="/" element={<Home />} />
              <Route path="/ask" element={<main><Ask /></main>} />
              <Route path="/search" element={<main className="wide"><Search /></main>} />
              <Route path="/explore" element={<main className="wide"><Explore /></main>} />
              <Route path="/analyze" element={<main><Analyze /></main>} />
              <Route path="/results" element={<main><Results /></main>} />
              <Route path="*" element={<main><PageHead title="Lost in space">That page does not exist.</PageHead><Link className="pill blue" to="/">Back to Earth</Link></main>} />
            </Routes>
            </Suspense>
            </ErrorBoundary>
            <footer className="mut foot">Data: NASA Space Biology publication list. Every answer cites its sources, and summaries are key sentences from each paper.</footer>
          </motion.div>
        </AnimatePresence>
      </div>
    </MotionConfig>
  );
}
