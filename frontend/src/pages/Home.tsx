import { FormEvent, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { motion } from "motion/react";
import { Arrow, CountUp, EASE, Reveal, SPRING } from "../components/ui";
import { useStats } from "../useStats";

const TOOLS: [string, string, string][] = [["/search", "Search", "Compare four retrievers"], ["/explore", "Explore", "Map papers and research gaps"],
  ["/analyze", "Analyze", "Tag a passage"], ["/results", "Results", "See every score"]];
const LINES = [["Ask", "space", "biology."], ["Get", "the", "evidence."]];
const ask = (q: string) => "/ask?q=" + encodeURIComponent(q);
const rise = (delay: number) => ({ initial: { opacity: 0, y: 18 }, animate: { opacity: 1, y: 0 }, transition: { duration: 0.8, ease: EASE, delay } });
const MotionLink = motion.create(Link);

export default function Home() {
  const stats = useStats();
  const go = useNavigate();
  const [q, setQ] = useState("");
  const submit = (e: FormEvent) => { e.preventDefault(); if (q.trim()) go(ask(q)); };

  return (
    <main className="descent">
      <section className="stage hero">
        <h1 aria-label="Ask space biology. Get the evidence.">
          {LINES.map((line, i) => (
            <span key={i} className="hero-line" aria-hidden="true">
              {line.map((word, j) => (
                <motion.span key={word} className="hero-word" initial={{ opacity: 0, y: 34, filter: "blur(10px)" }}
                  animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
                  transition={{ duration: 1, ease: EASE, delay: 0.2 + (i * 3 + j) * 0.075 }}>{word}{" "}</motion.span>
              ))}
            </span>
          ))}
        </h1>
        <motion.p className="lede" {...rise(0.7)}>Every answer shows the passage it came from. When the evidence is weak, CosmoCite refuses to answer.</motion.p>
        <motion.div className="cta" {...rise(0.85)}>
          <MotionLink className="pill blue" to="/ask" whileHover={{ scale: 1.04 }} whileTap={{ scale: 0.97 }} transition={SPRING}>Ask a question <Arrow /></MotionLink>
          <MotionLink className="pill white" to="/explore" whileHover={{ scale: 1.04 }} whileTap={{ scale: 0.97 }} transition={SPRING}>Explore papers</MotionLink>
        </motion.div>
        <motion.p className="cue" {...rise(1.1)}>Scroll to descend</motion.p>
      </section>

      <section className="stage">
        <Reveal className="stage-copy">
          <h2>Every paper is a point of light.</h2>
          <p className="lede">NASA's space-biology publications, read in full and split into passages you can search, map and question.</p>
          {stats && (
            <div className="stats">
              <div><CountUp to={stats.papers} /><span className="mut">papers read in full</span></div>
              <div><CountUp to={stats.passages} /><span className="mut">passages to search</span></div>
            </div>
          )}
        </Reveal>
      </section>

      <section className="stage">
        <div className="stage-copy">
          <Reveal>
            <h2>It answers. It also refuses.</h2>
          </Reveal>
          <Reveal delay={0.05}>
            <article className="card demo">
              <p className="mut">How does microgravity affect bone density in mice?</p>
              <p className="claim">The increase in bone volume reached significance in the mice exposed to microgravity while the change in cross-sectional thickness did not, possibly due to bone expansion into the cavities of the skull and the sagittal suture. <em>[1]</em></p>
              <Link className="more" to={ask("How does microgravity affect bone density in mice?")}>Try this question <Arrow /></Link>
            </article>
          </Reveal>
          <Reveal delay={0.13}>
            <article className="card demo refusal">
              <p className="mut">Does spaceflight increase dandruff in astronauts?</p>
              <p className="claim">Insufficient evidence in the corpus to answer this question.</p>
              <Link className="more" to={ask("Does spaceflight increase dandruff in astronauts?")}>See it refuse <Arrow /></Link>
            </article>
          </Reveal>
        </div>
      </section>

      <section className="stage horizon">
        <Reveal className="horizon-copy">
          <h2>Cross the horizon.</h2>
          <form className="composer" onSubmit={submit}>
            <label className="sr" htmlFor="home-q">Your question</label>
            <input id="home-q" value={q} onChange={(e) => setQ(e.target.value)} placeholder="Ask about space biology…" autoComplete="off" />
            <button className="send" type="submit" disabled={!q.trim()} aria-label="Ask"><Arrow /></button>
          </form>
          <nav className="horizon-links" aria-label="Tools">
            {TOOLS.map(([to, label, what]) => <Link key={to} to={to}><b>{label}</b><span>{what}</span></Link>)}
          </nav>
        </Reveal>
      </section>
    </main>
  );
}
