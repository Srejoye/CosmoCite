import { useEffect, useRef } from "react";
import { COSMOS_FRAG, COSMOS_VERT } from "../cosmos";
import { scrollDepth } from "../depth";

/**
 * Live background, driven by data attributes on <html>:
 * data-depth flies the camera (a number, or "scroll" to follow the page), data-side moves the hole aside for text,
 * data-energy spins the disk up while a query runs, data-dim eases the scene back behind a page's content.
 * Resolution drops by itself on slower GPUs, so the page stays smooth.
 */
export default function Cosmos() {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const cv = ref.current;
    const gl = cv?.getContext("webgl", { antialias: false, powerPreference: "high-performance" });
    if (!cv || !gl) return;
    const compile = (type: number, src: string) => {
      const s = gl.createShader(type)!;
      gl.shaderSource(s, src);
      gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) console.error(gl.getShaderInfoLog(s));
      return s;
    };
    const pr = gl.createProgram()!;
    gl.attachShader(pr, compile(gl.VERTEX_SHADER, COSMOS_VERT));
    gl.attachShader(pr, compile(gl.FRAGMENT_SHADER, COSMOS_FRAG));
    gl.linkProgram(pr);
    if (!gl.getProgramParameter(pr, gl.LINK_STATUS)) return;
    gl.useProgram(pr);
    gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
    const loc = gl.getAttribLocation(pr, "a");
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
    const u = (n: string) => gl.getUniformLocation(pr, n);
    const U = { R: u("R"), T: u("T"), M: u("M"), E: u("E"), D: u("D"), Z: u("Z"), V: u("V"), X: u("X") };

    const still = matchMedia("(prefers-reduced-motion: reduce)").matches;
    const root = document.documentElement;
    let mx = 0.5, my = 0.5, sx = 0.5, sy = 0.5, en = 0, dim = Number(root.dataset.dim ?? 0), z = 0, vel = 0, side = 0, raf = 0;
    let quality = Math.min(devicePixelRatio || 1, 1.25), last = performance.now(), slow = 0;
    const onMove = (e: PointerEvent) => { mx = e.clientX / innerWidth; my = 1 - e.clientY / innerHeight; };
    if (!still) addEventListener("pointermove", onMove, { passive: true });
    const t0 = performance.now();

    const frame = () => {
      const now = performance.now(), dt = now - last;
      last = now;
      if (dt > 26 && dt < 200) slow++; else slow = Math.max(0, slow - 1);
      if (slow > 30 && quality > 0.55) { quality *= 0.82; slow = 0; }
      const w = Math.max(1, Math.floor(cv.clientWidth * quality)), h = Math.max(1, Math.floor(cv.clientHeight * quality));
      if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; gl.viewport(0, 0, w, h); }
      sx += (mx - sx) * 0.05; sy += (my - sy) * 0.05;
      const ease = (rate: number) => (still ? 1 : rate);
      en += (Number(root.dataset.energy ?? 0) - en) * ease(0.06);
      dim += (Number(root.dataset.dim ?? 0) - dim) * ease(0.05);
      const depth = root.dataset.depth;
      const want = depth === "scroll" ? scrollDepth(scrollY, root.scrollHeight - innerHeight) : Number(depth ?? 0);
      const step = (want - z) * ease(0.055);
      z += step;
      vel += (Math.min(Math.abs(step) * 60, 1) - vel) * 0.1;
      const asp = cv.clientWidth / Math.max(cv.clientHeight, 1);
      side += ((root.dataset.side !== "1" ? 0 : asp > 1.15 ? 0.3 * Math.min(asp, 1.8) : 0.14) - side) * ease(0.05);
      gl.uniform2f(U.R, w, h);
      gl.uniform1f(U.T, (now - t0) / 1000);
      gl.uniform2f(U.M, sx, sy);
      gl.uniform1f(U.E, en);
      gl.uniform1f(U.D, dim);
      gl.uniform1f(U.Z, z);
      gl.uniform1f(U.V, still ? 0 : vel);
      gl.uniform1f(U.X, side);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      if (!still) raf = requestAnimationFrame(frame);
    };
    const onVisible = () => {
      cancelAnimationFrame(raf);
      if (!document.hidden) { last = performance.now(); frame(); }
    };
    document.addEventListener("visibilitychange", onVisible);
    const redraw = () => requestAnimationFrame(frame);
    const watch = new MutationObserver(redraw);
    if (still) {
      addEventListener("scroll", redraw, { passive: true });
      watch.observe(root, { attributes: true, attributeFilter: ["data-depth", "data-dim", "data-side"] });
    }
    frame();
    return () => {
      watch.disconnect();
      removeEventListener("scroll", redraw);
      cancelAnimationFrame(raf);
      removeEventListener("pointermove", onMove);
      document.removeEventListener("visibilitychange", onVisible);
    };
  }, []);

  return <canvas ref={ref} className="cosmos" aria-hidden="true" />;
}
