import { describe, expect, it } from "vitest";
import { depthFor, scrollDepth } from "./depth";

describe("depthFor", () => {
  it("lets the home page follow the scroll", () => {
    expect(depthFor("/")).toBeNull();
  });
  it("puts Ask deepest, near the event horizon", () => {
    const pages = ["/explore", "/search", "/analyze", "/results"].map((p) => depthFor(p)!);
    expect(Math.max(...pages)).toBeLessThan(depthFor("/ask")!);
  });
  it("gives unknown pages a safe mid-flight depth", () => {
    expect(depthFor("/nope")).toBe(0.4);
  });
});

describe("scrollDepth", () => {
  it("runs from Earth to the horizon across the page", () => {
    expect(scrollDepth(0, 1000)).toBe(0);
    expect(scrollDepth(500, 1000)).toBe(0.5);
    expect(scrollDepth(1000, 1000)).toBe(1);
  });
  it("stays in range when the page overscrolls", () => {
    expect(scrollDepth(-40, 1000)).toBe(0);
    expect(scrollDepth(1200, 1000)).toBe(1);
  });
  it("handles a page too short to scroll", () => {
    expect(scrollDepth(0, 0)).toBe(0);
  });
});
