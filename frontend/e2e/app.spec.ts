import { expect, Page, test } from "@playwright/test";

// The API is mocked so these tests need no models. They check what people click, not answer quality.
const cite = (n: number) => ({
  chunk_id: `c${n}`, paper_id: `000${n}`, pmc_id: `PMC${n}`, title: `Paper ${n}`, section: "Results", quote: `Quoted finding ${n}.`,
  passage: `Before. Quoted finding ${n}. After.`, spans: [[8, 8 + `Quoted finding ${n}.`.length]], year: 2014, authors: ["Smith A", "Jones B", "Lee C"], journal: "Bone",
});
const ANSWER = { answer: "Quoted finding 1. [1] Quoted finding 2. [2]", sufficient: true, citations: [cite(1), cite(2)], decoding: "greedy", engine: "rank", reason: null,
  evidence: { papers: 2, year_min: 2014, year_max: 2020, organisms: { mouse: 2 }, stressors: {}, tissues: {} }, closest: [] };
const REFUSED = { answer: "Insufficient evidence in the corpus to answer this question.", sufficient: false, citations: [], decoding: "greedy", engine: "rank", reason: "gate", evidence: null };
const hit = (id: string, title: string) => ({ chunk_id: id, paper_id: id, pmc_id: "", title, section: "Results", text: "Passage text.", score: 6.214, dense: 0.5, year: 2020, authors: [], journal: "" });
const paper = (id: string, organism: string) => ({ paper_id: id, pmc_id: "", title: `Title ${id}`, year: 2020, summary: "", tsne: [Number(id), Number(id)],
  entities: { organism: [organism], stressor: ["radiation"], tissue: [], outcome: [] } });
const PAPERS = [paper("1", "mouse"), paper("2", "plant"), paper("3", "mouse")];
const GAPS = [{ organism: "mouse", stressor: "radiation", count: 2, paper_ids: ["1", "3"] }, { organism: "plant", stressor: "radiation", count: 1, paper_ids: ["2"] }];
const RESULTS = { images: [], abstention: [{ set: "off-topic", questions: 10, correct: 1 }, { set: "test: answerable (should answer)", questions: 15, correct: 14 / 15 }],
  generation: { rank: { abstention_rate: 0.5, refused_by_gate: 4, refused_by_filter: 1, multi_paper_rate: 0.56 } }, relevance_check: { after: { on_topic: 49, sentences: 61 } } };

async function mock(page: Page) {
  await page.route("**/api/**", async (route) => {
    const url = new URL(route.request().url()), path = url.pathname.replace("/api", "");
    const json = (body: unknown) => route.fulfill({ json: body });
    if (path === "/stats") return json({ papers: 572, passages: 20242 });
    if (path === "/gaps") return json(GAPS);
    if (path === "/results") return json(RESULTS);
    if (path === "/papers") return json(PAPERS.filter((p) => !url.searchParams.get("organism") || p.entities.organism.includes(url.searchParams.get("organism")!)));
    if (path === "/search") return json({ bm25: [hit("a", "Shared"), hit("b", "Lonely")], dense: [hit("a", "Shared")], hybrid: [hit("a", "Shared")], rerank: [hit("a", "Shared")] });
    if (path === "/ask") {
      const q: string = route.request().postDataJSON().question;
      if (q.includes("capital")) return json({ ...REFUSED, closest: [] });
      if (q.includes("vitamin")) return json({ ...REFUSED, closest: [{ paper_id: "0009", pmc_id: "PMC9", title: "A close paper" }] });
      return json(ANSWER);
    }
    return route.fulfill({ status: 404, json: {} });
  });
}

test.beforeEach(async ({ page }) => mock(page));

test("a citation number picks out its source", async ({ page }) => {
  await page.goto("/ask?q=bone");
  await page.getByRole("button", { name: "Show source 2" }).click();
  await expect(page.locator("#source-2")).toHaveClass(/on/);
  await expect(page.locator("#source-1")).toContainText("Smith et al., 2014 · Bone");
});

test("show in context highlights the quote", async ({ page }) => {
  await page.goto("/ask?q=bone");
  await page.locator("#source-1").getByText("Show in context").click();
  await expect(page.locator("#source-1 mark")).toHaveText("Quoted finding 1.");
});

test("a near-topic refusal offers papers and a way to search", async ({ page }) => {
  await page.goto("/ask?q=vitamin");
  await expect(page.getByText("A close paper")).toBeVisible();
  await page.getByRole("link", { name: "Search the passages yourself" }).click();
  await expect(page).toHaveURL(/\/search\?q=vitamin/);
});

test("an off-topic refusal offers sample questions instead", async ({ page }) => {
  await page.goto("/ask?q=capital");
  await expect(page.getByText(/outside NASA/)).toBeVisible();
  await expect(page.locator(".refusal .chip")).toHaveCount(3);
});

test("search shows how many retrievers agree, not raw scores", async ({ page }) => {
  await page.goto("/search?q=bone");
  await expect(page.getByText("Found by 4 of 4").first()).toBeVisible();
  await expect(page.getByText("Only this retriever")).toBeVisible();
  await expect(page.getByText("6.214")).toHaveCount(0);
});

test("clicking a tag filters and the link can be shared", async ({ page }) => {
  await page.goto("/explore");
  await page.getByRole("button", { name: "Filter by organism: plant" }).click();
  await expect(page.locator(".paper")).toHaveCount(1);
  await expect(page).toHaveURL(/organism=plant/);
  await page.reload();
  await expect(page.locator(".paper")).toHaveCount(1);
});

test("a heat cell toggles its filter", async ({ page }) => {
  await page.goto("/explore");
  const cell = page.getByRole("button", { name: /^mouse, radiation/ });
  await cell.click();
  await expect(page.getByRole("link", { name: "Ask what they found" })).toBeVisible();
  await cell.click();
  await expect(page.getByRole("link", { name: "Ask what they found" })).toHaveCount(0);
});

test("the paper map works from the keyboard", async ({ page }) => {
  await page.goto("/explore");
  await expect(page.locator("circle.star")).toHaveCount(3);
  await page.getByRole("application").focus();
  await page.keyboard.press("ArrowRight");
  await expect(page.locator(".hovertitle")).toHaveText("Title 1");
  await page.keyboard.press("ArrowRight");
  await expect(page.locator(".hovertitle")).toHaveText("Title 2");
  await page.keyboard.press("Enter");
  await expect(page.locator(".paper").first()).toContainText("Title 2");
});

test("results lead with the headline numbers in plain words", async ({ page }) => {
  await page.goto("/results");
  await expect(page.locator(".headline").first()).toContainText("10 of 10");
  await expect(page.getByText("Question set")).toBeVisible();
  await expect(page.getByText("abstention_rate")).toHaveCount(0);
});
