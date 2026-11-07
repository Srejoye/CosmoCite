import { defineConfig } from "@playwright/test";

// CI installs Chromium. Locally the tests use Edge, so nothing extra is downloaded.
export default defineConfig({
  testDir: "e2e",
  outputDir: "node_modules/.playwright-results",
  timeout: 30_000,
  use: {
    baseURL: "http://127.0.0.1:4173",
    channel: process.env.CI ? undefined : "msedge",
    contextOptions: { reducedMotion: "reduce" },
    launchOptions: { args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"] },
  },
  webServer: { command: "npm run preview -- --host 127.0.0.1 --port 4173", url: "http://127.0.0.1:4173", reuseExistingServer: true },
});
