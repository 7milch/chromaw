import { defineConfig, devices } from "@playwright/test";

/**
 * Playwright config for chromaw's UI e2e suite (M4-5, technical-spec §12).
 *
 * There is no `webServer` entry here: unlike a typical SPA, chromaw's
 * "server" is the real Python CLI (`uv run chromaw ... --write`) which also
 * injects the per-run bearer token into index.html, so a plain `vite`/`vite
 * preview` dev server would not exercise the auth flow the app actually
 * uses. Instead, e2e/global-setup.ts seeds a throwaway ChromaDB directory
 * and launches the real subprocess once for the whole run, publishing its
 * URL/token to a file that e2e/fixtures.ts's `page` fixture reads lazily at
 * test time (this config object is evaluated before globalSetup runs, so
 * `use.baseURL` can't be set here directly).
 */
export default defineConfig({
  testDir: "./e2e",
  fullyParallel: false,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  workers: 1,
  reporter: process.env.CI ? [["github"], ["list"]] : "list",
  globalSetup: "./e2e/global-setup.ts",
  use: {
    trace: "on-first-retry",
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
});
