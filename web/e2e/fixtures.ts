import { readFileSync } from "node:fs";
import { test as base, expect, type Page } from "@playwright/test";
import { SERVER_INFO_PATH } from "./global-setup";

/**
 * chromaw injects its bearer token into `<meta name="chromaw-token">` on
 * index.html server-side (technical-spec §10.2) -- there's no login form to
 * drive, so tests just need to load the real page (which already carries
 * the token) rather than fake authentication.
 *
 * baseURL comes from the file global-setup.ts writes once the real chromaw
 * subprocess is up (see SERVER_INFO_PATH's docstring for why a file, not
 * playwright.config.ts's `use.baseURL`).
 */
function readServerInfo(): { baseURL: string; token: string } {
  return JSON.parse(readFileSync(SERVER_INFO_PATH, "utf-8"));
}

export const test = base.extend<{ baseURL: string }>({
  baseURL: async ({}, use) => {
    await use(readServerInfo().baseURL);
  },
  page: async ({ page, baseURL }, use) => {
    await page.goto(baseURL);
    await use(page);
  },
});

export { expect };

/** Opens the "e2e" seed collection (created by global-setup) and waits for
 * its records table to render. Used by every spec except delete.spec.ts,
 * which uses the separate "e2e-delete" scratch collection below so its
 * destructive operations can't affect this collection's fixed 5 records. */
export async function openSeedCollection(page: Page): Promise<void> {
  await openCollection(page, "e2e", /rec-0/);
}

/** Opens the "e2e-delete" scratch collection (created by global-setup) and
 * waits for its records table to render (delete.spec.ts only). */
export async function openDeleteScratchCollection(page: Page): Promise<void> {
  await openCollection(page, "e2e-delete", /del-0/);
}

async function openCollection(
  page: Page,
  name: string,
  firstRowPattern: RegExp
): Promise<void> {
  // "\s" (not "\b") after the name distinguishes "e2e" from "e2e-delete":
  // both share the "e2e" prefix, but only the exact match is followed by
  // whitespace before the record-count span in the button's accessible name.
  await page.getByRole("button", { name: new RegExp(`^${name}\\s`) }).click();
  await expect(page.getByRole("row", { name: firstRowPattern })).toBeVisible();
}
