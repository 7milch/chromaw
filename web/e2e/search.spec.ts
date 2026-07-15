import { test, expect, openSeedCollection } from "./fixtures";

test.describe("search", () => {
  test.beforeEach(async ({ page }) => {
    await openSeedCollection(page);
  });

  test("id search returns exactly the matching record", async ({ page }) => {
    await page.getByPlaceholder("id1, id2, ...").fill("rec-3");
    await page.getByRole("button", { name: "Search" }).click();

    await expect(page.getByRole("row", { name: /rec-3/ })).toBeVisible();
    await expect(page.getByRole("row", { name: /rec-0/ })).not.toBeVisible();
    await expect(page.getByText("1 match")).toBeVisible();
  });

  test("metadata search filters by key=value", async ({ page }) => {
    // Two <select> elements can be on screen at once (search mode, import
    // mode) -- disambiguate by an option only the search-mode select has.
    const searchModeSelect = page.locator("select", { has: page.locator('option[value="similarity"]') });
    await searchModeSelect.selectOption("metadata");
    await page.getByPlaceholder('key=value or {"key": "value"}').fill("category=odd");
    await page.getByRole("button", { name: "Search" }).click();

    // idx 1 and 3 are "odd" per the seed data (index % 2 == 1).
    await expect(page.getByRole("row", { name: /rec-1/ })).toBeVisible();
    await expect(page.getByRole("row", { name: /rec-3/ })).toBeVisible();
    await expect(page.getByRole("row", { name: /rec-0/ })).not.toBeVisible();
  });

  test("clear restores the full, unfiltered record list", async ({ page }) => {
    await page.getByPlaceholder("id1, id2, ...").fill("rec-3");
    await page.getByRole("button", { name: "Search" }).click();
    await expect(page.getByRole("row", { name: /rec-3/ })).toBeVisible();
    await expect(page.getByRole("row", { name: /rec-0/ })).not.toBeVisible();

    await page.getByRole("button", { name: "Clear" }).click();

    for (let i = 0; i < 5; i++) {
      await expect(page.getByRole("row", { name: new RegExp(`rec-${i}`) })).toBeVisible();
    }
  });
});
