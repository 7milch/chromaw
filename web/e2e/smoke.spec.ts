import { test, expect, openSeedCollection } from "./fixtures";

test.describe("smoke", () => {
  test("loads the page, shows the seeded collection, and opens a record", async ({ page }) => {
    await expect(page.getByText("chromaw", { exact: true })).toBeVisible();

    // Write mode badge (server started with --write in global-setup).
    await expect(page.getByText("write", { exact: true })).toBeVisible();

    await expect(page.getByRole("button", { name: /^e2e/ })).toBeVisible();
    await openSeedCollection(page);

    // Records table shows all five seeded rows.
    for (let i = 0; i < 5; i++) {
      await expect(page.getByRole("row", { name: new RegExp(`rec-${i}`) })).toBeVisible();
    }

    // Click a record row and confirm the detail pane (the right-hand
    // <aside>) populates with that record's full id/document.
    await page.getByRole("row", { name: /rec-2/ }).click();
    const detailPane = page.getByRole("complementary").last();
    await expect(detailPane.getByText("rec-2", { exact: true })).toBeVisible();
    await expect(detailPane.getByText("e2e document number 2")).toBeVisible();
  });
});
