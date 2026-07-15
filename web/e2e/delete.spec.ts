import { test, expect, openDeleteScratchCollection } from "./fixtures";

// Uses the dedicated "e2e-delete" scratch collection (see
// fixtures.ts/global-setup.ts) so these destructive operations never touch
// the "e2e" collection the other specs depend on.
test.describe("dangerous-operation confirmation guard", () => {
  test("deleting a record requires typing its exact id", async ({ page }) => {
    await openDeleteScratchCollection(page);
    await page.getByRole("row", { name: /del-1/ }).click();

    const detailPane = page.getByRole("complementary").last();
    await detailPane.getByRole("button", { name: "Delete", exact: true }).click();

    const dialog = page.getByRole("dialog", { name: "Delete record" });
    await expect(dialog).toBeVisible();
    const confirmButton = dialog.getByRole("button", { name: "Delete", exact: true });
    const input = dialog.locator("input[type='text']");

    // Wrong input keeps the confirm button disabled.
    await input.fill("del-1-typo");
    await expect(confirmButton).toBeDisabled();

    // Exact id enables it, and confirming actually removes the record.
    await input.fill("del-1");
    await expect(confirmButton).toBeEnabled();
    await confirmButton.click();

    await expect(dialog).not.toBeVisible();
    await expect(page.getByRole("row", { name: /del-1/ })).not.toBeVisible();
  });

  test("deleting a collection requires typing its exact name", async ({ page }) => {
    await openDeleteScratchCollection(page);

    // "Delete" (exact) also unambiguously distinguishes this action button
    // from the sidebar's "e2e-delete <count>" collection button, whose
    // accessible name otherwise contains "delete" as a substring.
    await page.getByRole("button", { name: "Delete", exact: true }).click();

    const dialog = page.getByRole("dialog", { name: "Delete collection" });
    await expect(dialog).toBeVisible();
    const confirmButton = dialog.getByRole("button", { name: "Delete", exact: true });
    const input = dialog.locator("input[type='text']");

    await input.fill("not-e2e-delete");
    await expect(confirmButton).toBeDisabled();

    await input.fill("e2e-delete");
    await expect(confirmButton).toBeEnabled();
    await confirmButton.click();

    await expect(dialog).not.toBeVisible();
    await expect(page.getByRole("button", { name: /^e2e-delete\s/ })).not.toBeVisible();
  });
});
