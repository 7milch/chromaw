import { test, expect, openSeedCollection } from "./fixtures";

test.describe("edit + diff + save", () => {
  test("editing metadata shows a diff on confirm and persists on save", async ({ page }) => {
    await openSeedCollection(page);
    await page.getByRole("row", { name: /rec-0/ }).click();

    const detailPane = page.getByRole("complementary").last();
    const metadataEditor = detailPane.getByTestId("metadata-editor");
    await metadataEditor.getByRole("button", { name: "Edit", exact: true }).click();

    const textarea = metadataEditor.locator("textarea");
    const current = await textarea.inputValue();
    const parsed = JSON.parse(current) as Record<string, unknown>;
    parsed.category = "updated-by-e2e";
    await textarea.fill(JSON.stringify(parsed, null, 2));

    await metadataEditor.getByRole("button", { name: "Save", exact: true }).click();

    // Confirmation screen: shows a diff of the change before it's applied.
    await expect(metadataEditor.getByText("Confirm changes")).toBeVisible();
    await expect(metadataEditor.getByText(/updated-by-e2e/)).toBeVisible();

    await metadataEditor.getByRole("button", { name: "Confirm", exact: true }).click();

    // The metadata pane reflects the saved value once it's back in
    // read-only mode. (The "Record updated." toast fades after ~3s, so it's
    // not asserted here to avoid a race against that timer.)
    await expect(metadataEditor.getByRole("button", { name: "Edit", exact: true })).toBeVisible();
    await expect(metadataEditor.getByText(/"updated-by-e2e"/)).toBeVisible();
  });
});
