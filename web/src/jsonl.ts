import { apiFetch } from "./api";
import type { ImportResponse } from "./types";

/**
 * Downloads `GET .../export.jsonl` (M4-3, technical-spec §8) as a Blob and
 * triggers a browser save via a throwaway `<a download>` element -- unlike
 * `exportCollectionRecords`/`exportSelectedRecords` (export.ts, which page
 * through `records/get` client-side into a single JSON array), this streams
 * straight from the server so it isn't subject to the 100,000-record client
 * cap.
 */
export async function downloadCollectionJsonl(collectionName: string): Promise<void> {
  const res = await apiFetch(
    `/api/collections/${encodeURIComponent(collectionName)}/export.jsonl`
  );

  if (!res.ok) {
    let detail = `request failed: ${res.status}`;
    const payload = await res.json().catch(() => null);
    if (payload && typeof payload.detail === "string") {
      detail = payload.detail;
    }
    throw new Error(detail);
  }

  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${collectionName}.jsonl`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

/**
 * Uploads `file` to `POST .../import` (M4-3, technical-spec §8) with the
 * given `mode` ("add" rejects rows whose id already exists; "upsert"
 * overwrites them). Returns the server's `{imported, skipped}` report even
 * on a non-empty `skipped` list -- that's still a 200; only a genuinely
 * failed request (network error, 4xx/5xx) throws.
 */
export async function importCollectionJsonl(
  collectionName: string,
  file: File,
  mode: "add" | "upsert"
): Promise<ImportResponse> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("mode", mode);

  const res = await apiFetch(
    `/api/collections/${encodeURIComponent(collectionName)}/import`,
    { method: "POST", body: formData }
  );

  if (!res.ok) {
    let detail = `request failed: ${res.status}`;
    const payload = await res.json().catch(() => null);
    if (payload && typeof payload.detail === "string") {
      detail = payload.detail;
    }
    throw new Error(detail);
  }

  return (await res.json()) as ImportResponse;
}
