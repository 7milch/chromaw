import { apiFetch } from "./api";
import type { RecordInfo, RecordsGetRequest, RecordsResponse } from "./types";

/** Filters that scope an export to a subset of a collection's records. */
export interface ExportFilters {
  ids?: string[];
  where?: Record<string, unknown>;
  where_document?: Record<string, unknown>;
}

export interface ExportResult {
  collection: string;
  exported_at: string;
  filters: ExportFilters | null;
  records: RecordInfo[];
  truncated: boolean;
}

const EXPORT_PAGE_LIMIT = 500;
const EXPORT_MAX_RECORDS = 100_000;

/**
 * Walks `POST .../records/get` with `offset`/`limit=500`, collecting every
 * record that matches `filters` (or the whole collection when `filters` is
 * omitted/empty) until `has_more` is false. Reports progress via
 * `onProgress(collected)` after each page and stops early - marking the
 * result `truncated` - once 100,000 records have been collected. Pass
 * `signal` to cancel mid-walk; a cancelled walk rejects with the fetch's
 * AbortError.
 */
export async function exportCollectionRecords(
  collectionName: string,
  filters: ExportFilters | null,
  options: { signal?: AbortSignal; onProgress?: (collected: number) => void } = {}
): Promise<ExportResult> {
  const { signal, onProgress } = options;
  const records: RecordInfo[] = [];
  let offset = 0;
  let truncated = false;

  while (true) {
    if (signal?.aborted) {
      throw new DOMException("Export cancelled", "AbortError");
    }

    const body: RecordsGetRequest = {
      ...(filters?.ids !== undefined ? { ids: filters.ids } : {}),
      ...(filters?.where !== undefined ? { where: filters.where } : {}),
      ...(filters?.where_document !== undefined
        ? { where_document: filters.where_document }
        : {}),
      limit: EXPORT_PAGE_LIMIT,
      offset,
    };

    const res = await apiFetch(
      `/api/collections/${encodeURIComponent(collectionName)}/records/get`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal,
      }
    );

    if (!res.ok) {
      let detail = `request failed: ${res.status}`;
      const payload = await res.json().catch(() => null);
      if (payload && typeof payload.detail === "string") {
        detail = payload.detail;
      }
      throw new Error(detail);
    }

    const data = (await res.json()) as RecordsResponse;
    records.push(...data.records);
    onProgress?.(records.length);

    if (records.length >= EXPORT_MAX_RECORDS) {
      truncated = data.has_more;
      break;
    }

    if (!data.has_more || data.records.length === 0) {
      break;
    }

    offset += data.records.length;
  }

  return {
    collection: collectionName,
    exported_at: new Date().toISOString(),
    filters: filters && Object.keys(filters).length > 0 ? filters : null,
    records:
      records.length > EXPORT_MAX_RECORDS ? records.slice(0, EXPORT_MAX_RECORDS) : records,
    truncated,
  };
}
