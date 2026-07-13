import { useEffect, useState } from "react";
import { apiFetch } from "./api";
import type {
  CollectionInfo,
  CollectionsResponse,
  HealthResponse,
  RecordInfo,
  RecordsResponse,
} from "./types";

const PAGE_LIMIT = 50;

function summarizeMetadata(metadata: Record<string, unknown> | null): string {
  if (!metadata) return "-";
  const entries = Object.entries(metadata);
  if (entries.length === 0) return "-";
  return entries.map(([k, v]) => `${k}=${String(v)}`).join(", ");
}

function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [collections, setCollections] = useState<CollectionInfo[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedName, setSelectedName] = useState<string | null>(null);
  const [detailOpen, setDetailOpen] = useState(false);

  const [records, setRecords] = useState<RecordInfo[] | null>(null);
  const [recordsTotal, setRecordsTotal] = useState(0);
  const [offset, setOffset] = useState(0);
  const [recordsError, setRecordsError] = useState<string | null>(null);
  const [selectedRecordId, setSelectedRecordId] = useState<string | null>(null);

  useEffect(() => {
    apiFetch("/api/health")
      .then((res) => {
        if (!res.ok) {
          throw new Error(`request failed: ${res.status}`);
        }
        return res.json() as Promise<HealthResponse>;
      })
      .then(setHealth)
      .catch((err: unknown) => {
        setError(err instanceof Error ? err.message : String(err));
      });

    apiFetch("/api/collections")
      .then((res) => {
        if (!res.ok) {
          throw new Error(`request failed: ${res.status}`);
        }
        return res.json() as Promise<CollectionsResponse>;
      })
      .then((data) => setCollections(data.collections))
      .catch((err: unknown) => {
        setError(err instanceof Error ? err.message : String(err));
      });
  }, []);

  // Reset paging/selection whenever the selected collection changes.
  useEffect(() => {
    setOffset(0);
    setRecordsTotal(0);
    setSelectedRecordId(null);
    setRecords(null);
    setRecordsError(null);
  }, [selectedName]);

  useEffect(() => {
    if (!selectedName) return;

    // Guard against out-of-order responses: if selectedName/offset change
    // again before this request resolves, ignore its result instead of
    // clobbering state set by the newer request.
    let ignore = false;

    const params = new URLSearchParams({
      include: "documents,metadatas,uris",
      limit: String(PAGE_LIMIT),
      offset: String(offset),
    });

    apiFetch(`/api/collections/${encodeURIComponent(selectedName)}/records?${params}`)
      .then((res) => {
        if (!res.ok) {
          throw new Error(`request failed: ${res.status}`);
        }
        return res.json() as Promise<RecordsResponse>;
      })
      .then((data) => {
        if (ignore) return;
        setRecords(data.records);
        setRecordsTotal(data.total);
      })
      .catch((err: unknown) => {
        if (ignore) return;
        setRecordsError(err instanceof Error ? err.message : String(err));
      });

    return () => {
      ignore = true;
    };
  }, [selectedName, offset]);

  const selected = collections?.find((c) => c.name === selectedName) ?? null;

  const rangeStart = recordsTotal === 0 ? 0 : offset + 1;
  const rangeEnd = records ? offset + records.length : 0;
  const canPrev = offset > 0;
  const canNext = offset + PAGE_LIMIT < recordsTotal;

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col">
      <header className="flex items-center gap-3 border-b border-slate-800 px-4 py-2 text-sm">
        <span className="font-semibold text-slate-100">chromaw</span>
        <span className="text-slate-500">|</span>
        <span className="text-slate-400 truncate">{health?.path ?? "..."}</span>
        <span className="text-slate-500">|</span>
        {health && (
          <span
            className={`rounded-full px-2 py-0.5 text-xs font-medium ${
              health.mode === "write"
                ? "bg-amber-500/20 text-amber-300"
                : "bg-emerald-500/20 text-emerald-300"
            }`}
          >
            {health.mode}
          </span>
        )}
      </header>

      {error && (
        <p className="px-4 py-2 text-sm text-red-400">Failed to reach API: {error}</p>
      )}

      <div className="flex flex-1 min-h-0">
        <aside className="w-64 shrink-0 border-r border-slate-800 overflow-y-auto">
          <h2 className="px-3 py-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
            Collections
          </h2>
          {collections === null && !error && (
            <p className="px-3 py-2 text-sm text-slate-400">Loading...</p>
          )}
          {collections !== null && collections.length === 0 && (
            <p className="px-3 py-2 text-sm text-slate-400">No collections found.</p>
          )}
          <ul>
            {collections?.map((c) => (
              <li key={c.id}>
                <button
                  type="button"
                  onClick={() => setSelectedName(c.name)}
                  className={`flex w-full items-center justify-between px-3 py-2 text-left text-sm hover:bg-slate-800 ${
                    c.name === selectedName ? "bg-slate-800 text-slate-50" : "text-slate-300"
                  }`}
                >
                  <span className="truncate">{c.name}</span>
                  <span className="ml-2 shrink-0 text-xs text-slate-500">{c.count}</span>
                </button>
              </li>
            ))}
          </ul>
        </aside>

        <main className="flex-1 min-w-0 overflow-y-auto p-4 flex flex-col gap-3">
          {!selected && (
            <p className="text-sm text-slate-400">
              Select a collection from the left to see its records.
            </p>
          )}

          {selected && (
            <>
              <div className="rounded border border-slate-800">
                <button
                  type="button"
                  onClick={() => setDetailOpen((v) => !v)}
                  className="flex w-full items-center justify-between px-3 py-2 text-left text-sm"
                >
                  <span className="font-semibold">{selected.name}</span>
                  <span className="text-xs text-slate-500">
                    {detailOpen ? "hide details ▲" : "show details ▼"}
                  </span>
                </button>
                {detailOpen && (
                  <div className="space-y-3 border-t border-slate-800 px-3 py-3">
                    <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-sm">
                      <dt className="text-slate-400">id</dt>
                      <dd className="break-all">{selected.id}</dd>
                      <dt className="text-slate-400">count</dt>
                      <dd>{selected.count}</dd>
                      <dt className="text-slate-400">dimension</dt>
                      <dd>{selected.dimension ?? "-"}</dd>
                    </dl>
                    <div>
                      <h3 className="mb-1 text-xs font-semibold uppercase tracking-wide text-slate-500">
                        metadata
                      </h3>
                      <pre className="overflow-x-auto rounded bg-slate-900 p-3 text-xs text-slate-300">
                        {JSON.stringify(selected.metadata, null, 2)}
                      </pre>
                    </div>
                  </div>
                )}
              </div>

              {recordsError && (
                <p className="text-sm text-red-400">Failed to load records: {recordsError}</p>
              )}

              {!recordsError && records === null && (
                <p className="text-sm text-slate-400">Loading records...</p>
              )}

              {!recordsError && records !== null && records.length === 0 && (
                <p className="text-sm text-slate-400">No records in this collection.</p>
              )}

              {!recordsError && records !== null && records.length > 0 && (
                <div className="flex flex-1 min-h-0 flex-col gap-2">
                  <div className="min-h-0 flex-1 overflow-auto rounded border border-slate-800">
                    <table className="w-full border-collapse text-sm">
                      <thead className="sticky top-0 bg-slate-900 text-xs uppercase tracking-wide text-slate-500">
                        <tr>
                          <th className="px-3 py-2 text-left font-semibold">id</th>
                          <th className="px-3 py-2 text-left font-semibold">document</th>
                          <th className="px-3 py-2 text-left font-semibold">metadata</th>
                        </tr>
                      </thead>
                      <tbody>
                        {records.map((r) => (
                          <tr
                            key={r.id}
                            onClick={() => setSelectedRecordId(r.id)}
                            className={`cursor-pointer border-t border-slate-800 hover:bg-slate-800/60 ${
                              r.id === selectedRecordId ? "bg-slate-800 text-slate-50" : ""
                            }`}
                          >
                            <td className="max-w-[10rem] truncate px-3 py-1.5 align-top font-mono text-xs">
                              {r.id}
                            </td>
                            <td className="max-w-xs truncate px-3 py-1.5 align-top">
                              {r.document ?? "-"}
                            </td>
                            <td className="max-w-xs truncate px-3 py-1.5 align-top text-slate-400">
                              {summarizeMetadata(r.metadata)}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>

                  <div className="flex items-center justify-between text-xs text-slate-400">
                    <span>
                      {rangeStart}–{rangeEnd} / {recordsTotal}
                    </span>
                    <div className="flex gap-2">
                      <button
                        type="button"
                        disabled={!canPrev}
                        onClick={() => setOffset((o) => Math.max(0, o - PAGE_LIMIT))}
                        className="rounded border border-slate-700 px-2 py-1 disabled:cursor-not-allowed disabled:opacity-40 hover:bg-slate-800"
                      >
                        Prev
                      </button>
                      <button
                        type="button"
                        disabled={!canNext}
                        onClick={() => setOffset((o) => o + PAGE_LIMIT)}
                        className="rounded border border-slate-700 px-2 py-1 disabled:cursor-not-allowed disabled:opacity-40 hover:bg-slate-800"
                      >
                        Next
                      </button>
                    </div>
                  </div>
                </div>
              )}
            </>
          )}
        </main>

        <aside className="w-72 shrink-0 border-l border-slate-800 overflow-y-auto p-3">
          <h2 className="text-xs font-semibold uppercase tracking-wide text-slate-500">
            Detail
          </h2>
          <p className="mt-2 text-sm text-slate-400">
            {selectedRecordId
              ? `Selected record: ${selectedRecordId}`
              : "Record detail view coming soon."}
          </p>
        </aside>
      </div>
    </div>
  );
}

export default App;
