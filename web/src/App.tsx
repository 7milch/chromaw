import { useEffect, useMemo, useRef, useState } from "react";
import { apiFetch } from "./api";
import { useKeyboardShortcuts } from "./useKeyboardShortcuts";
import ShortcutsHelpModal from "./ShortcutsHelpModal";
import type {
  CollectionInfo,
  CollectionsResponse,
  HealthResponse,
  RecordInfo,
  RecordsGetRequest,
  RecordsResponse,
} from "./types";

type SearchMode = "id" | "metadata" | "document";

interface ActiveSearch {
  ids?: string[];
  where?: Record<string, unknown>;
  where_document?: Record<string, unknown>;
}

/**
 * Parse the search bar's raw input for the given mode into the
 * ids/where/where_document fields sent to POST .../records/get
 * (technical-spec §5.5 1-3). Throws with a user-facing message on invalid
 * input.
 */
function parseSearchInput(mode: SearchMode, rawInput: string): ActiveSearch {
  const input = rawInput.trim();
  if (!input) {
    throw new Error("Enter a search value.");
  }

  if (mode === "id") {
    const ids = input
      .split(",")
      .map((s) => s.trim())
      .filter((s) => s.length > 0);
    if (ids.length === 0) {
      throw new Error("Enter at least one id.");
    }
    return { ids };
  }

  if (mode === "metadata") {
    if (input.startsWith("{")) {
      let where: unknown;
      try {
        where = JSON.parse(input);
      } catch (err) {
        throw new Error(
          `Invalid JSON: ${err instanceof Error ? err.message : String(err)}`
        );
      }
      if (typeof where !== "object" || where === null || Array.isArray(where)) {
        throw new Error("where JSON must be an object.");
      }
      return { where: where as Record<string, unknown> };
    }

    const eqIndex = input.indexOf("=");
    if (eqIndex === -1) {
      throw new Error('Use "key=value" or a raw JSON object starting with "{".');
    }
    const key = input.slice(0, eqIndex).trim();
    const value = input.slice(eqIndex + 1).trim();
    if (!key) {
      throw new Error("Missing metadata key before \"=\".");
    }
    return { where: { [key]: value } };
  }

  // document mode
  return { where_document: { $contains: input } };
}

function formatMetadataJson(metadata: Record<string, unknown> | null): string {
  return metadata ? JSON.stringify(metadata, null, 2) : "null";
}

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
  const [hasMore, setHasMore] = useState(false);
  const [offset, setOffset] = useState(0);
  const [recordsError, setRecordsError] = useState<string | null>(null);
  const [selectedRecordId, setSelectedRecordId] = useState<string | null>(null);

  const [searchMode, setSearchMode] = useState<SearchMode>("id");
  const [searchText, setSearchText] = useState("");
  const [activeSearch, setActiveSearch] = useState<ActiveSearch | null>(null);
  const [searchError, setSearchError] = useState<string | null>(null);

  const [detailRecord, setDetailRecord] = useState<RecordInfo | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailError, setDetailError] = useState<string | null>(null);
  const [detailMissing, setDetailMissing] = useState(false);

  const [helpOpen, setHelpOpen] = useState(false);
  const searchInputRef = useRef<HTMLInputElement | null>(null);

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

  // Reset paging/selection/search whenever the selected collection changes.
  useEffect(() => {
    setOffset(0);
    setRecordsTotal(0);
    setHasMore(false);
    setSelectedRecordId(null);
    setRecords(null);
    setRecordsError(null);
    setActiveSearch(null);
    setSearchText("");
    setSearchError(null);
  }, [selectedName]);

  useEffect(() => {
    if (!selectedName) return;

    // Guard against out-of-order responses: if selectedName/offset/search
    // change again before this request resolves, ignore its result instead
    // of clobbering state set by the newer request.
    let ignore = false;

    if (activeSearch) {
      // Search mode: POST .../records/get with ids/where/where_document,
      // keeping limit/offset so paging still works against the filtered
      // set (task §5.5, spec §8.3).
      const body: RecordsGetRequest = {
        ...activeSearch,
        limit: PAGE_LIMIT,
        offset,
        include: ["documents", "metadatas", "uris"],
      };

      apiFetch(`/api/collections/${encodeURIComponent(selectedName)}/records/get`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      })
        .then(async (res) => {
          if (!res.ok) {
            let detail = `request failed: ${res.status}`;
            if (res.status === 422) {
              const payload = await res.json().catch(() => null);
              if (payload && typeof payload.detail === "string") {
                detail = payload.detail;
              }
            }
            throw new Error(detail);
          }
          return res.json() as Promise<RecordsResponse>;
        })
        .then((data) => {
          if (ignore) return;
          setRecords(data.records);
          setRecordsTotal(data.total);
          setHasMore(data.has_more);
          setSearchError(null);
        })
        .catch((err: unknown) => {
          if (ignore) return;
          setSearchError(err instanceof Error ? err.message : String(err));
        });

      return () => {
        ignore = true;
      };
    }

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
        setHasMore(data.has_more);
      })
      .catch((err: unknown) => {
        if (ignore) return;
        setRecordsError(err instanceof Error ? err.message : String(err));
      });

    return () => {
      ignore = true;
    };
  }, [selectedName, offset, activeSearch]);

  // Fetch full detail (including embeddings) for the selected record.
  // Same out-of-order-response guard as the records list fetch above.
  useEffect(() => {
    setDetailRecord(null);
    setDetailError(null);
    setDetailMissing(false);

    if (!selectedName || !selectedRecordId) return;

    let ignore = false;
    setDetailLoading(true);

    apiFetch(`/api/collections/${encodeURIComponent(selectedName)}/records/get`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        ids: [selectedRecordId],
        include: ["documents", "metadatas", "uris", "embeddings"],
      }),
    })
      .then((res) => {
        if (!res.ok) {
          throw new Error(`request failed: ${res.status}`);
        }
        return res.json() as Promise<RecordsResponse>;
      })
      .then((data) => {
        if (ignore) return;
        const record = data.records[0] ?? null;
        setDetailRecord(record);
        setDetailMissing(record === null);
      })
      .catch((err: unknown) => {
        if (ignore) return;
        setDetailError(err instanceof Error ? err.message : String(err));
      })
      .finally(() => {
        if (ignore) return;
        setDetailLoading(false);
      });

    return () => {
      ignore = true;
    };
  }, [selectedName, selectedRecordId]);

  const selected = collections?.find((c) => c.name === selectedName) ?? null;
  const recordIds = useMemo(() => records?.map((r) => r.id) ?? [], [records]);

  useKeyboardShortcuts({
    searchInputRef,
    recordIds,
    selectedRecordId,
    onSelectRecordId: setSelectedRecordId,
    helpOpen,
    onSetHelpOpen: setHelpOpen,
  });

  function executeSearch() {
    try {
      const parsed = parseSearchInput(searchMode, searchText);
      setSearchError(null);
      setOffset(0);
      setActiveSearch(parsed);
    } catch (err) {
      setSearchError(err instanceof Error ? err.message : String(err));
    }
  }

  function clearSearch() {
    setSearchText("");
    setSearchError(null);
    setActiveSearch(null);
    setOffset(0);
  }

  // ids-based search returns every match in one page (chromaw ignores
  // limit/offset for it server-side), so paging controls are meaningless
  // and disabled in that mode.
  const isIdSearch = activeSearch?.ids !== undefined;
  const rangeStart = recordsTotal === 0 ? 0 : offset + 1;
  const rangeEnd = records ? offset + records.length : 0;
  const canPrev = !isIdSearch && offset > 0;
  const canNext = !isIdSearch && hasMore;

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

              <div className="flex flex-col gap-1">
                <div className="flex items-center gap-2">
                  <select
                    value={searchMode}
                    onChange={(e) => setSearchMode(e.target.value as SearchMode)}
                    className="rounded border border-slate-700 bg-slate-900 px-2 py-1 text-sm text-slate-200"
                  >
                    <option value="id">ID</option>
                    <option value="metadata">Metadata</option>
                    <option value="document">Document</option>
                  </select>
                  <input
                    ref={searchInputRef}
                    type="text"
                    value={searchText}
                    onChange={(e) => setSearchText(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") executeSearch();
                    }}
                    placeholder={
                      searchMode === "id"
                        ? "id1, id2, ..."
                        : searchMode === "metadata"
                          ? 'key=value or {"key": "value"}'
                          : "text the document should contain"
                    }
                    className="min-w-0 flex-1 rounded border border-slate-700 bg-slate-900 px-2 py-1 text-sm text-slate-200 placeholder:text-slate-600"
                  />
                  <button
                    type="button"
                    onClick={executeSearch}
                    className="rounded border border-slate-700 px-2 py-1 text-sm hover:bg-slate-800"
                  >
                    Search
                  </button>
                  {activeSearch && (
                    <button
                      type="button"
                      onClick={clearSearch}
                      className="rounded border border-slate-700 px-2 py-1 text-sm hover:bg-slate-800"
                    >
                      Clear
                    </button>
                  )}
                </div>
                {searchError && <p className="text-sm text-red-400">{searchError}</p>}
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
                      {isIdSearch ? (
                        <>
                          {records?.length ?? 0} match{records?.length === 1 ? "" : "es"}
                        </>
                      ) : activeSearch ? (
                        <>
                          {rangeStart}–{rangeEnd} (total unknown while filtering)
                        </>
                      ) : (
                        <>
                          {rangeStart}–{rangeEnd} / {recordsTotal}
                        </>
                      )}
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

        <aside className="w-96 shrink-0 border-l border-slate-800 overflow-y-auto p-3">
          <h2 className="text-xs font-semibold uppercase tracking-wide text-slate-500">
            Detail
          </h2>

          {!selectedRecordId && (
            <p className="mt-2 text-sm text-slate-400">
              Select a record from the table to see its details.
            </p>
          )}

          {selectedRecordId && detailLoading && (
            <p className="mt-2 text-sm text-slate-400">Loading record...</p>
          )}

          {selectedRecordId && !detailLoading && detailError && (
            <p className="mt-2 text-sm text-red-400">
              Failed to load record: {detailError}
            </p>
          )}

          {selectedRecordId && !detailLoading && !detailError && detailMissing && (
            <p className="mt-2 text-sm text-slate-400">
              This record is no longer present in the collection.
            </p>
          )}

          {selectedRecordId && !detailLoading && !detailError && detailRecord && (
            <div className="mt-2 space-y-4 text-sm">
              <div>
                <h3 className="mb-1 text-xs font-semibold uppercase tracking-wide text-slate-500">
                  id
                </h3>
                <code className="block select-all break-all rounded bg-slate-900 px-2 py-1 text-xs text-slate-300">
                  {detailRecord.id}
                </code>
              </div>

              <div>
                <h3 className="mb-1 text-xs font-semibold uppercase tracking-wide text-slate-500">
                  document
                </h3>
                <pre className="max-h-64 overflow-auto whitespace-pre-wrap break-words rounded bg-slate-900 p-2 font-mono text-xs text-slate-300">
                  {detailRecord.document ?? "-"}
                </pre>
              </div>

              <div>
                <h3 className="mb-1 text-xs font-semibold uppercase tracking-wide text-slate-500">
                  metadata
                </h3>
                <pre className="max-h-64 overflow-auto rounded bg-slate-900 p-2 font-mono text-xs text-slate-300">
                  {formatMetadataJson(detailRecord.metadata)}
                </pre>
              </div>

              <div>
                <h3 className="mb-1 text-xs font-semibold uppercase tracking-wide text-slate-500">
                  uri
                </h3>
                <p className="break-all text-xs text-slate-300">
                  {detailRecord.uri ?? "-"}
                </p>
              </div>

              <div>
                <h3 className="mb-1 text-xs font-semibold uppercase tracking-wide text-slate-500">
                  embedding
                </h3>
                {detailRecord.embedding_dimension === null ? (
                  <p className="text-xs text-slate-400">No embedding available.</p>
                ) : (
                  <div className="space-y-1">
                    <p className="text-xs text-slate-400">
                      dimension: {detailRecord.embedding_dimension}
                    </p>
                    <pre className="overflow-x-auto rounded bg-slate-900 p-2 font-mono text-xs text-slate-300">
                      [{(detailRecord.embedding_preview ?? []).join(", ")}
                      {(detailRecord.embedding_preview?.length ?? 0) <
                      detailRecord.embedding_dimension
                        ? ", ..."
                        : ""}
                      ]
                    </pre>
                  </div>
                )}
              </div>
            </div>
          )}
        </aside>
      </div>

      {helpOpen && <ShortcutsHelpModal onClose={() => setHelpOpen(false)} />}
    </div>
  );
}

export default App;
