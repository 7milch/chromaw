import { useEffect, useState } from "react";
import { apiFetch } from "./api";
import type { CollectionInfo, CollectionsResponse, HealthResponse } from "./types";

function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [collections, setCollections] = useState<CollectionInfo[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedName, setSelectedName] = useState<string | null>(null);

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

  const selected = collections?.find((c) => c.name === selectedName) ?? null;

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

        <main className="flex-1 min-w-0 overflow-y-auto p-4">
          {!selected && (
            <p className="text-sm text-slate-400">
              Select a collection from the left to see its details.
            </p>
          )}
          {selected && (
            <div className="space-y-4">
              <h2 className="text-lg font-semibold">{selected.name}</h2>
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
        </main>

        <aside className="w-72 shrink-0 border-l border-slate-800 overflow-y-auto p-3">
          <h2 className="text-xs font-semibold uppercase tracking-wide text-slate-500">
            Detail
          </h2>
          <p className="mt-2 text-sm text-slate-400">Record detail view coming soon.</p>
        </aside>
      </div>
    </div>
  );
}

export default App;
