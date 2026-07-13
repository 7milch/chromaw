import { useEffect, useState } from "react";

interface HealthResponse {
  ok: boolean;
  version: string;
  mode: string;
  path: string;
}

function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch("/api/health")
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
  }, []);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex items-center justify-center p-8">
      <div className="max-w-lg w-full space-y-4">
        <h1 className="text-2xl font-semibold">chromaw</h1>
        {error && (
          <p className="text-red-400">Failed to reach API: {error}</p>
        )}
        {!error && !health && <p className="text-slate-400">Loading...</p>}
        {health && (
          <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-sm">
            <dt className="text-slate-400">path</dt>
            <dd>{health.path}</dd>
            <dt className="text-slate-400">mode</dt>
            <dd>{health.mode}</dd>
            <dt className="text-slate-400">version</dt>
            <dd>{health.version}</dd>
          </dl>
        )}
      </div>
    </div>
  );
}

export default App;
