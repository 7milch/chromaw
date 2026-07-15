import { createContext, useContext, useEffect, useState, type ReactNode } from "react";
import { apiFetch } from "./api";
import type { HealthResponse } from "./types";

/**
 * Shares server-derived app configuration (mode, path, version - from
 * GET /api/health) across the component tree so descendants can read
 * ``mode`` without it being threaded through props from App.tsx
 * (technical-spec §3.2: read-only vs write mode drives which edit UI is
 * shown, so most of the tree eventually needs this).
 */
export interface AppConfig {
  /** Raw health response, or null while the initial fetch is in flight. */
  health: HealthResponse | null;
  /** Error message if the health fetch failed. */
  error: string | null;
  /** Convenience flag: true once health has resolved to write mode. */
  isWriteMode: boolean;
  /**
   * Convenience flag (M3-3): true once health has resolved and an explicit
   * ``--embedding-config`` is available server-side, driving whether
   * DocumentEditor offers a "Re-embed" option. This is a conservative
   * signal -- see ``HealthResponse.embedding_available``'s docstring
   * (server-side) for why a collection can still support re-embedding via
   * its own embedding function even when this is false.
   */
  embeddingAvailable: boolean;
}

const AppConfigContext = createContext<AppConfig | undefined>(undefined);

export function AppConfigProvider({ children }: { children: ReactNode }) {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

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
  }, []);

  const value: AppConfig = {
    health,
    error,
    isWriteMode: health?.mode === "write",
    embeddingAvailable: health?.embedding_available === true,
  };

  return <AppConfigContext.Provider value={value}>{children}</AppConfigContext.Provider>;
}

export function useAppConfig(): AppConfig {
  const ctx = useContext(AppConfigContext);
  if (ctx === undefined) {
    throw new Error("useAppConfig must be used within an AppConfigProvider");
  }
  return ctx;
}
