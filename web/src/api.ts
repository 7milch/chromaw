/**
 * Reads the chromaw bearer token from the `<meta name="chromaw-token">` tag
 * that the server injects into index.html (technical-spec §10.2) and wraps
 * `fetch` to attach it as an `Authorization: Bearer <token>` header on every
 * request.
 */
function getToken(): string | null {
  const meta = document.querySelector('meta[name="chromaw-token"]');
  return meta?.getAttribute("content") ?? null;
}

export function apiFetch(input: string, init: RequestInit = {}): Promise<Response> {
  const token = getToken();
  const headers = new Headers(init.headers);
  if (token) {
    headers.set("Authorization", `Bearer ${token}`);
  }
  // `init.signal` (e.g. from an AbortController) is part of RequestInit and
  // passes straight through to fetch, so callers can already cancel
  // in-flight requests by supplying one.
  return fetch(input, { ...init, headers });
}

/**
 * Fetch a unified diff from ``POST /api/diff`` (M2-4, technical-spec §8).
 *
 * Returns ``null`` on any failure (network error, non-2xx, malformed body)
 * so callers can fall back to their existing non-diff confirmation UI
 * instead of blocking the edit flow on this being available.
 */
export async function fetchDiff(
  before: string,
  after: string,
  beforeLabel = "before",
  afterLabel = "after"
): Promise<string | null> {
  try {
    const res = await apiFetch("/api/diff", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        before,
        after,
        before_label: beforeLabel,
        after_label: afterLabel,
      }),
    });
    if (!res.ok) return null;
    const payload = await res.json().catch(() => null);
    if (!payload || typeof payload.diff !== "string") return null;
    return payload.diff;
  } catch {
    return null;
  }
}
