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
