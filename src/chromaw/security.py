from __future__ import annotations

import secrets

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

__all__ = ["generate_token", "SecurityMiddleware"]


def generate_token() -> str:
    """Generate a random URL-safe token for API authentication."""

    return secrets.token_urlsafe(32)


def _unauthorized(detail: str) -> JSONResponse:
    return JSONResponse({"detail": detail}, status_code=401)


def _forbidden(detail: str) -> JSONResponse:
    return JSONResponse({"detail": detail}, status_code=403)


class SecurityMiddleware(BaseHTTPMiddleware):
    """Enforce local-web-attack protections (technical-spec §10.2).

    - ``Host`` / ``Origin`` headers are validated against an allow-list
      derived from the host/port chromaw was bound to (plus the usual
      localhost aliases). Violations -> 403.
    - Requests under ``/api`` must carry ``Authorization: Bearer <token>``
      matching the server's startup token, compared with
      ``secrets.compare_digest``. Missing/incorrect token -> 401.
    - Static assets (everything outside ``/api``) do not require the
      token -- the index page itself hands the token to the browser via a
      ``<meta>`` tag -- but Host/Origin validation still applies to them.
    """

    def __init__(self, app, *, token: str, host: str, port: int) -> None:
        super().__init__(app)
        self._token = token

        # Design intent (technical-spec §10.2): even when the operator opts
        # into the advanced "--host 0.0.0.0" mode to expose chromaw on the
        # LAN, the Host/Origin allow-list intentionally stays scoped to
        # localhost-style names rather than expanding to "any host that can
        # reach this socket". Binding to 0.0.0.0 only widens which network
        # interface the process listens on; it must not also widen which
        # Host headers are trusted, or a DNS-rebinding-style attack from an
        # untrusted network could bypass this middleware entirely. Clients
        # that connect via a LAN IP or hostname will therefore see 403s from
        # this middleware -- that is expected, see the CLI startup warning.
        bind_host = "127.0.0.1" if host == "0.0.0.0" else host
        hostnames = {"localhost", "127.0.0.1", "::1", bind_host}

        allowed_hosts: set[str] = set()
        allowed_origins: set[str] = set()
        for name in hostnames:
            host_part = f"[{name}]" if ":" in name else name
            allowed_hosts.add(f"{host_part}:{port}")
            allowed_origins.add(f"http://{host_part}:{port}")

        self._allowed_hosts = allowed_hosts
        self._allowed_origins = allowed_origins

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        host_header = request.headers.get("host")
        if host_header is not None and host_header not in self._allowed_hosts:
            return _forbidden("invalid host")

        origin_header = request.headers.get("origin")
        if origin_header is not None and origin_header not in self._allowed_origins:
            return _forbidden("invalid origin")

        if request.url.path.startswith("/api"):
            scheme, _, credential = request.headers.get("authorization", "").partition(" ")
            if scheme.lower() != "bearer" or not self._token_matches(credential):
                return _unauthorized("missing or invalid token")

        return await call_next(request)

    def _token_matches(self, credential: str) -> bool:
        """Constant-time compare ``credential`` against the server token.

        ``secrets.compare_digest`` requires both operands be the same type
        (``str``/``str`` or ``bytes``/``bytes``) and *ASCII-only* when given
        ``str`` arguments -- a non-ASCII ``Authorization`` header (e.g. a
        client sending stray unicode instead of a real bearer token) raises
        ``TypeError`` rather than just failing the comparison, which would
        otherwise surface as an unhandled 500. Encoding to UTF-8 bytes first
        keeps the comparison constant-time while accepting any input.
        """
        try:
            return secrets.compare_digest(
                credential.encode("utf-8"), self._token.encode("utf-8")
            )
        except (TypeError, UnicodeEncodeError):
            return False
