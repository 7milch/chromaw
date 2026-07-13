"""Tests for the ``require_write_mode`` dependency (technical-spec §3.2,
roadmap M2-1).

M2-2+ write endpoints will declare
``dependencies=[Depends(require_write_mode)]``; since no real write
endpoint exists yet, these tests attach a dummy route carrying the
dependency to a freshly built app (via the ``make_app`` fixture) to
exercise the guard in isolation.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import Depends

from chromaw.api import require_write_mode


def _add_dummy_write_route(app) -> None:
    @app.post("/api/_dummy-write", dependencies=[Depends(require_write_mode)])
    def _dummy_write() -> dict:
        return {"ok": True}

    # ``create_app`` mounts a catch-all StaticFiles handler at "/" last, so a
    # route added afterwards (as here) would otherwise be shadowed by it and
    # match nothing -> 405. Move the newly added route to the front so it is
    # tried before the static mount.
    app.router.routes.insert(0, app.router.routes.pop())


def test_write_route_rejected_in_read_only_mode(tmp_path: Path, make_app, make_client) -> None:
    app = make_app(tmp_path, write=False)
    _add_dummy_write_route(app)
    client = make_client(app)

    response = client.post("/api/_dummy-write")

    assert response.status_code == 403
    assert "--write" in response.json()["detail"]


def test_write_route_allowed_in_write_mode(tmp_path: Path, make_app, make_client) -> None:
    app = make_app(tmp_path, write=True)
    _add_dummy_write_route(app)
    client = make_client(app)

    response = client.post("/api/_dummy-write")

    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_write_route_without_token_returns_401_not_403(
    tmp_path: Path, make_app, make_client
) -> None:
    """Auth (SecurityMiddleware) runs before routing/dependencies, so a
    request with no bearer token against a read-only server must fail with
    401 (unauthenticated), not 403 (read-only) -- the client shouldn't be
    able to distinguish "wrong token" from "read-only mode" without first
    proving it holds a valid token.
    """
    app = make_app(tmp_path, write=False)
    _add_dummy_write_route(app)
    client = make_client(app, token=None)

    response = client.post("/api/_dummy-write")

    assert response.status_code == 401
    assert "token" in response.json()["detail"]


def test_write_route_with_wrong_token_returns_401_not_403(
    tmp_path: Path, make_app, make_client
) -> None:
    """Same ordering guarantee as above, but with an incorrect (rather than
    missing) token."""
    app = make_app(tmp_path, write=False)
    _add_dummy_write_route(app)
    client = make_client(app, token="wrong-token")

    response = client.post("/api/_dummy-write")

    assert response.status_code == 401


def test_write_route_with_valid_token_in_write_mode_still_requires_auth(
    tmp_path: Path, make_app, make_client
) -> None:
    """Sanity check that a correct token plus write mode is the only
    combination that reaches the handler -- i.e. the guard doesn't
    accidentally bypass auth."""
    app = make_app(tmp_path, write=True)
    _add_dummy_write_route(app)
    client = make_client(app, token="wrong-token")

    response = client.post("/api/_dummy-write")

    assert response.status_code == 401


def test_read_only_route_without_token_returns_401_not_bypassed(
    tmp_path: Path, make_app, make_client
) -> None:
    """A non-write endpoint (no require_write_mode dependency) still must
    not be reachable without a valid token -- confirms the write guard is
    additive to auth, not a replacement for it."""
    app = make_app(tmp_path, write=False)
    client = make_client(app, token=None)

    response = client.get("/api/health")

    assert response.status_code == 401
