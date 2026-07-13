from __future__ import annotations

from pathlib import Path

import chromadb
import pytest
from fastapi.testclient import TestClient

from chromaw import server as server_module
from chromaw.chroma_adapter import ChromaAdapter
from chromaw.server import create_app


def _make_adapter(tmp_path: Path) -> ChromaAdapter:
    chromadb.PersistentClient(path=str(tmp_path))
    return ChromaAdapter.open(tmp_path)


class _FakePackage:
    def __init__(self, static_dir: Path) -> None:
        self._static_dir = static_dir

    def joinpath(self, name: str) -> Path:
        assert name == "static"
        return self._static_dir


def test_no_static_dir_root_404_but_api_ok(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = _make_adapter(tmp_path / "chroma")

    missing_static_dir = tmp_path / "static"  # never created
    monkeypatch.setattr(
        server_module, "files", lambda _pkg: _FakePackage(missing_static_dir)
    )

    app = create_app(adapter, write=False)
    client = TestClient(app)

    root_response = client.get("/")
    health_response = client.get("/api/health")

    assert root_response.status_code == 404
    assert health_response.status_code == 200


def test_static_dir_present_serves_index_and_api(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = _make_adapter(tmp_path / "chroma")

    static_dir = tmp_path / "static"
    static_dir.mkdir()
    (static_dir / "index.html").write_text("<html><body>chromaw</body></html>")

    monkeypatch.setattr(server_module, "files", lambda _pkg: _FakePackage(static_dir))

    app = create_app(adapter, write=False)
    client = TestClient(app)

    root_response = client.get("/")
    health_response = client.get("/api/health")

    assert root_response.status_code == 200
    assert "chromaw" in root_response.text
    assert health_response.status_code == 200


def test_static_dir_present_without_index_html_root_404_but_api_ok(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A static dir that exists but has no index.html must not be mounted;
    otherwise StaticFiles(html=True) would 404 on GET / anyway with no
    listing, but we want to be sure we don't accidentally mount a directory
    that can't serve an SPA shell, and that /api keeps working regardless."""
    adapter = _make_adapter(tmp_path / "chroma")

    static_dir = tmp_path / "static"
    static_dir.mkdir()
    (static_dir / "assets").mkdir()
    (static_dir / "assets" / "index.js").write_text("console.log('x')")
    # deliberately no index.html

    monkeypatch.setattr(server_module, "files", lambda _pkg: _FakePackage(static_dir))

    app = create_app(adapter, write=False)
    client = TestClient(app)

    root_response = client.get("/")
    health_response = client.get("/api/health")

    assert root_response.status_code == 404
    assert health_response.status_code == 200


def test_spa_deep_path_currently_404s_not_shadowed_as_html(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """StaticFiles(html=True) only falls back to index.html for '/' (and
    directory paths); an unknown deep path like a client-side route on hard
    refresh currently 404s rather than serving the SPA shell. M0-5 is
    scoped to a placeholder page with no client-side routing yet, so this
    is documented current behavior, not a bug -- but it will need a catch-all
    fallback route once client-side routing (M1+) is introduced."""
    adapter = _make_adapter(tmp_path / "chroma")

    static_dir = tmp_path / "static"
    static_dir.mkdir()
    (static_dir / "index.html").write_text("<html><body>chromaw-spa</body></html>")

    monkeypatch.setattr(server_module, "files", lambda _pkg: _FakePackage(static_dir))

    app = create_app(adapter, write=False)
    client = TestClient(app)

    deep_response = client.get("/collections/some-collection/records/123")
    root_response = client.get("/")

    assert root_response.status_code == 200
    assert "chromaw-spa" in root_response.text
    assert deep_response.status_code == 404


def test_api_route_not_shadowed_by_static_mount(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """/api is registered before the static mount, so even an unknown /api/*
    path must return a JSON 404 from FastAPI's router, not fall through to
    the SPA's index.html (which would silently mask API errors as HTML)."""
    adapter = _make_adapter(tmp_path / "chroma")

    static_dir = tmp_path / "static"
    static_dir.mkdir()
    (static_dir / "index.html").write_text("<html><body>chromaw-spa</body></html>")

    monkeypatch.setattr(server_module, "files", lambda _pkg: _FakePackage(static_dir))

    app = create_app(adapter, write=False)
    client = TestClient(app)

    response = client.get("/api/does-not-exist")

    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/json")
    assert "chromaw-spa" not in response.text
