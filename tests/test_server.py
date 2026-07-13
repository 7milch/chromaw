from pathlib import Path

import chromadb
from fastapi.testclient import TestClient

import chromaw
from chromaw.chroma_adapter import ChromaAdapter
from chromaw.server import create_app

_TOKEN = "test-token"
_HOST = "127.0.0.1"
_PORT = 8000


def _make_adapter(tmp_path: Path) -> ChromaAdapter:
    chromadb.PersistentClient(path=str(tmp_path))
    return ChromaAdapter.open(tmp_path)


def _client(app) -> TestClient:
    """A TestClient whose Host header matches the app's allow-list and that
    carries the bearer token, so tests can exercise routes without also
    exercising the security middleware (that lives in test_security.py)."""

    client = TestClient(app, base_url=f"http://{_HOST}:{_PORT}")
    client.headers["Authorization"] = f"Bearer {_TOKEN}"
    return client


def test_create_app_sets_read_only_state(tmp_path: Path) -> None:
    adapter = _make_adapter(tmp_path)

    app = create_app(adapter, write=False, token=_TOKEN, host=_HOST, port=_PORT)

    assert app.state.adapter is adapter
    assert app.state.mode == "read-only"
    assert app.state.path == adapter.path


def test_create_app_sets_write_state(tmp_path: Path) -> None:
    adapter = _make_adapter(tmp_path)

    app = create_app(adapter, write=True, token=_TOKEN, host=_HOST, port=_PORT)

    assert app.state.mode == "write"


def test_create_app_generates_token_when_not_supplied(tmp_path: Path) -> None:
    adapter = _make_adapter(tmp_path)

    app = create_app(adapter, write=False, host=_HOST, port=_PORT)

    assert isinstance(app.state.token, str)
    assert len(app.state.token) > 0


def test_unknown_api_route_returns_404(tmp_path: Path) -> None:
    adapter = _make_adapter(tmp_path)
    app = create_app(adapter, write=False, token=_TOKEN, host=_HOST, port=_PORT)
    client = _client(app)

    response = client.get("/api/does-not-exist")

    assert response.status_code == 404


def test_on_startup_called_when_lifespan_starts(tmp_path: Path) -> None:
    adapter = _make_adapter(tmp_path)
    calls: list[str] = []
    app = create_app(
        adapter,
        write=False,
        token=_TOKEN,
        host=_HOST,
        port=_PORT,
        on_startup=lambda: calls.append("started"),
    )

    assert calls == []
    with _client(app) as client:
        assert calls == ["started"]
        response = client.get("/api/does-not-exist")
        assert response.status_code == 404

    assert calls == ["started"]


def test_on_startup_not_required(tmp_path: Path) -> None:
    adapter = _make_adapter(tmp_path)
    app = create_app(adapter, write=False, token=_TOKEN, host=_HOST, port=_PORT)

    with _client(app) as client:
        response = client.get("/api/does-not-exist")
        assert response.status_code == 404


def test_health_read_only(tmp_path: Path) -> None:
    adapter = _make_adapter(tmp_path)
    app = create_app(adapter, write=False, token=_TOKEN, host=_HOST, port=_PORT)
    client = _client(app)

    response = client.get("/api/health")

    assert response.status_code == 200
    body = response.json()
    assert body == {
        "ok": True,
        "version": chromaw.__version__,
        "mode": "read-only",
        "path": str(adapter.path),
    }
    assert Path(body["path"]).is_absolute()


def test_health_write(tmp_path: Path) -> None:
    adapter = _make_adapter(tmp_path)
    app = create_app(adapter, write=True, token=_TOKEN, host=_HOST, port=_PORT)
    client = _client(app)

    response = client.get("/api/health")

    assert response.status_code == 200
    body = response.json()
    assert body == {
        "ok": True,
        "version": chromaw.__version__,
        "mode": "write",
        "path": str(adapter.path),
    }
    assert Path(body["path"]).is_absolute()
