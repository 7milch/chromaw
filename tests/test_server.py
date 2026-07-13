from pathlib import Path

import chromaw
from chromaw.server import create_app


def test_create_app_sets_read_only_state(adapter) -> None:
    app = create_app(adapter, write=False, token="test-token", host="127.0.0.1", port=8000)

    assert app.state.adapter is adapter
    assert app.state.mode == "read-only"
    assert app.state.path == adapter.path


def test_create_app_sets_write_state(tmp_path: Path, make_app) -> None:
    app = make_app(tmp_path, write=True)

    assert app.state.mode == "write"


def test_create_app_generates_token_when_not_supplied(tmp_path: Path, make_app) -> None:
    app = make_app(tmp_path, token=None)

    assert isinstance(app.state.token, str)
    assert len(app.state.token) > 0


def test_unknown_api_route_returns_404(tmp_path: Path, make_app, make_client) -> None:
    app = make_app(tmp_path)
    client = make_client(app)

    response = client.get("/api/does-not-exist")

    assert response.status_code == 404


def test_on_startup_called_when_lifespan_starts(tmp_path: Path, make_app, make_client) -> None:
    calls: list[str] = []
    app = make_app(tmp_path, on_startup=lambda: calls.append("started"))

    assert calls == []
    with make_client(app) as client:
        assert calls == ["started"]
        response = client.get("/api/does-not-exist")
        assert response.status_code == 404

    assert calls == ["started"]


def test_on_startup_not_required(tmp_path: Path, make_app, make_client) -> None:
    app = make_app(tmp_path)

    with make_client(app) as client:
        response = client.get("/api/does-not-exist")
        assert response.status_code == 404


def test_health_read_only(tmp_path: Path, make_app, make_client) -> None:
    app = make_app(tmp_path, write=False)
    client = make_client(app)

    response = client.get("/api/health")

    assert response.status_code == 200
    body = response.json()
    assert body == {
        "ok": True,
        "version": chromaw.__version__,
        "mode": "read-only",
        "path": str(app.state.path),
    }
    assert Path(body["path"]).is_absolute()


def test_health_write(tmp_path: Path, make_app, make_client) -> None:
    app = make_app(tmp_path, write=True)
    client = make_client(app)

    response = client.get("/api/health")

    assert response.status_code == 200
    body = response.json()
    assert body == {
        "ok": True,
        "version": chromaw.__version__,
        "mode": "write",
        "path": str(app.state.path),
    }
    assert Path(body["path"]).is_absolute()
