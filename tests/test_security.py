from __future__ import annotations

from pathlib import Path

import chromadb
from fastapi.testclient import TestClient

from chromaw.chroma_adapter import ChromaAdapter
from chromaw.security import generate_token
from chromaw.server import create_app

_TOKEN = "test-token"
_HOST = "127.0.0.1"
_PORT = 8000


def _make_adapter(tmp_path: Path) -> ChromaAdapter:
    chromadb.PersistentClient(path=str(tmp_path))
    return ChromaAdapter.open(tmp_path)


def _app(tmp_path: Path):
    adapter = _make_adapter(tmp_path)
    return create_app(adapter, write=False, token=_TOKEN, host=_HOST, port=_PORT)


def _client(app) -> TestClient:
    return TestClient(app, base_url=f"http://{_HOST}:{_PORT}")


def test_generate_token_returns_distinct_urlsafe_strings() -> None:
    a = generate_token()
    b = generate_token()
    assert isinstance(a, str)
    assert len(a) > 20
    assert a != b


def test_api_without_token_returns_401(tmp_path: Path) -> None:
    client = _client(_app(tmp_path))

    response = client.get("/api/health")

    assert response.status_code == 401


def test_api_with_wrong_token_returns_401(tmp_path: Path) -> None:
    client = _client(_app(tmp_path))
    client.headers["Authorization"] = "Bearer wrong-token"

    response = client.get("/api/health")

    assert response.status_code == 401


def test_api_with_malformed_scheme_returns_401(tmp_path: Path) -> None:
    client = _client(_app(tmp_path))
    client.headers["Authorization"] = f"Basic {_TOKEN}"

    response = client.get("/api/health")

    assert response.status_code == 401


def test_api_with_correct_token_returns_200(tmp_path: Path) -> None:
    client = _client(_app(tmp_path))
    client.headers["Authorization"] = f"Bearer {_TOKEN}"

    response = client.get("/api/health")

    assert response.status_code == 200


def test_static_route_does_not_require_token(tmp_path: Path) -> None:
    # No Authorization header is sent; static routes must not 401 even
    # though they are unauthenticated (the HTML itself hands out the token).
    client = _client(_app(tmp_path))

    response = client.get("/")

    assert response.status_code != 401


def test_invalid_origin_returns_403(tmp_path: Path) -> None:
    client = _client(_app(tmp_path))
    client.headers["Authorization"] = f"Bearer {_TOKEN}"
    client.headers["Origin"] = "http://evil.example.com"

    response = client.get("/api/health")

    assert response.status_code == 403


def test_valid_origin_passes(tmp_path: Path) -> None:
    client = _client(_app(tmp_path))
    client.headers["Authorization"] = f"Bearer {_TOKEN}"
    client.headers["Origin"] = f"http://{_HOST}:{_PORT}"

    response = client.get("/api/health")

    assert response.status_code == 200


def test_localhost_origin_passes(tmp_path: Path) -> None:
    client = _client(_app(tmp_path))
    client.headers["Authorization"] = f"Bearer {_TOKEN}"
    client.headers["Origin"] = f"http://localhost:{_PORT}"

    response = client.get("/api/health")

    assert response.status_code == 200


def test_missing_origin_passes(tmp_path: Path) -> None:
    client = _client(_app(tmp_path))
    client.headers["Authorization"] = f"Bearer {_TOKEN}"

    response = client.get("/api/health")

    assert response.status_code == 200


def test_invalid_host_returns_403(tmp_path: Path) -> None:
    adapter = _make_adapter(tmp_path)
    app = create_app(adapter, write=False, token=_TOKEN, host=_HOST, port=_PORT)
    client = TestClient(app, base_url="http://evil.example.com")
    client.headers["Authorization"] = f"Bearer {_TOKEN}"

    response = client.get("/api/health")

    assert response.status_code == 403


def test_bind_host_0_0_0_0_still_allows_127_0_0_1_host_header(tmp_path: Path) -> None:
    adapter = _make_adapter(tmp_path)
    app = create_app(adapter, write=False, token=_TOKEN, host="0.0.0.0", port=_PORT)
    client = TestClient(app, base_url=f"http://127.0.0.1:{_PORT}")
    client.headers["Authorization"] = f"Bearer {_TOKEN}"

    response = client.get("/api/health")

    assert response.status_code == 200


def test_api_with_empty_authorization_header_returns_401(tmp_path: Path) -> None:
    client = _client(_app(tmp_path))
    client.headers["Authorization"] = ""

    response = client.get("/api/health")

    assert response.status_code == 401


def test_api_with_bearer_scheme_only_no_token_returns_401(tmp_path: Path) -> None:
    client = _client(_app(tmp_path))
    client.headers["Authorization"] = "Bearer"

    response = client.get("/api/health")

    assert response.status_code == 401


def test_api_with_lowercase_bearer_scheme_passes(tmp_path: Path) -> None:
    client = _client(_app(tmp_path))
    client.headers["Authorization"] = f"bearer {_TOKEN}"

    response = client.get("/api/health")

    assert response.status_code == 200


def test_api_with_uppercase_bearer_scheme_passes(tmp_path: Path) -> None:
    client = _client(_app(tmp_path))
    client.headers["Authorization"] = f"BEARER {_TOKEN}"

    response = client.get("/api/health")

    assert response.status_code == 200


def test_null_origin_returns_403(tmp_path: Path) -> None:
    client = _client(_app(tmp_path))
    client.headers["Authorization"] = f"Bearer {_TOKEN}"
    client.headers["Origin"] = "null"

    response = client.get("/api/health")

    assert response.status_code == 403


def test_https_scheme_origin_returns_403(tmp_path: Path) -> None:
    client = _client(_app(tmp_path))
    client.headers["Authorization"] = f"Bearer {_TOKEN}"
    client.headers["Origin"] = f"https://{_HOST}:{_PORT}"

    response = client.get("/api/health")

    assert response.status_code == 403


def test_origin_with_wrong_port_returns_403(tmp_path: Path) -> None:
    client = _client(_app(tmp_path))
    client.headers["Authorization"] = f"Bearer {_TOKEN}"
    client.headers["Origin"] = f"http://{_HOST}:9999"

    response = client.get("/api/health")

    assert response.status_code == 403


def test_host_header_with_wrong_port_returns_403(tmp_path: Path) -> None:
    adapter = _make_adapter(tmp_path)
    app = create_app(adapter, write=False, token=_TOKEN, host=_HOST, port=_PORT)
    client = TestClient(app, base_url=f"http://{_HOST}:9999")
    client.headers["Authorization"] = f"Bearer {_TOKEN}"

    response = client.get("/api/health")

    assert response.status_code == 403


def test_static_asset_path_does_not_require_token_but_origin_still_checked(
    tmp_path: Path,
) -> None:
    client = _client(_app(tmp_path))
    client.headers["Origin"] = "http://evil.example.com"

    response = client.get("/")

    assert response.status_code == 403


def test_api_health_options_request_has_no_cors_headers(tmp_path: Path) -> None:
    client = _client(_app(tmp_path))
    client.headers["Authorization"] = f"Bearer {_TOKEN}"
    client.headers["Origin"] = f"http://{_HOST}:{_PORT}"
    client.headers["Access-Control-Request-Method"] = "GET"

    response = client.options("/api/health")

    assert "access-control-allow-origin" not in {
        k.lower() for k in response.headers.keys()
    }


def test_get_request_has_no_cors_headers(tmp_path: Path) -> None:
    client = _client(_app(tmp_path))
    client.headers["Authorization"] = f"Bearer {_TOKEN}"
    client.headers["Origin"] = f"http://{_HOST}:{_PORT}"

    response = client.get("/api/health")

    assert "access-control-allow-origin" not in {
        k.lower() for k in response.headers.keys()
    }


def test_index_html_contains_token_meta_tag(tmp_path: Path) -> None:
    client = _client(_app(tmp_path))

    response = client.get("/")

    assert f'<meta name="chromaw-token" content="{_TOKEN}">' in response.text


def test_compare_digest_used_for_token_comparison() -> None:
    import inspect

    from chromaw import security

    source = inspect.getsource(security.SecurityMiddleware._token_matches)
    assert "secrets.compare_digest" in source


def test_api_with_non_ascii_authorization_header_returns_401(tmp_path: Path) -> None:
    # Regression: secrets.compare_digest raises TypeError for non-ASCII str
    # arguments, which previously surfaced as an unhandled 500 instead of a
    # 401. The credential must be rejected cleanly instead.
    #
    # httpx's high-level ``headers`` mapping only accepts ASCII values, but
    # real-world clients can still send non-ASCII bytes on the wire (HTTP
    # header values are ISO-8859-1 per RFC 7230), so the raw
    # ``list[tuple[bytes, bytes]]`` header form is used here to reproduce
    # that directly.
    client = _client(_app(tmp_path))

    response = client.get(
        "/api/health",
        headers=[(b"Authorization", "Bearer tökén-ünïcödé".encode("utf-8"))],
    )

    assert response.status_code == 401
