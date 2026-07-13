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


def test_collections_empty(tmp_path: Path, make_app, make_client) -> None:
    app = make_app(tmp_path)
    client = make_client(app)

    response = client.get("/api/collections")

    assert response.status_code == 200
    assert response.json() == {"collections": []}


def test_collections_returns_schema(tmp_path: Path, make_app, make_client) -> None:
    import chromadb

    client_lib = chromadb.PersistentClient(path=str(tmp_path))
    collection = client_lib.create_collection("foo", metadata={"a": 1})
    collection.add(ids=["1"], embeddings=[[0.1, 0.2, 0.3]])

    app = make_app(tmp_path)
    client = make_client(app)

    response = client.get("/api/collections")

    assert response.status_code == 200
    body = response.json()
    assert len(body["collections"]) == 1
    info = body["collections"][0]
    assert info["name"] == "foo"
    assert info["count"] == 1
    assert info["metadata"] == {"a": 1}
    assert info["dimension"] == 3
    assert isinstance(info["id"], str) and info["id"]


def test_collections_requires_token(tmp_path: Path, make_app, make_client) -> None:
    app = make_app(tmp_path)
    client = make_client(app, token=None)

    response = client.get("/api/collections")

    assert response.status_code == 401


def test_collections_metadata_none_is_null(tmp_path: Path, make_app, make_client) -> None:
    """A collection created without metadata should surface metadata: null,
    not be omitted or coerced into {}."""
    import chromadb

    client_lib = chromadb.PersistentClient(path=str(tmp_path))
    client_lib.create_collection("no_meta")

    app = make_app(tmp_path)
    client = make_client(app)

    response = client.get("/api/collections")

    assert response.status_code == 200
    info = response.json()["collections"][0]
    assert info["metadata"] is None
    assert info["count"] == 0
    assert info["dimension"] is None


def test_collections_many(tmp_path: Path, make_app, make_client) -> None:
    """Enumerating a larger number of collections (30) returns all of them,
    each with the correct name, and no duplicates/omissions."""
    import chromadb

    client_lib = chromadb.PersistentClient(path=str(tmp_path))
    expected_names = {f"collection_{i:02d}" for i in range(30)}
    for name in expected_names:
        client_lib.create_collection(name)

    app = make_app(tmp_path)
    client = make_client(app)

    response = client.get("/api/collections")

    assert response.status_code == 200
    collections = response.json()["collections"]
    assert len(collections) == 30
    returned_names = {c["name"] for c in collections}
    assert returned_names == expected_names
    # ids must be unique
    ids = [c["id"] for c in collections]
    assert len(ids) == len(set(ids))


def test_collections_symbol_names(tmp_path: Path, make_app, make_client) -> None:
    """Collection names containing the full range of Chroma-legal symbol
    characters ('.', '-', '_') round-trip correctly through the API.

    Note: ChromaDB itself restricts collection names to 3-512 characters
    from [a-zA-Z0-9._-] (validated server-side by chromadb, not chromaw), so
    unicode/emoji names are not a valid input to test here.
    """
    import chromadb

    client_lib = chromadb.PersistentClient(path=str(tmp_path))
    names = ["a.b-c_9", "dotted.name.v2", "dash-separated-name", "under_score_name"]
    for name in names:
        client_lib.create_collection(name)

    app = make_app(tmp_path)
    client = make_client(app)

    response = client.get("/api/collections")

    assert response.status_code == 200
    returned_names = {c["name"] for c in response.json()["collections"]}
    assert returned_names == set(names)


def test_collections_readable_in_read_only_mode(tmp_path: Path, make_app, make_client) -> None:
    """GET /api/collections must succeed in read-only mode (write=False):
    listing is a read operation and must not be blocked by write-mode gating."""
    import chromadb

    client_lib = chromadb.PersistentClient(path=str(tmp_path))
    client_lib.create_collection("readable")

    app = make_app(tmp_path, write=False)
    assert app.state.mode == "read-only"
    client = make_client(app)

    response = client.get("/api/collections")

    assert response.status_code == 200
    assert response.json()["collections"][0]["name"] == "readable"


def test_collections_response_keys_match_spec(tmp_path: Path, make_app, make_client) -> None:
    """Response JSON keys for each collection entry must exactly match
    technical-spec §8.2: id, name, count, metadata, dimension (no more, no
    less; e.g. no stray tenant/database fields yet, per M1-1 scope)."""
    import chromadb

    client_lib = chromadb.PersistentClient(path=str(tmp_path))
    collection = client_lib.create_collection("spec_check", metadata={"k": "v"})
    collection.add(ids=["1"], embeddings=[[0.1, 0.2]])

    app = make_app(tmp_path)
    client = make_client(app)

    response = client.get("/api/collections")

    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"collections"}
    info = body["collections"][0]
    assert set(info.keys()) == {"id", "name", "count", "metadata", "dimension"}
