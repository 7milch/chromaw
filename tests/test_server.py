import json
from pathlib import Path

import pytest

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
        "embedding_available": False,
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
        "embedding_available": False,
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


def _add_records(tmp_path: Path, name: str, count: int, *, with_embeddings: bool = False):
    import chromadb

    client_lib = chromadb.PersistentClient(path=str(tmp_path))
    collection = client_lib.create_collection(name)
    ids = [str(i) for i in range(count)]
    documents = [f"doc-{i}" for i in range(count)]
    metadatas = [{"idx": i} for i in range(count)]
    kwargs = {"ids": ids, "documents": documents, "metadatas": metadatas}
    if with_embeddings:
        kwargs["embeddings"] = [[float(i), float(i) + 1] for i in range(count)]
    collection.add(**kwargs)
    return collection


def test_records_returns_schema(tmp_path: Path, make_app, make_client) -> None:
    _add_records(tmp_path, "foo", 3)

    app = make_app(tmp_path)
    client = make_client(app)

    response = client.get("/api/collections/foo/records")

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 3
    assert len(body["records"]) == 3
    record = body["records"][0]
    assert set(record.keys()) == {
        "id",
        "document",
        "metadata",
        "uri",
        "embedding_dimension",
        "embedding_preview",
    }


def test_records_paging_limit(tmp_path: Path, make_app, make_client) -> None:
    _add_records(tmp_path, "foo", 10)

    app = make_app(tmp_path)
    client = make_client(app)

    response = client.get("/api/collections/foo/records?limit=4&offset=0")

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 10
    assert len(body["records"]) == 4


def test_records_paging_offset_beyond_total_is_empty(tmp_path: Path, make_app, make_client) -> None:
    _add_records(tmp_path, "foo", 3)

    app = make_app(tmp_path)
    client = make_client(app)

    response = client.get("/api/collections/foo/records?limit=50&offset=100")

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 3
    assert body["records"] == []


def test_records_include_embeddings(tmp_path: Path, make_app, make_client) -> None:
    _add_records(tmp_path, "foo", 1, with_embeddings=True)

    app = make_app(tmp_path)
    client = make_client(app)

    response = client.get(
        "/api/collections/foo/records?include=documents,metadatas,uris,embeddings"
    )

    assert response.status_code == 200
    record = response.json()["records"][0]
    assert record["embedding_dimension"] == 2
    assert record["embedding_preview"] == [0.0, 1.0]


def test_records_without_embeddings_include_has_none_dimension(
    tmp_path: Path, make_app, make_client
) -> None:
    _add_records(tmp_path, "foo", 1, with_embeddings=True)

    app = make_app(tmp_path)
    client = make_client(app)

    response = client.get("/api/collections/foo/records")

    assert response.status_code == 200
    record = response.json()["records"][0]
    assert record["embedding_dimension"] is None
    assert record["embedding_preview"] is None


def test_records_nonexistent_collection_returns_404(tmp_path: Path, make_app, make_client) -> None:
    app = make_app(tmp_path)
    client = make_client(app)

    response = client.get("/api/collections/does-not-exist/records")

    assert response.status_code == 404


def test_records_limit_out_of_range_returns_422(tmp_path: Path, make_app, make_client) -> None:
    _add_records(tmp_path, "foo", 1)

    app = make_app(tmp_path)
    client = make_client(app)

    assert client.get("/api/collections/foo/records?limit=0").status_code == 422
    assert client.get("/api/collections/foo/records?limit=501").status_code == 422


def test_records_negative_offset_returns_422(tmp_path: Path, make_app, make_client) -> None:
    _add_records(tmp_path, "foo", 1)

    app = make_app(tmp_path)
    client = make_client(app)

    response = client.get("/api/collections/foo/records?offset=-1")

    assert response.status_code == 422


def test_records_invalid_include_returns_422(tmp_path: Path, make_app, make_client) -> None:
    _add_records(tmp_path, "foo", 1)

    app = make_app(tmp_path)
    client = make_client(app)

    response = client.get("/api/collections/foo/records?include=documents,bogus")

    assert response.status_code == 422


def test_records_requires_token(tmp_path: Path, make_app, make_client) -> None:
    _add_records(tmp_path, "foo", 1)

    app = make_app(tmp_path)
    client = make_client(app, token=None)

    response = client.get("/api/collections/foo/records")

    assert response.status_code == 401


def test_records_offset_equal_to_total_is_empty(tmp_path: Path, make_app, make_client) -> None:
    """offset == total (not beyond) must yield an empty page, not an error."""
    _add_records(tmp_path, "foo", 5)

    app = make_app(tmp_path)
    client = make_client(app)

    response = client.get("/api/collections/foo/records?limit=10&offset=5")

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 5
    assert body["records"] == []


def test_records_limit_boundary_min(tmp_path: Path, make_app, make_client) -> None:
    """limit=1 (the minimum allowed) returns exactly one record."""
    _add_records(tmp_path, "foo", 5)

    app = make_app(tmp_path)
    client = make_client(app)

    response = client.get("/api/collections/foo/records?limit=1&offset=0")

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 5
    assert len(body["records"]) == 1


def test_records_limit_boundary_max(tmp_path: Path, make_app, make_client) -> None:
    """limit=500 (the maximum allowed) succeeds and is capped by total when
    fewer records exist."""
    _add_records(tmp_path, "foo", 10)

    app = make_app(tmp_path)
    client = make_client(app)

    response = client.get("/api/collections/foo/records?limit=500&offset=0")

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 10
    assert len(body["records"]) == 10


def test_records_include_embeddings_only(tmp_path: Path, make_app, make_client) -> None:
    """include=embeddings alone must still return embedding info, and the
    other fields (document/metadata/uri) must be None since they were not
    requested."""
    _add_records(tmp_path, "foo", 1, with_embeddings=True)

    app = make_app(tmp_path)
    client = make_client(app)

    response = client.get("/api/collections/foo/records?include=embeddings")

    assert response.status_code == 200
    record = response.json()["records"][0]
    assert record["embedding_dimension"] == 2
    assert record["embedding_preview"] == [0.0, 1.0]
    assert record["document"] is None
    assert record["metadata"] is None
    assert record["uri"] is None


def test_records_document_metadata_uri_none(tmp_path: Path, make_app, make_client) -> None:
    """A record added with no document/metadata/uri (only id + embedding)
    must surface those fields as null, not omitted or coerced."""
    import chromadb

    client_lib = chromadb.PersistentClient(path=str(tmp_path))
    collection = client_lib.create_collection("bare")
    collection.add(ids=["1"], embeddings=[[0.1, 0.2]])

    app = make_app(tmp_path)
    client = make_client(app)

    response = client.get("/api/collections/bare/records")

    assert response.status_code == 200
    record = response.json()["records"][0]
    assert record["id"] == "1"
    assert record["document"] is None
    assert record["metadata"] is None
    assert record["uri"] is None


def test_records_empty_collection(tmp_path: Path, make_app, make_client) -> None:
    """A collection with zero records returns total=0 and an empty list,
    not an error."""
    import chromadb

    client_lib = chromadb.PersistentClient(path=str(tmp_path))
    client_lib.create_collection("empty")

    app = make_app(tmp_path)
    client = make_client(app)

    response = client.get("/api/collections/empty/records")

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 0
    assert body["records"] == []


def test_records_non_ascii_metadata(tmp_path: Path, make_app, make_client) -> None:
    """Non-ASCII characters (Japanese, emoji) in document/metadata round-trip
    correctly through JSON, without mangling or escaping artifacts."""
    import chromadb

    client_lib = chromadb.PersistentClient(path=str(tmp_path))
    collection = client_lib.create_collection("intl")
    collection.add(
        ids=["1"],
        documents=["こんにちは世界 🌍"],
        metadatas=[{"source": "日本語ファイル.txt", "emoji": "🔥"}],
    )

    app = make_app(tmp_path)
    client = make_client(app)

    response = client.get("/api/collections/intl/records")

    assert response.status_code == 200
    record = response.json()["records"][0]
    assert record["document"] == "こんにちは世界 🌍"
    assert record["metadata"] == {"source": "日本語ファイル.txt", "emoji": "🔥"}


def test_records_paging_consistency_large_collection(tmp_path: Path, make_app, make_client) -> None:
    """Paging through a 500-record collection page by page (limit=50) must
    return every id exactly once: no duplicates, no gaps, and the union must
    match the full set of ids added."""
    total_count = 500
    _add_records(tmp_path, "big", total_count)

    app = make_app(tmp_path)
    client = make_client(app)

    limit = 50
    seen_ids: list[str] = []
    offset = 0
    while True:
        response = client.get(f"/api/collections/big/records?limit={limit}&offset={offset}")
        assert response.status_code == 200
        body = response.json()
        assert body["total"] == total_count
        page_records = body["records"]
        if not page_records:
            break
        seen_ids.extend(record["id"] for record in page_records)
        offset += limit
        if offset > total_count + limit:
            # Safety valve against an infinite loop if pagination is broken.
            break

    assert len(seen_ids) == total_count
    assert len(set(seen_ids)) == total_count
    assert set(seen_ids) == {str(i) for i in range(total_count)}


def test_records_get_returns_only_requested_ids(tmp_path: Path, make_app, make_client) -> None:
    _add_records(tmp_path, "foo", 5)

    app = make_app(tmp_path)
    client = make_client(app)

    response = client.post("/api/collections/foo/records/get", json={"ids": ["1", "3"]})

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 2
    assert {record["id"] for record in body["records"]} == {"1", "3"}


def test_records_get_nonexistent_id_returns_empty(tmp_path: Path, make_app, make_client) -> None:
    _add_records(tmp_path, "foo", 3)

    app = make_app(tmp_path)
    client = make_client(app)

    response = client.post("/api/collections/foo/records/get", json={"ids": ["does-not-exist"]})

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 0
    assert body["records"] == []


def test_records_get_with_embeddings_include(tmp_path: Path, make_app, make_client) -> None:
    _add_records(tmp_path, "foo", 3, with_embeddings=True)

    app = make_app(tmp_path)
    client = make_client(app)

    response = client.post(
        "/api/collections/foo/records/get",
        json={"ids": ["2"], "include": ["documents", "metadatas", "uris", "embeddings"]},
    )

    assert response.status_code == 200
    record = response.json()["records"][0]
    assert record["embedding_dimension"] == 2
    assert record["embedding_preview"] == [2.0, 3.0]


def test_records_get_nonexistent_collection_returns_404(
    tmp_path: Path, make_app, make_client
) -> None:
    app = make_app(tmp_path)
    client = make_client(app)

    response = client.post(
        "/api/collections/does-not-exist/records/get", json={"ids": ["1"]}
    )

    assert response.status_code == 404


def test_records_get_requires_token(tmp_path: Path, make_app, make_client) -> None:
    _add_records(tmp_path, "foo", 1)

    app = make_app(tmp_path)
    client = make_client(app, token=None)

    response = client.post("/api/collections/foo/records/get", json={"ids": ["0"]})

    assert response.status_code == 401


def test_records_get_invalid_include_returns_422(tmp_path: Path, make_app, make_client) -> None:
    _add_records(tmp_path, "foo", 1)

    app = make_app(tmp_path)
    client = make_client(app)

    response = client.post(
        "/api/collections/foo/records/get",
        json={"ids": ["0"], "include": ["documents", "bogus"]},
    )

    assert response.status_code == 422


def test_records_get_empty_ids_list_returns_empty(tmp_path: Path, make_app, make_client) -> None:
    """ids=[] is a valid (if odd) empty selection and scopes to zero
    records with a 200. chromadb's collection.get(ids=[]) raises
    ValueError('Expected IDs to be a non-empty list...'), so chroma_adapter
    short-circuits and returns an empty page without calling into chromadb
    when ids is an empty list."""
    _add_records(tmp_path, "foo", 5)

    app = make_app(tmp_path)
    client = make_client(app)

    response = client.post("/api/collections/foo/records/get", json={"ids": []})

    assert response.status_code == 200
    body = response.json()
    assert body["records"] == []
    assert body["total"] == 0


def test_records_get_duplicate_ids_returns_each_occurrence(
    tmp_path: Path, make_app, make_client
) -> None:
    """Duplicate ids in the request body (e.g. a UI double-submitting the
    same id) previously caused chromadb's collection.get(ids=...) to raise
    DuplicateIDError, propagating as an unhandled 500. chroma_adapter now
    dedupes the ids (preserving order) before calling into chromadb, so each
    requested id is represented exactly once in the response."""
    _add_records(tmp_path, "foo", 5)

    app = make_app(tmp_path)
    client = make_client(app)

    response = client.post(
        "/api/collections/foo/records/get", json={"ids": ["1", "1", "2"]}
    )

    assert response.status_code == 200
    body = response.json()
    assert [r["id"] for r in body["records"]] == ["1", "2"]
    assert body["total"] == 2


def test_records_get_large_ids_list(tmp_path: Path, make_app, make_client) -> None:
    """150 requested ids (> the 500 limit's usual page size, and > typical
    default limit of 50) must all come back -- ids-based lookup bypasses
    limit/offset entirely per the adapter's documented contract."""
    _add_records(tmp_path, "foo", 200)

    app = make_app(tmp_path)
    client = make_client(app)

    requested_ids = [str(i) for i in range(150)]
    response = client.post(
        "/api/collections/foo/records/get", json={"ids": requested_ids}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 150
    assert len(body["records"]) == 150
    assert {record["id"] for record in body["records"]} == set(requested_ids)


def test_records_get_ids_ignores_limit(tmp_path: Path, make_app, make_client) -> None:
    """When ids is given, limit must be ignored (per chroma_adapter.get_records
    docstring: 'paging (limit/offset) is not applied by Chroma to the ids list
    itself'). Request 10 ids with a limit far below 10; all 10 must return."""
    _add_records(tmp_path, "foo", 20)

    app = make_app(tmp_path)
    client = make_client(app)

    requested_ids = [str(i) for i in range(10)]
    response = client.post(
        "/api/collections/foo/records/get",
        json={"ids": requested_ids, "limit": 1},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 10
    assert len(body["records"]) == 10
    assert {record["id"] for record in body["records"]} == set(requested_ids)


def test_records_get_ids_ignores_offset(tmp_path: Path, make_app, make_client) -> None:
    """Likewise offset must be ignored when ids is given."""
    _add_records(tmp_path, "foo", 20)

    app = make_app(tmp_path)
    client = make_client(app)

    requested_ids = [str(i) for i in range(5)]
    response = client.post(
        "/api/collections/foo/records/get",
        json={"ids": requested_ids, "offset": 100},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 5
    assert {record["id"] for record in body["records"]} == set(requested_ids)


def test_records_get_large_document(tmp_path: Path, make_app, make_client) -> None:
    """A single record with a document body of ~50KB must round-trip intact
    through the API (no truncation)."""
    collection_name = "foo"
    chromadb_module = __import__("chromadb")
    client_lib = chromadb_module.PersistentClient(path=str(tmp_path))
    collection = client_lib.create_collection(collection_name)
    large_document = "x" * 50_000
    collection.add(ids=["big"], documents=[large_document], metadatas=[{"idx": 0}])

    app = make_app(tmp_path)
    client = make_client(app)

    response = client.post(
        "/api/collections/foo/records/get", json={"ids": ["big"]}
    )

    assert response.status_code == 200
    record = response.json()["records"][0]
    assert record["document"] == large_document
    assert len(record["document"]) == 50_000


def test_records_get_nested_metadata_is_out_of_range(tmp_path: Path, make_app, make_client) -> None:
    """ChromaDB metadata values must be str/int/float/bool/None (no nested
    dict/list); attempting to add a record with nested metadata must fail at
    the chromadb layer, not silently corrupt/flatten it. This documents that
    'metadata がネスト JSON' is not actually representable by the current
    chroma_adapter/records schema (RecordInfo.metadata: dict | None assumes a
    flat dict of chromadb-primitive values), which is a spec-relevant
    constraint rather than a chromaw bug."""
    chromadb_module = __import__("chromadb")
    client_lib = chromadb_module.PersistentClient(path=str(tmp_path))
    collection = client_lib.create_collection("foo")

    with pytest.raises(Exception):
        collection.add(
            ids=["1"],
            documents=["doc"],
            metadatas=[{"nested": {"a": 1}}],
        )


def test_records_get_ids_null_falls_back_to_paged_listing(
    tmp_path: Path, make_app, make_client
) -> None:
    """ids omitted entirely (None) must behave like the paged GET endpoint,
    honoring limit/offset and total == collection.count()."""
    _add_records(tmp_path, "foo", 10)

    app = make_app(tmp_path)
    client = make_client(app)

    response = client.post(
        "/api/collections/foo/records/get", json={"limit": 3, "offset": 0}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 10
    assert len(body["records"]) == 3


def test_records_get_where_returns_matching_records(
    tmp_path: Path, make_app, make_client
) -> None:
    _add_records(tmp_path, "foo", 5)

    app = make_app(tmp_path)
    client = make_client(app)

    response = client.post(
        "/api/collections/foo/records/get", json={"where": {"idx": 2}}
    )

    assert response.status_code == 200
    body = response.json()
    assert {record["id"] for record in body["records"]} == {"2"}


def test_records_get_where_document_returns_matching_records(
    tmp_path: Path, make_app, make_client
) -> None:
    _add_records(tmp_path, "foo", 5)

    app = make_app(tmp_path)
    client = make_client(app)

    response = client.post(
        "/api/collections/foo/records/get",
        json={"where_document": {"$contains": "doc-3"}},
    )

    assert response.status_code == 200
    body = response.json()
    assert {record["id"] for record in body["records"]} == {"3"}


def test_records_get_invalid_where_returns_422(
    tmp_path: Path, make_app, make_client
) -> None:
    _add_records(tmp_path, "foo", 3)

    app = make_app(tmp_path)
    client = make_client(app)

    response = client.post(
        "/api/collections/foo/records/get",
        json={"where": {"idx": {"$badop": 1}}},
    )

    assert response.status_code == 422


def test_records_get_where_requires_token(tmp_path: Path, make_app, make_client) -> None:
    _add_records(tmp_path, "foo", 3)

    app = make_app(tmp_path)
    client = make_client(app, token=None)

    response = client.post(
        "/api/collections/foo/records/get", json={"where": {"idx": 0}}
    )

    assert response.status_code == 401


def test_records_get_where_and_operator_returns_matching_records(
    tmp_path: Path, make_app, make_client
) -> None:
    _add_records(tmp_path, "foo", 5)

    app = make_app(tmp_path)
    client = make_client(app)

    response = client.post(
        "/api/collections/foo/records/get",
        json={"where": {"$and": [{"idx": {"$gte": 1}}, {"idx": {"$lte": 3}}]}},
    )

    assert response.status_code == 200
    body = response.json()
    assert {record["id"] for record in body["records"]} == {"1", "2", "3"}


def test_records_get_where_or_operator_returns_matching_records(
    tmp_path: Path, make_app, make_client
) -> None:
    _add_records(tmp_path, "foo", 5)

    app = make_app(tmp_path)
    client = make_client(app)

    response = client.post(
        "/api/collections/foo/records/get",
        json={"where": {"$or": [{"idx": 0}, {"idx": 4}]}},
    )

    assert response.status_code == 200
    body = response.json()
    assert {record["id"] for record in body["records"]} == {"0", "4"}


def test_records_get_where_in_operator_returns_matching_records(
    tmp_path: Path, make_app, make_client
) -> None:
    _add_records(tmp_path, "foo", 5)

    app = make_app(tmp_path)
    client = make_client(app)

    response = client.post(
        "/api/collections/foo/records/get",
        json={"where": {"idx": {"$in": [1, 3]}}},
    )

    assert response.status_code == 200
    body = response.json()
    assert {record["id"] for record in body["records"]} == {"1", "3"}


def test_records_get_where_numeric_value_returns_matching_records(
    tmp_path: Path, make_app, make_client
) -> None:
    _add_records(tmp_path, "foo", 5)

    app = make_app(tmp_path)
    client = make_client(app)

    response = client.post(
        "/api/collections/foo/records/get", json={"where": {"idx": 2}}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["records"][0]["metadata"]["idx"] == 2


def test_records_get_where_bool_value_returns_matching_records(
    tmp_path: Path, make_app, make_client
) -> None:
    import chromadb

    client_lib = chromadb.PersistentClient(path=str(tmp_path))
    collection = client_lib.create_collection("foo")
    collection.add(
        ids=["1", "2"],
        documents=["a", "b"],
        metadatas=[{"active": True}, {"active": False}],
    )

    app = make_app(tmp_path)
    client = make_client(app)

    response = client.post(
        "/api/collections/foo/records/get", json={"where": {"active": True}}
    )

    assert response.status_code == 200
    body = response.json()
    assert {record["id"] for record in body["records"]} == {"1"}


def test_records_get_where_document_contains_empty_string_returns_422(
    tmp_path: Path, make_app, make_client
) -> None:
    """chromadb rejects an empty-string $contains operand; the API surfaces
    this as a 422 InvalidFilterError, not a 200 "matches everything"."""
    _add_records(tmp_path, "foo", 3)

    app = make_app(tmp_path)
    client = make_client(app)

    response = client.post(
        "/api/collections/foo/records/get",
        json={"where_document": {"$contains": ""}},
    )

    assert response.status_code == 422


def test_records_get_where_and_where_document_combined(
    tmp_path: Path, make_app, make_client
) -> None:
    _add_records(tmp_path, "foo", 5)

    app = make_app(tmp_path)
    client = make_client(app)

    response = client.post(
        "/api/collections/foo/records/get",
        json={"where": {"idx": 3}, "where_document": {"$contains": "doc-3"}},
    )

    assert response.status_code == 200
    body = response.json()
    assert {record["id"] for record in body["records"]} == {"3"}


def test_records_get_where_and_ids_combined(
    tmp_path: Path, make_app, make_client
) -> None:
    _add_records(tmp_path, "foo", 5)

    app = make_app(tmp_path)
    client = make_client(app)

    response = client.post(
        "/api/collections/foo/records/get",
        json={"ids": ["0", "1", "2"], "where": {"idx": {"$gte": 1}}},
    )

    assert response.status_code == 200
    body = response.json()
    assert {record["id"] for record in body["records"]} == {"1", "2"}


def test_records_get_where_zero_matches_returns_empty(
    tmp_path: Path, make_app, make_client
) -> None:
    _add_records(tmp_path, "foo", 3)

    app = make_app(tmp_path)
    client = make_client(app)

    response = client.post(
        "/api/collections/foo/records/get", json={"where": {"idx": 999}}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["records"] == []
    assert body["total"] == 0


def test_records_get_where_offset_beyond_matches_returns_empty_records(
    tmp_path: Path, make_app, make_client
) -> None:
    _add_records(tmp_path, "foo", 3)

    app = make_app(tmp_path)
    client = make_client(app)

    response = client.post(
        "/api/collections/foo/records/get",
        json={"where": {"idx": {"$gte": 0}}, "limit": 10, "offset": 100},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["records"] == []
    # Documented approximation: total tracks offset+len(records) while
    # filtering, not the true match count.
    assert body["total"] == 100


def test_records_paging_has_more_true_then_false(
    tmp_path: Path, make_app, make_client
) -> None:
    _add_records(tmp_path, "foo", 10)

    app = make_app(tmp_path)
    client = make_client(app)

    response1 = client.get("/api/collections/foo/records?limit=4&offset=0")
    assert response1.status_code == 200
    body1 = response1.json()
    assert body1["has_more"] is True

    response2 = client.get("/api/collections/foo/records?limit=4&offset=8")
    assert response2.status_code == 200
    body2 = response2.json()
    assert body2["has_more"] is False


def test_records_get_ids_has_more_always_false(
    tmp_path: Path, make_app, make_client
) -> None:
    _add_records(tmp_path, "foo", 10)

    app = make_app(tmp_path)
    client = make_client(app)

    response = client.post(
        "/api/collections/foo/records/get",
        json={"ids": [str(i) for i in range(10)], "limit": 1},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["has_more"] is False
    assert body["total"] == 10


def test_records_get_where_paging_has_more_transitions_and_reaches_all_matches(
    tmp_path: Path, make_app, make_client
) -> None:
    """120 records match a where filter with limit=50: has_more must go
    True -> True -> False across pages, and paging through must reach every
    matching record exactly once."""
    _add_records(tmp_path, "foo", 200)

    app = make_app(tmp_path)
    client = make_client(app)

    limit = 50
    offset = 0
    seen_ids: list[str] = []
    has_more_flags: list[bool] = []
    for _ in range(10):
        response = client.post(
            "/api/collections/foo/records/get",
            json={"where": {"idx": {"$lt": 120}}, "limit": limit, "offset": offset},
        )
        assert response.status_code == 200
        body = response.json()
        seen_ids.extend(record["id"] for record in body["records"])
        has_more_flags.append(body["has_more"])
        if not body["has_more"]:
            break
        offset += limit

    assert has_more_flags == [True, True, False]
    assert len(seen_ids) == 120
    assert set(seen_ids) == {str(i) for i in range(120)}


def test_patch_record_metadata_reflected_in_get(tmp_path: Path, make_app, make_client) -> None:
    _add_records(tmp_path, "foo", 3)

    app = make_app(tmp_path, write=True)
    client = make_client(app)

    response = client.patch(
        "/api/collections/foo/records/1", json={"metadata": {"idx": 99, "tag": "x"}}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["metadata"] == {"idx": 99, "tag": "x"}

    get_response = client.get("/api/collections/foo/records?limit=10")
    updated = next(r for r in get_response.json()["records"] if r["id"] == "1")
    assert updated["metadata"] == {"idx": 99, "tag": "x"}


def test_patch_record_uri_only_update(tmp_path: Path, make_app, make_client) -> None:
    _add_records(tmp_path, "foo", 1)

    app = make_app(tmp_path, write=True)
    client = make_client(app)

    response = client.patch(
        "/api/collections/foo/records/0", json={"uri": "file:///new-uri"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["uri"] == "file:///new-uri"
    # metadata untouched
    assert body["metadata"] == {"idx": 0}


def test_patch_record_read_only_returns_403(tmp_path: Path, make_app, make_client) -> None:
    _add_records(tmp_path, "foo", 1)

    app = make_app(tmp_path, write=False)
    client = make_client(app)

    response = client.patch("/api/collections/foo/records/0", json={"metadata": {"a": 1}})

    assert response.status_code == 403


def test_patch_record_without_token_returns_401(tmp_path: Path, make_app, make_client) -> None:
    _add_records(tmp_path, "foo", 1)

    app = make_app(tmp_path, write=True)
    client = make_client(app, token=None)

    response = client.patch("/api/collections/foo/records/0", json={"metadata": {"a": 1}})

    assert response.status_code == 401


def test_patch_record_nonexistent_record_returns_404(tmp_path: Path, make_app, make_client) -> None:
    _add_records(tmp_path, "foo", 1)

    app = make_app(tmp_path, write=True)
    client = make_client(app)

    response = client.patch(
        "/api/collections/foo/records/does-not-exist", json={"metadata": {"a": 1}}
    )

    assert response.status_code == 404


def test_patch_record_nonexistent_collection_returns_404(
    tmp_path: Path, make_app, make_client
) -> None:
    app = make_app(tmp_path, write=True)
    client = make_client(app)

    response = client.patch(
        "/api/collections/missing/records/0", json={"metadata": {"a": 1}}
    )

    assert response.status_code == 404


def test_patch_record_nested_dict_metadata_returns_422(
    tmp_path: Path, make_app, make_client
) -> None:
    _add_records(tmp_path, "foo", 1)

    app = make_app(tmp_path, write=True)
    client = make_client(app)

    response = client.patch(
        "/api/collections/foo/records/0", json={"metadata": {"a": {"nested": 1}}}
    )

    assert response.status_code == 422


def test_patch_record_list_value_metadata_returns_422(
    tmp_path: Path, make_app, make_client
) -> None:
    _add_records(tmp_path, "foo", 1)

    app = make_app(tmp_path, write=True)
    client = make_client(app)

    response = client.patch("/api/collections/foo/records/0", json={"metadata": {"a": [1, 2]}})

    assert response.status_code == 422


def test_patch_record_both_metadata_and_uri_none_returns_422(
    tmp_path: Path, make_app, make_client
) -> None:
    _add_records(tmp_path, "foo", 1)

    app = make_app(tmp_path, write=True)
    client = make_client(app)

    response = client.patch("/api/collections/foo/records/0", json={})

    assert response.status_code == 422


def test_patch_record_empty_metadata_dict_returns_422(tmp_path: Path, make_app, make_client) -> None:
    """``{}`` is rejected by ``RecordUpdateRequest`` validation with a 422
    before it ever reaches chromadb's ``collection.update()`` (which itself
    would reject an empty metadata dict with a ``ValueError``, and which also
    merges rather than replaces metadata, so an empty dict would otherwise be
    a silent no-op). See
    test_chroma_adapter.test_update_record_empty_metadata_dict_raises_value_error
    for the adapter-level equivalent."""
    _add_records(tmp_path, "foo", 1)

    app = make_app(tmp_path, write=True)
    client = make_client(app)

    response = client.patch("/api/collections/foo/records/0", json={"metadata": {}})

    assert response.status_code == 422


def test_patch_record_bool_metadata_value_round_trips_as_bool(
    tmp_path: Path, make_app, make_client
) -> None:
    """``bool`` is a subclass of ``int``; make sure the request validator's
    per-value type check (which special-cases ``bool`` before the
    ``int``/``float`` branch) doesn't coerce it, and that it round-trips
    as a JSON bool through the PATCH response."""
    _add_records(tmp_path, "foo", 1)

    app = make_app(tmp_path, write=True)
    client = make_client(app)

    response = client.patch(
        "/api/collections/foo/records/0", json={"metadata": {"flag": True, "off": False}}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["metadata"]["flag"] is True
    assert body["metadata"]["off"] is False


def test_patch_record_empty_string_uri(tmp_path: Path, make_app, make_client) -> None:
    _add_records(tmp_path, "foo", 1)

    app = make_app(tmp_path, write=True)
    client = make_client(app)

    response = client.patch("/api/collections/foo/records/0", json={"uri": ""})

    assert response.status_code == 200
    assert response.json()["uri"] == ""


def test_patch_record_non_ascii_metadata_value(tmp_path: Path, make_app, make_client) -> None:
    _add_records(tmp_path, "foo", 1)

    app = make_app(tmp_path, write=True)
    client = make_client(app)

    response = client.patch(
        "/api/collections/foo/records/0", json={"metadata": {"name": "日本語テスト🎉"}}
    )

    assert response.status_code == 200
    assert response.json()["metadata"]["name"] == "日本語テスト🎉"


def test_patch_record_large_metadata(tmp_path: Path, make_app, make_client) -> None:
    _add_records(tmp_path, "foo", 1)

    app = make_app(tmp_path, write=True)
    client = make_client(app)

    large_metadata = {f"k{i}": "v" * 100 for i in range(500)}
    response = client.patch(
        "/api/collections/foo/records/0", json={"metadata": large_metadata}
    )

    assert response.status_code == 200
    # chromadb metadata update merges into the record's existing metadata
    # (here {"idx": 0} from _add_records) rather than replacing it.
    assert response.json()["metadata"] == {"idx": 0, **large_metadata}


def test_patch_record_persists_across_app_instances(tmp_path: Path, make_app, make_client) -> None:
    """A PATCH via one app/adapter instance must be durably visible to a
    freshly-created app/adapter over the same persistent directory."""
    _add_records(tmp_path, "foo", 1)

    app1 = make_app(tmp_path, write=True)
    client1 = make_client(app1)
    response = client1.patch(
        "/api/collections/foo/records/0",
        json={"metadata": {"idx": 0, "updated": True}, "uri": "file:///new"},
    )
    assert response.status_code == 200

    app2 = make_app(tmp_path, write=False)
    client2 = make_client(app2)
    get_response = client2.get("/api/collections/foo/records?limit=10")
    updated = next(r for r in get_response.json()["records"] if r["id"] == "0")
    assert updated["metadata"] == {"idx": 0, "updated": True}
    assert updated["uri"] == "file:///new"


def test_patch_record_document_reflected_and_marks_stale(
    tmp_path: Path, make_app, make_client
) -> None:
    """document update (technical-spec §3.3): document is reflected in a
    subsequent GET, metadata gets chromaw_embedding_status=stale, and the
    embedding itself is left unchanged."""
    _add_records(tmp_path, "foo", 1, with_embeddings=True)

    app = make_app(tmp_path, write=True)
    client = make_client(app)

    before = client.post(
        "/api/collections/foo/records/get",
        json={"ids": ["0"], "include": ["embeddings"]},
    ).json()["records"][0]

    response = client.patch(
        "/api/collections/foo/records/0",
        json={"document": "new document text", "embedding_mode": "keep"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["document"] == "new document text"
    assert body["metadata"]["chromaw_embedding_status"] == "stale"
    # pre-existing metadata (idx) must be preserved, not clobbered.
    assert body["metadata"]["idx"] == 0

    get_response = client.get("/api/collections/foo/records?limit=10")
    updated = next(r for r in get_response.json()["records"] if r["id"] == "0")
    assert updated["document"] == "new document text"
    assert updated["metadata"]["chromaw_embedding_status"] == "stale"

    after = client.post(
        "/api/collections/foo/records/get",
        json={"ids": ["0"], "include": ["embeddings"]},
    ).json()["records"][0]
    assert after["embedding_preview"] == before["embedding_preview"]
    assert after["embedding_dimension"] == before["embedding_dimension"]


def test_patch_record_document_without_embedding_mode_returns_422(
    tmp_path: Path, make_app, make_client
) -> None:
    _add_records(tmp_path, "foo", 1)

    app = make_app(tmp_path, write=True)
    client = make_client(app)

    response = client.patch(
        "/api/collections/foo/records/0", json={"document": "new text"}
    )

    assert response.status_code == 422


def test_patch_record_document_and_metadata_together(
    tmp_path: Path, make_app, make_client
) -> None:
    """document + user-supplied metadata in the same PATCH: both must apply,
    and the stale flag must be merged into the user's metadata rather than
    overwriting it."""
    _add_records(tmp_path, "foo", 1)

    app = make_app(tmp_path, write=True)
    client = make_client(app)

    response = client.patch(
        "/api/collections/foo/records/0",
        json={
            "document": "new text",
            "embedding_mode": "keep",
            "metadata": {"tag": "reviewed"},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["document"] == "new text"
    assert body["metadata"]["tag"] == "reviewed"
    assert body["metadata"]["chromaw_embedding_status"] == "stale"


def test_patch_record_document_read_only_returns_403(
    tmp_path: Path, make_app, make_client
) -> None:
    _add_records(tmp_path, "foo", 1)

    app = make_app(tmp_path, write=False)
    client = make_client(app)

    response = client.patch(
        "/api/collections/foo/records/0",
        json={"document": "new text", "embedding_mode": "keep"},
    )

    assert response.status_code == 403


def test_patch_record_document_without_token_returns_401(
    tmp_path: Path, make_app, make_client
) -> None:
    _add_records(tmp_path, "foo", 1)

    app = make_app(tmp_path, write=True)
    client = make_client(app, token=None)

    response = client.patch(
        "/api/collections/foo/records/0",
        json={"document": "new text", "embedding_mode": "keep"},
    )

    assert response.status_code == 401


def test_patch_record_empty_document_is_valid(
    tmp_path: Path, make_app, make_client
) -> None:
    """An empty string document is a legitimate value (distinct from
    omitting document, which is None/unset) and must be accepted, not
    treated as a no-op."""
    _add_records(tmp_path, "foo", 1)

    app = make_app(tmp_path, write=True)
    client = make_client(app)

    response = client.patch(
        "/api/collections/foo/records/0",
        json={"document": "", "embedding_mode": "keep"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["document"] == ""
    assert body["metadata"]["chromaw_embedding_status"] == "stale"


def test_health_embedding_available_true_with_explicit_config(
    tmp_path: Path, make_app, make_client
) -> None:
    from chromaw.embedding import EmbeddingConfig, EmbeddingResolver

    app = make_app(tmp_path, write=False)
    app.state.adapter.embedding_resolver = EmbeddingResolver(
        EmbeddingConfig(provider="default")
    )
    client = make_client(app)

    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["embedding_available"] is True


def test_patch_record_document_unknown_embedding_mode_returns_422(
    tmp_path: Path, make_app, make_client
) -> None:
    _add_records(tmp_path, "foo", 1)

    app = make_app(tmp_path, write=True)
    client = make_client(app)

    response = client.patch(
        "/api/collections/foo/records/0",
        json={"document": "new text", "embedding_mode": "manual"},
    )

    assert response.status_code == 422


def test_patch_record_document_reembed_computes_fresh_vector_and_marks_fresh(
    tmp_path: Path, make_app, make_client
) -> None:
    """embedding_mode="reembed" (roadmap M3-3): the vector changes, the
    record is not flagged stale, and any prior stale flag is cleared."""
    from unittest import mock

    from chromaw import embedding as embedding_module
    from chromaw.embedding import EmbeddingConfig, EmbeddingResolver

    _add_records(tmp_path, "foo", 1, with_embeddings=True)

    app = make_app(tmp_path, write=True)
    app.state.adapter.embedding_resolver = EmbeddingResolver(
        EmbeddingConfig(provider="default")
    )
    client = make_client(app)

    before = client.post(
        "/api/collections/foo/records/get",
        json={"ids": ["0"], "include": ["embeddings"]},
    ).json()["records"][0]

    class _MockEF:
        def __call__(self, input: list[str]) -> list[list[float]]:
            return [[7.0, 7.0] for _ in input]

    with mock.patch.object(
        embedding_module, "_build_embedding_function", return_value=_MockEF()
    ):
        response = client.patch(
            "/api/collections/foo/records/0",
            json={"document": "new document text", "embedding_mode": "reembed"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["document"] == "new document text"
    assert body["metadata"]["chromaw_embedding_status"] == "fresh"
    assert body["metadata"]["idx"] == 0

    after = client.post(
        "/api/collections/foo/records/get",
        json={"ids": ["0"], "include": ["embeddings"]},
    ).json()["records"][0]
    assert after["embedding_preview"] == pytest.approx([7.0, 7.0])
    assert after["embedding_preview"] != before["embedding_preview"]


def test_patch_record_document_reembed_unavailable_returns_503_and_leaves_record(
    tmp_path: Path, make_app, make_client
) -> None:
    """No embedding function available for reembed: 503, and the record
    (document/metadata/embedding) must be entirely untouched."""
    from unittest import mock

    _add_records(tmp_path, "foo", 1, with_embeddings=True)

    app = make_app(tmp_path, write=True)
    client = make_client(app)

    before = client.post(
        "/api/collections/foo/records/get",
        json={"ids": ["0"], "include": ["embeddings"]},
    ).json()["records"][0]

    real_collection = app.state.adapter._client.get_collection("foo")
    real_collection._embedding_function = None
    with mock.patch.object(
        app.state.adapter._client, "get_collection", return_value=real_collection
    ):
        response = client.patch(
            "/api/collections/foo/records/0",
            json={"document": "new document text", "embedding_mode": "reembed"},
        )

    assert response.status_code == 503

    after = client.post(
        "/api/collections/foo/records/get",
        json={"ids": ["0"], "include": ["embeddings"]},
    ).json()["records"][0]
    assert after["document"] == before["document"]
    assert after["metadata"] == before["metadata"]
    assert after["embedding_preview"] == before["embedding_preview"]


def test_patch_record_embedding_mode_without_document_is_ignored(
    tmp_path: Path, make_app, make_client
) -> None:
    """embedding_mode given alongside metadata but no document: there is no
    document re-embed to gate, so embedding_mode has no effect -- the
    request succeeds (not a validation error, since metadata alone is a
    valid update) and no chromaw_embedding_status sentinel is written."""
    _add_records(tmp_path, "foo", 1)

    app = make_app(tmp_path, write=True)
    client = make_client(app)

    response = client.patch(
        "/api/collections/foo/records/0",
        json={"metadata": {"tag": "reviewed"}, "embedding_mode": "reembed"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["metadata"] == {"idx": 0, "tag": "reviewed"}
    assert "chromaw_embedding_status" not in body["metadata"]


def test_patch_record_embedding_mode_alone_without_document_returns_422(
    tmp_path: Path, make_app, make_client
) -> None:
    """embedding_mode is the *only* field given (no document/metadata/uri):
    rejected by RecordUpdateRequest's "at least one field" guard, same as an
    entirely empty body."""
    _add_records(tmp_path, "foo", 1)

    app = make_app(tmp_path, write=True)
    client = make_client(app)

    response = client.patch(
        "/api/collections/foo/records/0", json={"embedding_mode": "reembed"}
    )

    assert response.status_code == 422


def test_patch_record_document_reembed_with_metadata_merges_sentinel(
    tmp_path: Path, make_app, make_client
) -> None:
    """reembed + caller-supplied metadata together at the API layer: both
    apply, and the sentinel ends up "fresh" alongside the user's metadata
    key rather than being clobbered by it (order independence)."""
    from unittest import mock

    from chromaw import embedding as embedding_module
    from chromaw.embedding import EmbeddingConfig, EmbeddingResolver

    _add_records(tmp_path, "foo", 1, with_embeddings=True)

    app = make_app(tmp_path, write=True)
    app.state.adapter.embedding_resolver = EmbeddingResolver(
        EmbeddingConfig(provider="default")
    )
    client = make_client(app)

    class _MockEF:
        def __call__(self, input: list[str]) -> list[list[float]]:
            return [[2.0, 2.0] for _ in input]

    with mock.patch.object(
        embedding_module, "_build_embedding_function", return_value=_MockEF()
    ):
        response = client.patch(
            "/api/collections/foo/records/0",
            json={
                "document": "new document text",
                "embedding_mode": "reembed",
                "metadata": {"tag": "reviewed"},
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["document"] == "new document text"
    assert body["metadata"]["tag"] == "reviewed"
    assert body["metadata"]["chromaw_embedding_status"] == "fresh"


def test_patch_record_document_audit_records_embedding_mode(
    tmp_path: Path, make_app, make_client
) -> None:
    _add_records(tmp_path, "foo", 1)

    app = make_app(tmp_path, write=True)
    client = make_client(app)

    response = client.patch(
        "/api/collections/foo/records/0",
        json={"document": "new text", "embedding_mode": "keep"},
    )
    assert response.status_code == 200

    audit_path = tmp_path / ".chromaw" / "audit.jsonl"
    entries = [json.loads(line) for line in audit_path.read_text().splitlines()]
    assert entries[-1]["embedding_mode"] == "keep"


def test_patch_record_metadata_only_with_embedding_mode_audit_records_none(
    tmp_path: Path, make_app, make_client
) -> None:
    """embedding_mode given alongside metadata but no document (ignored by
    the handler per test_patch_record_embedding_mode_without_document_is_ignored
    above): the audit entry must record embedding_mode as null rather than
    echoing back the ignored "reembed"/"keep" value, since no re-embed
    actually happened."""
    _add_records(tmp_path, "foo", 1)

    app = make_app(tmp_path, write=True)
    client = make_client(app)

    response = client.patch(
        "/api/collections/foo/records/0",
        json={"metadata": {"tag": "reviewed"}, "embedding_mode": "reembed"},
    )
    assert response.status_code == 200

    audit_path = tmp_path / ".chromaw" / "audit.jsonl"
    entries = [json.loads(line) for line in audit_path.read_text().splitlines()]
    assert entries[-1]["embedding_mode"] is None


def test_post_diff_basic(tmp_path: Path, make_app, make_client) -> None:
    app = make_app(tmp_path)
    client = make_client(app)

    response = client.post(
        "/api/diff", json={"before": "hello\n", "after": "world\n"}
    )

    assert response.status_code == 200
    diff = response.json()["diff"]
    assert "-hello" in diff
    assert "+world" in diff
    assert "--- before" in diff
    assert "+++ after" in diff


def test_post_diff_no_changes_returns_empty_diff(
    tmp_path: Path, make_app, make_client
) -> None:
    app = make_app(tmp_path)
    client = make_client(app)

    response = client.post(
        "/api/diff", json={"before": "same text", "after": "same text"}
    )

    assert response.status_code == 200
    assert response.json()["diff"] == ""


def test_post_diff_without_token_returns_401(
    tmp_path: Path, make_app, make_client
) -> None:
    app = make_app(tmp_path)
    client = make_client(app, token=None)

    response = client.post("/api/diff", json={"before": "a", "after": "b"})

    assert response.status_code == 401


def test_post_diff_read_only_mode_is_allowed(
    tmp_path: Path, make_app, make_client
) -> None:
    """POST /api/diff has no side effects, so it must work even without
    --write (unlike PATCH, it does not depend on require_write_mode)."""
    app = make_app(tmp_path, write=False)
    client = make_client(app)

    response = client.post("/api/diff", json={"before": "a", "after": "b"})

    assert response.status_code == 200


def test_post_diff_multiline_with_custom_labels(
    tmp_path: Path, make_app, make_client
) -> None:
    app = make_app(tmp_path)
    client = make_client(app)

    before = "line1\nline2\nline3\n"
    after = "line1\nline2 changed\nline3\nline4\n"

    response = client.post(
        "/api/diff",
        json={
            "before": before,
            "after": after,
            "before_label": "old.txt",
            "after_label": "new.txt",
        },
    )

    assert response.status_code == 200
    diff = response.json()["diff"]
    assert "--- old.txt" in diff
    assert "+++ new.txt" in diff
    assert "-line2" in diff
    assert "+line2 changed" in diff
    assert "+line4" in diff


def test_post_diff_no_trailing_newline(tmp_path: Path, make_app, make_client) -> None:
    app = make_app(tmp_path)
    client = make_client(app)

    response = client.post(
        "/api/diff", json={"before": "hello", "after": "world"}
    )

    assert response.status_code == 200
    diff = response.json()["diff"]
    assert "-hello" in diff
    assert "+world" in diff


def test_post_diff_crlf_line_endings(tmp_path: Path, make_app, make_client) -> None:
    app = make_app(tmp_path)
    client = make_client(app)

    response = client.post(
        "/api/diff",
        json={"before": "line1\r\nline2\r\n", "after": "line1\r\nline2 changed\r\n"},
    )

    assert response.status_code == 200
    diff = response.json()["diff"]
    assert "line2" in diff
    assert "changed" in diff


def test_post_diff_unicode(tmp_path: Path, make_app, make_client) -> None:
    app = make_app(tmp_path)
    client = make_client(app)

    response = client.post(
        "/api/diff",
        json={"before": "こんにちは 🎉\n", "after": "さようなら 🚀\n"},
    )

    assert response.status_code == 200
    diff = response.json()["diff"]
    assert "こんにちは" in diff
    assert "さようなら" in diff
    assert "🎉" in diff
    assert "🚀" in diff


def test_post_diff_large_input(tmp_path: Path, make_app, make_client) -> None:
    app = make_app(tmp_path)
    client = make_client(app)

    before_lines = [f"line {i}\n" for i in range(5000)]
    after_lines = list(before_lines)
    after_lines[2500] = "line 2500 CHANGED\n"

    response = client.post(
        "/api/diff",
        json={"before": "".join(before_lines), "after": "".join(after_lines)},
    )

    assert response.status_code == 200
    diff = response.json()["diff"]
    assert "-line 2500" in diff
    assert "+line 2500 CHANGED" in diff


def test_post_diff_empty_strings(tmp_path: Path, make_app, make_client) -> None:
    app = make_app(tmp_path)
    client = make_client(app)

    response = client.post("/api/diff", json={"before": "", "after": ""})

    assert response.status_code == 200
    assert response.json()["diff"] == ""


def test_post_diff_missing_required_field_returns_422(
    tmp_path: Path, make_app, make_client
) -> None:
    app = make_app(tmp_path)
    client = make_client(app)

    response = client.post("/api/diff", json={"before": "a"})
    assert response.status_code == 422


# --- DELETE .../records/{id} (roadmap M2-7) ---


def test_delete_record_confirm_match_deletes(tmp_path: Path, make_app, make_client) -> None:
    _add_records(tmp_path, "foo", 2)

    app = make_app(tmp_path, write=True)
    client = make_client(app)

    response = client.request(
        "DELETE", "/api/collections/foo/records/0", json={"confirm": "0"}
    )

    assert response.status_code == 200
    assert response.json() == {"deleted": True, "id": "0"}

    remaining = client.get("/api/collections/foo/records?limit=10").json()["records"]
    assert [r["id"] for r in remaining] == ["1"]


def test_delete_record_confirm_mismatch_returns_409(
    tmp_path: Path, make_app, make_client
) -> None:
    _add_records(tmp_path, "foo", 1)

    app = make_app(tmp_path, write=True)
    client = make_client(app)

    response = client.request(
        "DELETE", "/api/collections/foo/records/0", json={"confirm": "wrong"}
    )

    assert response.status_code == 409
    remaining = client.get("/api/collections/foo/records?limit=10").json()["records"]
    assert [r["id"] for r in remaining] == ["0"]


def test_delete_record_missing_confirm_returns_422(
    tmp_path: Path, make_app, make_client
) -> None:
    _add_records(tmp_path, "foo", 1)

    app = make_app(tmp_path, write=True)
    client = make_client(app)

    response = client.request("DELETE", "/api/collections/foo/records/0", json={})

    assert response.status_code == 422


def test_delete_record_nonexistent_record_returns_404(
    tmp_path: Path, make_app, make_client
) -> None:
    _add_records(tmp_path, "foo", 1)

    app = make_app(tmp_path, write=True)
    client = make_client(app)

    response = client.request(
        "DELETE", "/api/collections/foo/records/missing", json={"confirm": "missing"}
    )

    assert response.status_code == 404


def test_delete_record_nonexistent_collection_returns_404(
    tmp_path: Path, make_app, make_client
) -> None:
    app = make_app(tmp_path, write=True)
    client = make_client(app)

    response = client.request(
        "DELETE", "/api/collections/missing/records/0", json={"confirm": "0"}
    )

    assert response.status_code == 404


def test_delete_record_read_only_returns_403(tmp_path: Path, make_app, make_client) -> None:
    _add_records(tmp_path, "foo", 1)

    app = make_app(tmp_path, write=False)
    client = make_client(app)

    response = client.request(
        "DELETE", "/api/collections/foo/records/0", json={"confirm": "0"}
    )

    assert response.status_code == 403


def test_delete_record_without_token_returns_401(tmp_path: Path, make_app, make_client) -> None:
    _add_records(tmp_path, "foo", 1)

    app = make_app(tmp_path, write=True)
    client = make_client(app, token=None)

    response = client.request(
        "DELETE", "/api/collections/foo/records/0", json={"confirm": "0"}
    )

    assert response.status_code == 401


def test_delete_record_writes_audit_entry(tmp_path: Path, make_app, make_client) -> None:
    _add_records(tmp_path, "foo", 1)

    app = make_app(tmp_path, write=True)
    client = make_client(app)

    client.request("DELETE", "/api/collections/foo/records/0", json={"confirm": "0"})

    audit_path = tmp_path / ".chromaw" / "audit.jsonl"
    lines = audit_path.read_text().strip().splitlines()
    entry = json.loads(lines[-1])
    assert entry["operation"] == "record.delete"
    assert entry["collection"] == "foo"
    assert entry["id"] == "0"
    assert entry["before"]["metadata"] == {"idx": 0}


# --- DELETE .../collections/{name} (roadmap M2-7) ---


def test_delete_collection_confirm_match_deletes(tmp_path: Path, make_app, make_client) -> None:
    _add_records(tmp_path, "foo", 1)

    app = make_app(tmp_path, write=True)
    client = make_client(app)

    response = client.request("DELETE", "/api/collections/foo", json={"confirm": "foo"})

    assert response.status_code == 200
    assert response.json() == {"deleted": True, "id": "foo"}
    assert client.get("/api/collections").json()["collections"] == []


def test_delete_collection_confirm_mismatch_returns_409(
    tmp_path: Path, make_app, make_client
) -> None:
    _add_records(tmp_path, "foo", 1)

    app = make_app(tmp_path, write=True)
    client = make_client(app)

    response = client.request("DELETE", "/api/collections/foo", json={"confirm": "bar"})

    assert response.status_code == 409
    names = [c["name"] for c in client.get("/api/collections").json()["collections"]]
    assert names == ["foo"]


def test_delete_collection_nonexistent_returns_404(
    tmp_path: Path, make_app, make_client
) -> None:
    app = make_app(tmp_path, write=True)
    client = make_client(app)

    response = client.request(
        "DELETE", "/api/collections/missing", json={"confirm": "missing"}
    )

    assert response.status_code == 404


def test_delete_collection_read_only_returns_403(tmp_path: Path, make_app, make_client) -> None:
    _add_records(tmp_path, "foo", 1)

    app = make_app(tmp_path, write=False)
    client = make_client(app)

    response = client.request("DELETE", "/api/collections/foo", json={"confirm": "foo"})

    assert response.status_code == 403


def test_delete_collection_without_token_returns_401(
    tmp_path: Path, make_app, make_client
) -> None:
    _add_records(tmp_path, "foo", 1)

    app = make_app(tmp_path, write=True)
    client = make_client(app, token=None)

    response = client.request("DELETE", "/api/collections/foo", json={"confirm": "foo"})

    assert response.status_code == 401


def test_delete_collection_writes_audit_entry(tmp_path: Path, make_app, make_client) -> None:
    _add_records(tmp_path, "foo", 1)

    app = make_app(tmp_path, write=True)
    client = make_client(app)

    client.request("DELETE", "/api/collections/foo", json={"confirm": "foo"})

    audit_path = tmp_path / ".chromaw" / "audit.jsonl"
    entry = json.loads(audit_path.read_text().strip().splitlines()[-1])
    assert entry["operation"] == "collection.delete"
    assert entry["collection"] == "foo"


# --- PATCH .../collections/{name} rename (roadmap M2-7) ---


def test_rename_collection_confirm_match_renames(
    tmp_path: Path, make_app, make_client
) -> None:
    _add_records(tmp_path, "foo", 1)

    app = make_app(tmp_path, write=True)
    client = make_client(app)

    response = client.patch(
        "/api/collections/foo", json={"name": "bar", "confirm": "foo"}
    )

    assert response.status_code == 200
    assert response.json()["name"] == "bar"
    names = [c["name"] for c in client.get("/api/collections").json()["collections"]]
    assert names == ["bar"]


def test_rename_collection_old_name_returns_404_after_rename(
    tmp_path: Path, make_app, make_client
) -> None:
    _add_records(tmp_path, "foo", 1)

    app = make_app(tmp_path, write=True)
    client = make_client(app)

    client.patch("/api/collections/foo", json={"name": "bar", "confirm": "foo"})

    old_name_response = client.get("/api/collections/foo/records?limit=10")
    assert old_name_response.status_code == 404

    new_name_response = client.get("/api/collections/bar/records?limit=10")
    assert new_name_response.status_code == 200


def test_rename_collection_confirm_mismatch_returns_409(
    tmp_path: Path, make_app, make_client
) -> None:
    _add_records(tmp_path, "foo", 1)

    app = make_app(tmp_path, write=True)
    client = make_client(app)

    response = client.patch(
        "/api/collections/foo", json={"name": "bar", "confirm": "wrong"}
    )

    assert response.status_code == 409
    names = [c["name"] for c in client.get("/api/collections").json()["collections"]]
    assert names == ["foo"]


def test_rename_collection_missing_confirm_returns_422(
    tmp_path: Path, make_app, make_client
) -> None:
    _add_records(tmp_path, "foo", 1)

    app = make_app(tmp_path, write=True)
    client = make_client(app)

    response = client.patch("/api/collections/foo", json={"name": "bar"})

    assert response.status_code == 422


def test_rename_collection_duplicate_name_returns_409(
    tmp_path: Path, make_app, make_client
) -> None:
    _add_records(tmp_path, "foo", 1)
    _add_records(tmp_path, "bar", 1)

    app = make_app(tmp_path, write=True)
    client = make_client(app)

    response = client.patch(
        "/api/collections/foo", json={"name": "bar", "confirm": "foo"}
    )

    assert response.status_code == 409


def test_rename_collection_nonexistent_returns_404(
    tmp_path: Path, make_app, make_client
) -> None:
    app = make_app(tmp_path, write=True)
    client = make_client(app)

    response = client.patch(
        "/api/collections/missing", json={"name": "bar", "confirm": "missing"}
    )

    assert response.status_code == 404


def test_rename_collection_read_only_returns_403(tmp_path: Path, make_app, make_client) -> None:
    _add_records(tmp_path, "foo", 1)

    app = make_app(tmp_path, write=False)
    client = make_client(app)

    response = client.patch(
        "/api/collections/foo", json={"name": "bar", "confirm": "foo"}
    )

    assert response.status_code == 403


def test_rename_collection_without_token_returns_401(
    tmp_path: Path, make_app, make_client
) -> None:
    _add_records(tmp_path, "foo", 1)

    app = make_app(tmp_path, write=True)
    client = make_client(app, token=None)

    response = client.patch(
        "/api/collections/foo", json={"name": "bar", "confirm": "foo"}
    )

    assert response.status_code == 401


def test_rename_collection_writes_audit_entry(tmp_path: Path, make_app, make_client) -> None:
    _add_records(tmp_path, "foo", 1)

    app = make_app(tmp_path, write=True)
    client = make_client(app)

    client.patch("/api/collections/foo", json={"name": "bar", "confirm": "foo"})

    audit_path = tmp_path / ".chromaw" / "audit.jsonl"
    entry = json.loads(audit_path.read_text().strip().splitlines()[-1])
    assert entry["operation"] == "collection.rename"
    assert entry["changes"]["name"] == {"before": "foo", "after": "bar"}


def test_patch_collection_metadata_only_does_not_require_confirm(
    tmp_path: Path, make_app, make_client
) -> None:
    _add_records(tmp_path, "foo", 1)

    app = make_app(tmp_path, write=True)
    client = make_client(app)

    response = client.patch("/api/collections/foo", json={"metadata": {"k": "v"}})

    assert response.status_code == 200
    assert response.json()["metadata"] == {"k": "v"}


def test_patch_collection_empty_body_returns_422(tmp_path: Path, make_app, make_client) -> None:
    _add_records(tmp_path, "foo", 1)

    app = make_app(tmp_path, write=True)
    client = make_client(app)

    response = client.patch("/api/collections/foo", json={})

    assert response.status_code == 422


# --- M2-7 edge cases: confirm exact-match, re-delete, rename no-op, i18n audit ---


def test_delete_record_confirm_with_surrounding_whitespace_returns_409(
    tmp_path: Path, make_app, make_client
) -> None:
    _add_records(tmp_path, "foo", 1)

    app = make_app(tmp_path, write=True)
    client = make_client(app)

    response = client.request(
        "DELETE", "/api/collections/foo/records/0", json={"confirm": " 0 "}
    )

    assert response.status_code == 409


def test_delete_collection_confirm_wrong_case_returns_409(
    tmp_path: Path, make_app, make_client
) -> None:
    _add_records(tmp_path, "Foo", 1)

    app = make_app(tmp_path, write=True)
    client = make_client(app)

    response = client.request("DELETE", "/api/collections/Foo", json={"confirm": "foo"})

    assert response.status_code == 409


def test_delete_record_second_delete_returns_404(tmp_path: Path, make_app, make_client) -> None:
    _add_records(tmp_path, "foo", 1)

    app = make_app(tmp_path, write=True)
    client = make_client(app)

    first = client.request("DELETE", "/api/collections/foo/records/0", json={"confirm": "0"})
    assert first.status_code == 200

    second = client.request("DELETE", "/api/collections/foo/records/0", json={"confirm": "0"})
    assert second.status_code == 404


def test_delete_collection_second_delete_returns_404(
    tmp_path: Path, make_app, make_client
) -> None:
    _add_records(tmp_path, "foo", 1)

    app = make_app(tmp_path, write=True)
    client = make_client(app)

    first = client.request("DELETE", "/api/collections/foo", json={"confirm": "foo"})
    assert first.status_code == 200

    second = client.request("DELETE", "/api/collections/foo", json={"confirm": "foo"})
    assert second.status_code == 404


def test_rename_collection_to_same_name(tmp_path: Path, make_app, make_client) -> None:
    _add_records(tmp_path, "foo", 1)

    app = make_app(tmp_path, write=True)
    client = make_client(app)

    response = client.patch(
        "/api/collections/foo", json={"name": "foo", "confirm": "foo"}
    )

    # Documents observed behavior: renaming a collection to its own current
    # name is accepted as a no-op rename (chromadb allows modify(name=same)),
    # not rejected as a duplicate-name conflict.
    assert response.status_code == 200
    assert response.json()["name"] == "foo"


def test_collection_usable_under_new_name_after_rename(
    tmp_path: Path, make_app, make_client
) -> None:
    _add_records(tmp_path, "foo", 1)

    app = make_app(tmp_path, write=True)
    client = make_client(app)

    rename = client.patch("/api/collections/foo", json={"name": "bar", "confirm": "foo"})
    assert rename.status_code == 200

    records = client.get("/api/collections/bar/records")
    assert records.status_code == 200
    assert records.json()["total"] == 1

    patch = client.patch(
        "/api/collections/bar/records/0", json={"metadata": {"k": "v"}}
    )
    assert patch.status_code == 200
    assert patch.json()["metadata"]["k"] == "v"


def test_delete_collection_with_japanese_metadata_writes_audit_entry(
    tmp_path: Path, make_app, make_client
) -> None:
    _add_records(tmp_path, "foo", 1)

    app = make_app(tmp_path, write=True)
    client = make_client(app)

    client.patch("/api/collections/foo", json={"metadata": {"名前": "日本語の値"}})

    client.request("DELETE", "/api/collections/foo", json={"confirm": "foo"})

    audit_path = tmp_path / ".chromaw" / "audit.jsonl"
    lines = audit_path.read_text().strip().splitlines()
    entry = json.loads(lines[-1])
    assert entry["operation"] == "collection.delete"
    assert entry["collection"] == "foo"
    assert entry["before"]["metadata"]["名前"] == "日本語の値"


# ---------------------------------------------------------------------------
# POST /api/collections/{name}/query (technical-spec §5.6 4, §8.4, roadmap
# M3-1)
#
# Same network-free constraint as tests/test_chroma_adapter.py's
# query_records tests: query_embedding is exercised for the happy paths
# (embeddings supplied explicitly, no embedding function ever invoked);
# query_text is only exercised via its 422 (mutual-exclusivity) and mocked
# 503 (embedding function failure) error paths.
# ---------------------------------------------------------------------------


def _add_query_records(tmp_path: Path):
    import chromadb

    client_lib = chromadb.PersistentClient(path=str(tmp_path))
    collection = client_lib.create_collection("foo")
    collection.add(
        ids=["near", "mid", "far"],
        documents=["doc-near", "doc-mid", "doc-far"],
        metadatas=[{"idx": 0}, {"idx": 1}, {"idx": 2}],
        embeddings=[[0.0, 0.0], [1.0, 0.0], [10.0, 0.0]],
    )
    return collection


def test_query_orders_matches_by_distance(tmp_path: Path, make_app, make_client) -> None:
    _add_query_records(tmp_path)
    app = make_app(tmp_path)
    client = make_client(app)

    response = client.post(
        "/api/collections/foo/query", json={"query_embedding": [0.0, 0.0]}
    )

    assert response.status_code == 200
    body = response.json()
    assert [m["id"] for m in body["matches"]] == ["near", "mid", "far"]
    assert body["matches"][0]["distance"] == 0.0


def test_query_n_results_limits_matches(tmp_path: Path, make_app, make_client) -> None:
    _add_query_records(tmp_path)
    app = make_app(tmp_path)
    client = make_client(app)

    response = client.post(
        "/api/collections/foo/query",
        json={"query_embedding": [0.0, 0.0], "n_results": 2},
    )

    assert response.status_code == 200
    assert len(response.json()["matches"]) == 2


def test_query_where_narrows_matches(tmp_path: Path, make_app, make_client) -> None:
    _add_query_records(tmp_path)
    app = make_app(tmp_path)
    client = make_client(app)

    response = client.post(
        "/api/collections/foo/query",
        json={"query_embedding": [0.0, 0.0], "where": {"idx": 2}},
    )

    assert response.status_code == 200
    assert [m["id"] for m in response.json()["matches"]] == ["far"]


def test_query_include_documents_metadatas(tmp_path: Path, make_app, make_client) -> None:
    _add_query_records(tmp_path)
    app = make_app(tmp_path)
    client = make_client(app)

    response = client.post(
        "/api/collections/foo/query",
        json={
            "query_embedding": [0.0, 0.0],
            "include": ["documents", "metadatas", "distances"],
        },
    )

    assert response.status_code == 200
    match = response.json()["matches"][0]
    assert match["document"] == "doc-near"
    assert match["metadata"] == {"idx": 0}


def test_query_nonexistent_collection_returns_404(
    tmp_path: Path, make_app, make_client
) -> None:
    app = make_app(tmp_path)
    client = make_client(app)

    response = client.post(
        "/api/collections/does-not-exist/query", json={"query_embedding": [0.0, 0.0]}
    )

    assert response.status_code == 404


def test_query_requires_token(tmp_path: Path, make_app, make_client) -> None:
    _add_query_records(tmp_path)
    app = make_app(tmp_path)
    client = make_client(app, token=None)

    response = client.post(
        "/api/collections/foo/query", json={"query_embedding": [0.0, 0.0]}
    )

    assert response.status_code == 401


def test_query_available_in_read_only_mode(tmp_path: Path, make_app, make_client) -> None:
    """Similarity search is a read operation, so it must not be gated by
    require_write_mode (technical-spec §3.2 -- only mutations require
    --write)."""
    _add_query_records(tmp_path)
    app = make_app(tmp_path, write=False)
    client = make_client(app)

    response = client.post(
        "/api/collections/foo/query", json={"query_embedding": [0.0, 0.0]}
    )

    assert response.status_code == 200


def test_query_invalid_include_returns_422(tmp_path: Path, make_app, make_client) -> None:
    _add_query_records(tmp_path)
    app = make_app(tmp_path)
    client = make_client(app)

    response = client.post(
        "/api/collections/foo/query",
        json={"query_embedding": [0.0, 0.0], "include": ["bogus"]},
    )

    assert response.status_code == 422


def test_query_wrong_dimension_embedding_returns_422(
    tmp_path: Path, make_app, make_client
) -> None:
    _add_query_records(tmp_path)
    app = make_app(tmp_path)
    client = make_client(app)

    response = client.post(
        "/api/collections/foo/query",
        json={"query_embedding": [0.0, 0.0, 0.0]},
    )

    assert response.status_code == 422


def test_query_invalid_where_returns_422(tmp_path: Path, make_app, make_client) -> None:
    _add_query_records(tmp_path)
    app = make_app(tmp_path)
    client = make_client(app)

    response = client.post(
        "/api/collections/foo/query",
        json={"query_embedding": [0.0, 0.0], "where": {"idx": {"$bogus": 1}}},
    )

    assert response.status_code == 422


def test_query_neither_text_nor_embedding_returns_422(
    tmp_path: Path, make_app, make_client
) -> None:
    _add_query_records(tmp_path)
    app = make_app(tmp_path)
    client = make_client(app)

    response = client.post("/api/collections/foo/query", json={})

    assert response.status_code == 422


def test_query_both_text_and_embedding_returns_422(
    tmp_path: Path, make_app, make_client
) -> None:
    _add_query_records(tmp_path)
    app = make_app(tmp_path)
    client = make_client(app)

    response = client.post(
        "/api/collections/foo/query",
        json={"query_text": "hello", "query_embedding": [0.0, 0.0]},
    )

    assert response.status_code == 422


def test_query_empty_query_embedding_returns_422(
    tmp_path: Path, make_app, make_client
) -> None:
    _add_query_records(tmp_path)
    app = make_app(tmp_path)
    client = make_client(app)

    response = client.post(
        "/api/collections/foo/query", json={"query_embedding": []}
    )

    assert response.status_code == 422


def test_query_n_results_zero_returns_422(tmp_path: Path, make_app, make_client) -> None:
    _add_query_records(tmp_path)
    app = make_app(tmp_path)
    client = make_client(app)

    response = client.post(
        "/api/collections/foo/query",
        json={"query_embedding": [0.0, 0.0], "n_results": 0},
    )

    assert response.status_code == 422


def test_query_text_embedding_function_failure_returns_503(
    tmp_path: Path, make_app, make_client, monkeypatch
) -> None:
    """query_text delegates to the collection's embedding function; a
    failure there (e.g. unavailable, can't load the model) must surface as
    503, not a generic 500 or 422. The chromadb call is mocked here to
    simulate that failure without requiring network access to actually
    download an embedding model."""
    collection = _add_query_records(tmp_path)
    app = make_app(tmp_path)
    client = make_client(app)

    adapter = app.state.adapter

    def _boom(**kwargs):
        raise ValueError("embedding function unavailable")

    monkeypatch.setattr(collection, "query", _boom)
    monkeypatch.setattr(adapter._client, "get_collection", lambda name: collection)

    response = client.post(
        "/api/collections/foo/query", json={"query_text": "hello"}
    )

    assert response.status_code == 503


def test_query_n_results_500_boundary_accepted(
    tmp_path: Path, make_app, make_client
) -> None:
    _add_query_records(tmp_path)
    app = make_app(tmp_path)
    client = make_client(app)

    response = client.post(
        "/api/collections/foo/query",
        json={"query_embedding": [0.0, 0.0], "n_results": 500},
    )

    assert response.status_code == 200
    assert len(response.json()["matches"]) == 3


def test_query_n_results_501_returns_422(tmp_path: Path, make_app, make_client) -> None:
    _add_query_records(tmp_path)
    app = make_app(tmp_path)
    client = make_client(app)

    response = client.post(
        "/api/collections/foo/query",
        json={"query_embedding": [0.0, 0.0], "n_results": 501},
    )

    assert response.status_code == 422


def test_query_empty_collection_returns_empty_matches(
    tmp_path: Path, make_app, make_client
) -> None:
    import chromadb

    client_lib = chromadb.PersistentClient(path=str(tmp_path))
    client_lib.create_collection("empty")
    app = make_app(tmp_path)
    client = make_client(app)

    response = client.post(
        "/api/collections/empty/query", json={"query_embedding": [0.0, 0.0]}
    )

    assert response.status_code == 200
    assert response.json()["matches"] == []


def test_query_where_zero_matches_returns_empty_list(
    tmp_path: Path, make_app, make_client
) -> None:
    _add_query_records(tmp_path)
    app = make_app(tmp_path)
    client = make_client(app)

    response = client.post(
        "/api/collections/foo/query",
        json={"query_embedding": [0.0, 0.0], "where": {"idx": 999}},
    )

    assert response.status_code == 200
    assert response.json()["matches"] == []


def test_query_zero_vector_embedding_accepted(
    tmp_path: Path, make_app, make_client
) -> None:
    _add_query_records(tmp_path)
    app = make_app(tmp_path)
    client = make_client(app)

    response = client.post(
        "/api/collections/foo/query", json={"query_embedding": [0.0, 0.0]}
    )

    assert response.status_code == 200
    assert response.json()["matches"][0]["distance"] == 0.0


def test_query_include_without_embeddings_leaves_embedding_fields_null(
    tmp_path: Path, make_app, make_client
) -> None:
    _add_query_records(tmp_path)
    app = make_app(tmp_path)
    client = make_client(app)

    response = client.post(
        "/api/collections/foo/query",
        json={"query_embedding": [0.0, 0.0], "include": ["documents", "distances"]},
    )

    assert response.status_code == 200
    for match in response.json()["matches"]:
        assert match["embedding_dimension"] is None
        assert match["embedding_preview"] is None
