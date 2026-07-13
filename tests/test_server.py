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
