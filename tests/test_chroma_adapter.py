from pathlib import Path

import chromadb
import pytest

from chromaw.chroma_adapter import ChromaAdapter
from chromaw.errors import (
    ChromaEmptyDirectoryError,
    ChromaInvalidDirectoryError,
    ChromaPathNotFoundError,
    CollectionAlreadyExistsError,
    CollectionNotFoundError,
    EmbeddingFunctionUnavailableError,
    InvalidCollectionNameError,
    InvalidFilterError,
    InvalidQueryEmbeddingError,
    RecordNotFoundError,
)


def _make_real_chroma_dir(path: Path, collection_names: list[str]) -> None:
    client = chromadb.PersistentClient(path=str(path))
    for name in collection_names:
        client.create_collection(name)


def _names(adapter: ChromaAdapter) -> list[str]:
    return [c.name for c in adapter.list_collections()]


def test_open_existing_chroma_dir_lists_collections(tmp_path: Path) -> None:
    _make_real_chroma_dir(tmp_path, ["foo", "bar"])

    adapter = ChromaAdapter.open(tmp_path)

    assert adapter.path == tmp_path
    assert sorted(_names(adapter)) == ["bar", "foo"]


def test_open_existing_empty_chroma_dir_lists_no_collections(tmp_path: Path) -> None:
    _make_real_chroma_dir(tmp_path, [])

    adapter = ChromaAdapter.open(tmp_path)

    assert adapter.list_collections() == []


def test_open_nonexistent_path_without_create_raises(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"

    with pytest.raises(ChromaPathNotFoundError):
        ChromaAdapter.open(missing)


def test_open_nonexistent_path_with_create_creates_dir(tmp_path: Path) -> None:
    missing = tmp_path / "new-chroma-dir"

    adapter = ChromaAdapter.open(missing, create=True)

    assert missing.exists()
    assert adapter.list_collections() == []


def test_open_empty_directory_without_create_raises(tmp_path: Path) -> None:
    with pytest.raises(ChromaEmptyDirectoryError):
        ChromaAdapter.open(tmp_path)


def test_open_empty_directory_with_create_succeeds(tmp_path: Path) -> None:
    adapter = ChromaAdapter.open(tmp_path, create=True)

    assert adapter.list_collections() == []
    assert (tmp_path / "chroma.sqlite3").exists()


def test_open_non_empty_non_chroma_directory_raises(tmp_path: Path) -> None:
    (tmp_path / "some-unrelated-file.txt").write_text("hello")

    with pytest.raises(ChromaInvalidDirectoryError):
        ChromaAdapter.open(tmp_path)

    # must not have been mutated
    assert not (tmp_path / "chroma.sqlite3").exists()


def test_open_non_empty_non_chroma_directory_with_create_still_raises(tmp_path: Path) -> None:
    (tmp_path / "some-unrelated-file.txt").write_text("hello")

    with pytest.raises(ChromaInvalidDirectoryError):
        ChromaAdapter.open(tmp_path, create=True)


def test_open_corrupted_chroma_sqlite_raises_with_version_in_message(tmp_path: Path) -> None:
    (tmp_path / "chroma.sqlite3").write_text("not a real sqlite database")

    with pytest.raises(ChromaInvalidDirectoryError) as excinfo:
        ChromaAdapter.open(tmp_path)

    message = str(excinfo.value)
    assert chromadb.__version__ in message


def test_open_existing_valid_chroma_dir_with_create_true_succeeds(tmp_path: Path) -> None:
    _make_real_chroma_dir(tmp_path, ["foo"])

    adapter = ChromaAdapter.open(tmp_path, create=True)

    assert _names(adapter) == ["foo"]


def test_open_nested_nonexistent_path_with_create_creates_parents(tmp_path: Path) -> None:
    nested = tmp_path / "a" / "b" / "c"

    adapter = ChromaAdapter.open(nested, create=True)

    assert nested.exists()
    assert adapter.list_collections() == []


def test_open_path_is_a_file_raises(tmp_path: Path) -> None:
    file_path = tmp_path / "im-a-file"
    file_path.write_text("not a directory")

    with pytest.raises(ChromaInvalidDirectoryError):
        ChromaAdapter.open(file_path)


def test_open_path_is_a_file_with_create_still_raises(tmp_path: Path) -> None:
    file_path = tmp_path / "im-a-file"
    file_path.write_text("not a directory")

    with pytest.raises(ChromaInvalidDirectoryError):
        ChromaAdapter.open(file_path, create=True)


def test_open_corrupted_chroma_sqlite_with_create_true_still_raises(tmp_path: Path) -> None:
    (tmp_path / "chroma.sqlite3").write_text("not a real sqlite database")

    with pytest.raises(ChromaInvalidDirectoryError):
        ChromaAdapter.open(tmp_path, create=True)


def test_list_collections_returns_multiple_collections_with_basic_fields(tmp_path: Path) -> None:
    client = chromadb.PersistentClient(path=str(tmp_path))
    client.create_collection("foo", metadata={"a": 1})
    client.create_collection("bar")

    adapter = ChromaAdapter.open(tmp_path)
    collections = {c.name: c for c in adapter.list_collections()}

    assert set(collections) == {"foo", "bar"}
    assert collections["foo"].metadata == {"a": 1}
    assert collections["foo"].count == 0
    assert isinstance(collections["foo"].id, str) and collections["foo"].id


def test_list_collections_on_empty_db_returns_empty_list(tmp_path: Path) -> None:
    chromadb.PersistentClient(path=str(tmp_path))

    adapter = ChromaAdapter.open(tmp_path)

    assert adapter.list_collections() == []


def test_list_collections_estimates_dimension_from_sample_embedding(tmp_path: Path) -> None:
    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.create_collection("foo")
    collection.add(ids=["1", "2"], embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

    adapter = ChromaAdapter.open(tmp_path)
    info = adapter.list_collections()[0]

    assert info.count == 2
    assert info.dimension == 3


def test_list_collections_dimension_is_none_for_empty_collection(tmp_path: Path) -> None:
    client = chromadb.PersistentClient(path=str(tmp_path))
    client.create_collection("foo")

    adapter = ChromaAdapter.open(tmp_path)
    info = adapter.list_collections()[0]

    assert info.count == 0
    assert info.dimension is None


def test_get_records_returns_documents_metadatas_uris(tmp_path: Path) -> None:
    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.create_collection("foo")
    collection.add(
        ids=["1", "2"],
        documents=["doc1", "doc2"],
        metadatas=[{"a": 1}, {"a": 2}],
        embeddings=[[0.1, 0.2], [0.3, 0.4]],
    )

    adapter = ChromaAdapter.open(tmp_path)
    records, total, has_more = adapter.get_records("foo")

    assert total == 2
    assert len(records) == 2
    ids = {r.id for r in records}
    assert ids == {"1", "2"}
    by_id = {r.id: r for r in records}
    assert by_id["1"].document == "doc1"
    assert by_id["1"].metadata == {"a": 1}
    # embeddings not requested by default -> dimension/preview stay None
    assert by_id["1"].embedding_dimension is None
    assert by_id["1"].embedding_preview is None


def test_get_records_limit_restricts_page_size(tmp_path: Path) -> None:
    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.create_collection("foo")
    collection.add(ids=[str(i) for i in range(10)], documents=[f"d{i}" for i in range(10)])

    adapter = ChromaAdapter.open(tmp_path)
    records, total, has_more = adapter.get_records("foo", limit=3, offset=0)

    assert total == 10
    assert len(records) == 3


def test_get_records_offset_beyond_count_returns_empty(tmp_path: Path) -> None:
    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.create_collection("foo")
    collection.add(ids=["1", "2"], documents=["a", "b"])

    adapter = ChromaAdapter.open(tmp_path)
    records, total, has_more = adapter.get_records("foo", limit=50, offset=100)

    assert total == 2
    assert records == []


def test_get_records_include_embeddings_populates_dimension_and_preview(tmp_path: Path) -> None:
    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.create_collection("foo")
    embedding = [float(i) for i in range(12)]
    collection.add(ids=["1"], embeddings=[embedding])

    adapter = ChromaAdapter.open(tmp_path)
    records, _, _ = adapter.get_records(
        "foo", include=("documents", "metadatas", "uris", "embeddings")
    )

    assert records[0].embedding_dimension == 12
    assert records[0].embedding_preview == embedding[:8]


def test_get_records_without_embeddings_include_leaves_dimension_none(tmp_path: Path) -> None:
    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.create_collection("foo")
    collection.add(ids=["1"], embeddings=[[0.1, 0.2, 0.3]])

    adapter = ChromaAdapter.open(tmp_path)
    records, _, _ = adapter.get_records("foo", include=("documents", "metadatas", "uris"))

    assert records[0].embedding_dimension is None
    assert records[0].embedding_preview is None


def test_get_records_nonexistent_collection_raises(tmp_path: Path) -> None:
    chromadb.PersistentClient(path=str(tmp_path))
    adapter = ChromaAdapter.open(tmp_path)

    with pytest.raises(CollectionNotFoundError):
        adapter.get_records("does-not-exist")


def test_get_records_ids_filters_to_requested_records(tmp_path: Path) -> None:
    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.create_collection("foo")
    collection.add(
        ids=[str(i) for i in range(5)],
        documents=[f"d{i}" for i in range(5)],
    )

    adapter = ChromaAdapter.open(tmp_path)
    records, total, has_more = adapter.get_records("foo", ids=["1", "3"])

    assert total == 2
    assert {r.id for r in records} == {"1", "3"}


def test_get_records_ids_with_nonexistent_id_returns_empty(tmp_path: Path) -> None:
    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.create_collection("foo")
    collection.add(ids=["1", "2"], documents=["a", "b"])

    adapter = ChromaAdapter.open(tmp_path)
    records, total, has_more = adapter.get_records("foo", ids=["does-not-exist"])

    assert total == 0
    assert records == []


def test_get_records_ids_with_embeddings_populates_dimension_and_preview(
    tmp_path: Path,
) -> None:
    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.create_collection("foo")
    embedding = [float(i) for i in range(10)]
    collection.add(ids=["1", "2"], embeddings=[embedding, [9.0] * 10])

    adapter = ChromaAdapter.open(tmp_path)
    records, total, has_more = adapter.get_records(
        "foo", ids=["1"], include=("documents", "metadatas", "uris", "embeddings")
    )

    assert total == 1
    assert records[0].embedding_dimension == 10
    assert records[0].embedding_preview == embedding[:8]


def test_get_records_where_equality_filters_by_metadata(tmp_path: Path) -> None:
    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.create_collection("foo")
    collection.add(
        ids=["1", "2", "3"],
        documents=["a", "b", "c"],
        metadatas=[{"source": "x"}, {"source": "y"}, {"source": "x"}],
    )

    adapter = ChromaAdapter.open(tmp_path)
    records, total, has_more = adapter.get_records("foo", where={"source": "x"})

    assert {r.id for r in records} == {"1", "3"}
    assert total == 2


def test_get_records_where_document_contains_filters_by_document(tmp_path: Path) -> None:
    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.create_collection("foo")
    collection.add(
        ids=["1", "2"],
        documents=["hello world", "goodbye"],
    )

    adapter = ChromaAdapter.open(tmp_path)
    records, total, has_more = adapter.get_records(
        "foo", where_document={"$contains": "hello"}
    )

    assert {r.id for r in records} == {"1"}
    assert total == 1


def test_get_records_where_with_paging(tmp_path: Path) -> None:
    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.create_collection("foo")
    collection.add(
        ids=[str(i) for i in range(10)],
        documents=[f"d{i}" for i in range(10)],
        metadatas=[{"group": "a" if i < 6 else "b"} for i in range(10)],
    )

    adapter = ChromaAdapter.open(tmp_path)
    records_page1, total1, has_more1 = adapter.get_records(
        "foo", where={"group": "a"}, limit=4, offset=0
    )
    records_page2, total2, has_more2 = adapter.get_records(
        "foo", where={"group": "a"}, limit=4, offset=4
    )

    assert len(records_page1) == 4
    assert total1 == 4
    assert len(records_page2) == 2
    assert total2 == 6


def test_get_records_invalid_where_raises_invalid_filter_error(tmp_path: Path) -> None:
    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.create_collection("foo")
    collection.add(ids=["1"], documents=["a"], metadatas=[{"source": "x"}])

    adapter = ChromaAdapter.open(tmp_path)

    with pytest.raises(InvalidFilterError):
        adapter.get_records("foo", where={"source": {"$badop": "x"}})


def test_get_records_where_and_operator_combines_conditions(tmp_path: Path) -> None:
    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.create_collection("foo")
    collection.add(
        ids=["1", "2", "3"],
        documents=["a", "b", "c"],
        metadatas=[
            {"source": "x", "group": "a"},
            {"source": "x", "group": "b"},
            {"source": "y", "group": "a"},
        ],
    )

    adapter = ChromaAdapter.open(tmp_path)
    records, total, has_more = adapter.get_records(
        "foo",
        where={"$and": [{"source": "x"}, {"group": "a"}]},
    )

    assert {r.id for r in records} == {"1"}
    assert total == 1


def test_get_records_where_or_operator_combines_conditions(tmp_path: Path) -> None:
    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.create_collection("foo")
    collection.add(
        ids=["1", "2", "3"],
        documents=["a", "b", "c"],
        metadatas=[{"source": "x"}, {"source": "y"}, {"source": "z"}],
    )

    adapter = ChromaAdapter.open(tmp_path)
    records, total, has_more = adapter.get_records(
        "foo",
        where={"$or": [{"source": "x"}, {"source": "z"}]},
    )

    assert {r.id for r in records} == {"1", "3"}
    assert total == 2


def test_get_records_where_in_operator_filters_by_membership(tmp_path: Path) -> None:
    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.create_collection("foo")
    collection.add(
        ids=["1", "2", "3"],
        documents=["a", "b", "c"],
        metadatas=[{"source": "x"}, {"source": "y"}, {"source": "z"}],
    )

    adapter = ChromaAdapter.open(tmp_path)
    records, total, has_more = adapter.get_records(
        "foo",
        where={"source": {"$in": ["x", "z"]}},
    )

    assert {r.id for r in records} == {"1", "3"}
    assert total == 2


def test_get_records_where_numeric_value_filters(tmp_path: Path) -> None:
    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.create_collection("foo")
    collection.add(
        ids=["1", "2", "3"],
        documents=["a", "b", "c"],
        metadatas=[{"count": 1}, {"count": 2}, {"count": 3}],
    )

    adapter = ChromaAdapter.open(tmp_path)
    records, total, has_more = adapter.get_records("foo", where={"count": {"$gte": 2}})

    assert {r.id for r in records} == {"2", "3"}
    assert total == 2


def test_get_records_where_bool_value_filters(tmp_path: Path) -> None:
    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.create_collection("foo")
    collection.add(
        ids=["1", "2"],
        documents=["a", "b"],
        metadatas=[{"active": True}, {"active": False}],
    )

    adapter = ChromaAdapter.open(tmp_path)
    records, total, has_more = adapter.get_records("foo", where={"active": True})

    assert {r.id for r in records} == {"1"}
    assert total == 1


def test_get_records_where_document_contains_empty_string_raises_invalid_filter(
    tmp_path: Path,
) -> None:
    """chromadb rejects an empty-string $contains operand outright rather than
    treating it as "matches everything"; this must surface as InvalidFilterError,
    consistent with other malformed where/where_document handling."""
    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.create_collection("foo")
    collection.add(ids=["1", "2"], documents=["hello world", "goodbye"])

    adapter = ChromaAdapter.open(tmp_path)

    with pytest.raises(InvalidFilterError):
        adapter.get_records("foo", where_document={"$contains": ""})


def test_get_records_where_and_where_document_combined(tmp_path: Path) -> None:
    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.create_collection("foo")
    collection.add(
        ids=["1", "2", "3"],
        documents=["hello world", "hello there", "goodbye"],
        metadatas=[{"source": "x"}, {"source": "y"}, {"source": "x"}],
    )

    adapter = ChromaAdapter.open(tmp_path)
    records, total, has_more = adapter.get_records(
        "foo",
        where={"source": "x"},
        where_document={"$contains": "hello"},
    )

    assert {r.id for r in records} == {"1"}
    assert total == 1


def test_get_records_where_and_ids_combined(tmp_path: Path) -> None:
    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.create_collection("foo")
    collection.add(
        ids=["1", "2", "3"],
        documents=["a", "b", "c"],
        metadatas=[{"source": "x"}, {"source": "x"}, {"source": "y"}],
    )

    adapter = ChromaAdapter.open(tmp_path)
    records, total, has_more = adapter.get_records(
        "foo", ids=["1", "2", "3"], where={"source": "x"}
    )

    assert {r.id for r in records} == {"1", "2"}
    # ids given alongside where: paging (offset+len) approximation applies,
    # per get_records docstring, since where is also given.
    assert total == 2


def test_get_records_where_zero_matches_returns_empty(tmp_path: Path) -> None:
    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.create_collection("foo")
    collection.add(
        ids=["1", "2"],
        documents=["a", "b"],
        metadatas=[{"source": "x"}, {"source": "y"}],
    )

    adapter = ChromaAdapter.open(tmp_path)
    records, total, has_more = adapter.get_records("foo", where={"source": "does-not-exist"})

    assert records == []
    assert total == 0


def test_get_records_where_offset_beyond_matches_returns_empty_but_total_reflects_offset(
    tmp_path: Path,
) -> None:
    """Documented approximation: while filtering, total == offset + len(records),
    not the true match count, so an offset beyond the real match count still
    reports total == offset (with zero records), not the true (smaller) count."""
    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.create_collection("foo")
    collection.add(
        ids=["1", "2", "3"],
        documents=["a", "b", "c"],
        metadatas=[{"source": "x"}, {"source": "x"}, {"source": "y"}],
    )

    adapter = ChromaAdapter.open(tmp_path)
    records, total, has_more = adapter.get_records(
        "foo", where={"source": "x"}, limit=10, offset=50
    )

    assert records == []
    assert total == 50


def test_get_records_has_more_false_without_filter_when_page_not_full(
    tmp_path: Path,
) -> None:
    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.create_collection("foo")
    collection.add(ids=[str(i) for i in range(5)], documents=[f"d{i}" for i in range(5)])

    adapter = ChromaAdapter.open(tmp_path)
    records, total, has_more = adapter.get_records("foo", limit=50, offset=0)

    assert total == 5
    assert len(records) == 5
    assert has_more is False


def test_get_records_has_more_true_without_filter_when_more_pages_remain(
    tmp_path: Path,
) -> None:
    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.create_collection("foo")
    collection.add(ids=[str(i) for i in range(10)], documents=[f"d{i}" for i in range(10)])

    adapter = ChromaAdapter.open(tmp_path)
    records, total, has_more = adapter.get_records("foo", limit=4, offset=0)

    assert total == 10
    assert len(records) == 4
    assert has_more is True

    records_last, total_last, has_more_last = adapter.get_records(
        "foo", limit=4, offset=8
    )
    assert len(records_last) == 2
    assert has_more_last is False


def test_get_records_ids_only_has_more_always_false(tmp_path: Path) -> None:
    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.create_collection("foo")
    collection.add(
        ids=[str(i) for i in range(10)], documents=[f"d{i}" for i in range(10)]
    )

    adapter = ChromaAdapter.open(tmp_path)
    records, total, has_more = adapter.get_records(
        "foo", ids=[str(i) for i in range(10)], limit=1
    )

    assert total == 10
    assert len(records) == 10
    assert has_more is False


def test_get_records_where_paging_has_more_transitions_and_reaches_all_matches(
    tmp_path: Path,
) -> None:
    """120 records match a where filter with limit=50: has_more must go
    True -> True -> False across three pages, and paging through must reach
    every matching record exactly once (regression for has_more correctness
    with limit+1 fetch-and-trim, technical-spec paging semantics)."""
    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.create_collection("foo")
    total_count = 200
    ids = [str(i) for i in range(total_count)]
    documents = [f"d{i}" for i in range(total_count)]
    metadatas = [
        {"group": "match" if i < 120 else "other"} for i in range(total_count)
    ]
    collection.add(ids=ids, documents=documents, metadatas=metadatas)

    adapter = ChromaAdapter.open(tmp_path)
    limit = 50
    offset = 0
    seen_ids: list[str] = []
    has_more_flags: list[bool] = []
    for _ in range(10):
        records, total, has_more = adapter.get_records(
            "foo", where={"group": "match"}, limit=limit, offset=offset
        )
        seen_ids.extend(r.id for r in records)
        has_more_flags.append(has_more)
        if not has_more:
            break
        offset += limit

    assert has_more_flags == [True, True, False]
    assert len(seen_ids) == 120
    assert set(seen_ids) == {str(i) for i in range(120)}


def test_get_records_where_paging_has_more_false_at_exact_multiple_boundary(
    tmp_path: Path,
) -> None:
    """100 records match a where filter with limit=50 (an exact multiple):
    the second (final) page must have has_more False, not a phantom True
    from the limit+1 fetch-and-trim leaking past the true match count."""
    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.create_collection("foo")
    total_count = 150
    ids = [str(i) for i in range(total_count)]
    documents = [f"d{i}" for i in range(total_count)]
    metadatas = [
        {"group": "match" if i < 100 else "other"} for i in range(total_count)
    ]
    collection.add(ids=ids, documents=documents, metadatas=metadatas)

    adapter = ChromaAdapter.open(tmp_path)
    records_page1, total1, has_more1 = adapter.get_records(
        "foo", where={"group": "match"}, limit=50, offset=0
    )
    records_page2, total2, has_more2 = adapter.get_records(
        "foo", where={"group": "match"}, limit=50, offset=50
    )

    assert len(records_page1) == 50
    assert has_more1 is True
    assert len(records_page2) == 50
    assert has_more2 is False
    assert {r.id for r in records_page1} | {r.id for r in records_page2} == {
        str(i) for i in range(100)
    }


def test_update_record_metadata_reflected_in_get(tmp_path: Path) -> None:
    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.create_collection("foo")
    collection.add(ids=["1"], embeddings=[[0.1, 0.2]], metadatas=[{"a": 1}])

    adapter = ChromaAdapter.open(tmp_path)
    adapter.update_record("foo", "1", metadata={"a": 2, "b": "x"})

    records, _, _ = adapter.get_records("foo", ids=["1"])
    assert records[0].metadata == {"a": 2, "b": "x"}


def test_update_record_nonexistent_id_raises(tmp_path: Path) -> None:
    client = chromadb.PersistentClient(path=str(tmp_path))
    client.create_collection("foo")

    adapter = ChromaAdapter.open(tmp_path)

    with pytest.raises(RecordNotFoundError):
        adapter.update_record("foo", "missing", metadata={"a": 1})


def test_update_record_empty_metadata_dict_raises_value_error(tmp_path: Path) -> None:
    """chromadb's ``collection.update()`` rejects an empty metadata dict
    outright (``ValueError: Expected metadata to be a non-empty dict``).

    ``{}`` is not ``None``, so it passes through ``update_record``'s
    ``metadata is not None`` check and reaches chromadb, which raises.
    Pinning this so a chromadb upgrade that changes the behavior is
    noticed. Note the API layer (``RecordUpdateRequest``) now rejects an
    empty ``metadata`` dict with a 422 before it would ever reach the
    adapter (see test_server.py); this test exercises the adapter
    directly, bypassing that validation, to confirm chromadb's own
    behavior."""
    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.create_collection("foo")
    collection.add(ids=["1"], embeddings=[[0.1, 0.2]], metadatas=[{"a": 1}])

    adapter = ChromaAdapter.open(tmp_path)

    with pytest.raises(ValueError, match="non-empty dict"):
        adapter.update_record("foo", "1", metadata={})


def test_update_record_bool_metadata_value_preserved_as_bool(tmp_path: Path) -> None:
    """``bool`` is a subclass of ``int`` in Python; make sure a bool
    metadata value round-trips as an actual bool and not ``1``/``0``."""
    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.create_collection("foo")
    collection.add(ids=["1"], embeddings=[[0.1, 0.2]], metadatas=[{"a": 1}])

    adapter = ChromaAdapter.open(tmp_path)
    adapter.update_record("foo", "1", metadata={"flag": True, "off": False})

    records, _, _ = adapter.get_records("foo", ids=["1"])
    metadata = records[0].metadata
    assert metadata["flag"] is True
    assert metadata["off"] is False


def test_update_record_empty_string_uri(tmp_path: Path) -> None:
    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.create_collection("foo")
    collection.add(ids=["1"], embeddings=[[0.1, 0.2]], uris=["file:///old"])

    adapter = ChromaAdapter.open(tmp_path)
    adapter.update_record("foo", "1", uri="")

    records, _, _ = adapter.get_records("foo", ids=["1"])
    assert records[0].uri == ""


def test_update_record_non_ascii_metadata_value(tmp_path: Path) -> None:
    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.create_collection("foo")
    collection.add(ids=["1"], embeddings=[[0.1, 0.2]], metadatas=[{"a": 1}])

    adapter = ChromaAdapter.open(tmp_path)
    adapter.update_record("foo", "1", metadata={"name": "日本語テスト🎉"})

    # chromadb's metadata update merges into the existing dict rather than
    # replacing it wholesale, so the original "a" key survives alongside
    # the new "name" key.
    records, _, _ = adapter.get_records("foo", ids=["1"])
    assert records[0].metadata == {"a": 1, "name": "日本語テスト🎉"}


def test_update_record_large_metadata(tmp_path: Path) -> None:
    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.create_collection("foo")
    collection.add(ids=["1"], embeddings=[[0.1, 0.2]], metadatas=[{"a": 1}])

    adapter = ChromaAdapter.open(tmp_path)
    large_metadata = {f"k{i}": "v" * 100 for i in range(500)}
    adapter.update_record("foo", "1", metadata=large_metadata)

    # merge semantics (see non-ascii test above): original "a" key survives.
    records, _, _ = adapter.get_records("foo", ids=["1"])
    assert records[0].metadata == {"a": 1, **large_metadata}


# Note (no test needed): ``update_record`` does an existence pre-check via
# ``collection.get(ids=[record_id])`` followed by ``collection.update()``.
# There is a TOCTOU window between the two calls (e.g. concurrent delete of
# the same id) where the pre-check could pass but the update race with a
# deletion. This is accepted as out of scope per the M2-2 spec review --
# chromadb's persistent client is not designed for concurrent multi-writer
# access in the first place, and testing the race deterministically would
# require intrusive mocking that doesn't reflect real chromadb internals.


def test_update_record_persists_across_adapter_instances(tmp_path: Path) -> None:
    """Updates written by one ``ChromaAdapter`` instance must be visible
    to a freshly-opened instance against the same persistent directory
    (i.e. the update is actually durable, not just cached in-process)."""
    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.create_collection("foo")
    collection.add(ids=["1"], embeddings=[[0.1, 0.2]], metadatas=[{"a": 1}])

    adapter1 = ChromaAdapter.open(tmp_path)
    adapter1.update_record("foo", "1", metadata={"a": 2, "b": "new"}, uri="file:///new")

    adapter2 = ChromaAdapter.open(tmp_path)
    records, _, _ = adapter2.get_records("foo", ids=["1"])
    assert records[0].metadata == {"a": 2, "b": "new"}
    assert records[0].uri == "file:///new"


def test_update_record_document_updates_document_and_leaves_embedding_untouched(
    tmp_path: Path,
) -> None:
    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.create_collection("foo")
    collection.add(
        ids=["1"], embeddings=[[0.1, 0.2]], documents=["old text"], metadatas=[{"a": 1}]
    )

    adapter = ChromaAdapter.open(tmp_path)
    adapter.update_record("foo", "1", document="new text")

    records, _, _ = adapter.get_records("foo", ids=["1"], include=("documents", "embeddings"))
    assert records[0].document == "new text"
    assert records[0].embedding_preview == pytest.approx([0.1, 0.2])


def test_update_record_mark_stale_without_metadata_sets_status_only(
    tmp_path: Path,
) -> None:
    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.create_collection("foo")
    collection.add(ids=["1"], embeddings=[[0.1, 0.2]], metadatas=[{"a": 1}])

    adapter = ChromaAdapter.open(tmp_path)
    adapter.update_record("foo", "1", document="new text", mark_stale=True)

    records, _, _ = adapter.get_records("foo", ids=["1"])
    assert records[0].metadata == {"a": 1, "chromaw_embedding_status": "stale"}


def test_update_record_mark_stale_merges_into_given_metadata(tmp_path: Path) -> None:
    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.create_collection("foo")
    collection.add(ids=["1"], embeddings=[[0.1, 0.2]], metadatas=[{"a": 1}])

    adapter = ChromaAdapter.open(tmp_path)
    adapter.update_record(
        "foo", "1", document="new text", metadata={"tag": "reviewed"}, mark_stale=True
    )

    records, _, _ = adapter.get_records("foo", ids=["1"])
    assert records[0].metadata == {
        "a": 1,
        "tag": "reviewed",
        "chromaw_embedding_status": "stale",
    }


def test_update_record_no_mark_stale_leaves_metadata_untouched(tmp_path: Path) -> None:
    """document update without mark_stale (e.g. hypothetical future modes)
    must not add the stale flag."""
    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.create_collection("foo")
    collection.add(ids=["1"], embeddings=[[0.1, 0.2]], metadatas=[{"a": 1}])

    adapter = ChromaAdapter.open(tmp_path)
    adapter.update_record("foo", "1", document="new text")

    records, _, _ = adapter.get_records("foo", ids=["1"])
    assert records[0].metadata == {"a": 1}


# --- delete_record / delete_collection / update_collection (roadmap M2-7) ---


def test_delete_record_removes_record(tmp_path: Path) -> None:
    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.create_collection("foo")
    collection.add(ids=["1", "2"], embeddings=[[0.1, 0.2], [0.3, 0.4]])

    adapter = ChromaAdapter.open(tmp_path)
    adapter.delete_record("foo", "1")

    records, _, _ = adapter.get_records("foo")
    assert [r.id for r in records] == ["2"]


def test_delete_record_nonexistent_id_raises(tmp_path: Path) -> None:
    client = chromadb.PersistentClient(path=str(tmp_path))
    client.create_collection("foo")

    adapter = ChromaAdapter.open(tmp_path)

    with pytest.raises(RecordNotFoundError):
        adapter.delete_record("foo", "missing")


def test_delete_record_nonexistent_collection_raises(tmp_path: Path) -> None:
    adapter = ChromaAdapter.open(tmp_path, create=True)

    with pytest.raises(CollectionNotFoundError):
        adapter.delete_record("missing", "1")


def test_delete_record_persists_across_adapter_instances(tmp_path: Path) -> None:
    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.create_collection("foo")
    collection.add(ids=["1"], embeddings=[[0.1, 0.2]])

    adapter1 = ChromaAdapter.open(tmp_path)
    adapter1.delete_record("foo", "1")

    adapter2 = ChromaAdapter.open(tmp_path)
    records, _, _ = adapter2.get_records("foo")
    assert records == []


def test_delete_collection_removes_collection(tmp_path: Path) -> None:
    client = chromadb.PersistentClient(path=str(tmp_path))
    client.create_collection("foo")

    adapter = ChromaAdapter.open(tmp_path)
    adapter.delete_collection("foo")

    assert adapter.list_collections() == []


def test_delete_collection_nonexistent_raises(tmp_path: Path) -> None:
    adapter = ChromaAdapter.open(tmp_path, create=True)

    with pytest.raises(CollectionNotFoundError):
        adapter.delete_collection("missing")


def test_delete_collection_persists_across_adapter_instances(tmp_path: Path) -> None:
    client = chromadb.PersistentClient(path=str(tmp_path))
    client.create_collection("foo")

    adapter1 = ChromaAdapter.open(tmp_path)
    adapter1.delete_collection("foo")

    adapter2 = ChromaAdapter.open(tmp_path)
    assert adapter2.list_collections() == []


def test_update_collection_renames_collection(tmp_path: Path) -> None:
    client = chromadb.PersistentClient(path=str(tmp_path))
    client.create_collection("foo")

    adapter = ChromaAdapter.open(tmp_path)
    info = adapter.update_collection("foo", new_name="bar")

    assert info.name == "bar"
    names = [c.name for c in adapter.list_collections()]
    assert names == ["bar"]


def test_update_collection_rename_nonexistent_raises(tmp_path: Path) -> None:
    adapter = ChromaAdapter.open(tmp_path, create=True)

    with pytest.raises(CollectionNotFoundError):
        adapter.update_collection("missing", new_name="bar")


def test_update_collection_rename_to_existing_name_raises(tmp_path: Path) -> None:
    client = chromadb.PersistentClient(path=str(tmp_path))
    client.create_collection("foo")
    client.create_collection("bar")

    adapter = ChromaAdapter.open(tmp_path)

    with pytest.raises(CollectionAlreadyExistsError):
        adapter.update_collection("foo", new_name="bar")


def test_update_collection_rename_to_invalid_name_raises(tmp_path: Path) -> None:
    client = chromadb.PersistentClient(path=str(tmp_path))
    client.create_collection("foo")

    adapter = ChromaAdapter.open(tmp_path)

    with pytest.raises(InvalidCollectionNameError):
        adapter.update_collection("foo", new_name="a")


def test_update_collection_metadata_merges(tmp_path: Path) -> None:
    client = chromadb.PersistentClient(path=str(tmp_path))
    client.create_collection("foo", metadata={"a": 1})

    adapter = ChromaAdapter.open(tmp_path)
    info = adapter.update_collection("foo", metadata={"b": 2})

    assert info.metadata == {"a": 1, "b": 2}


def test_update_collection_rename_persists_across_adapter_instances(tmp_path: Path) -> None:
    client = chromadb.PersistentClient(path=str(tmp_path))
    client.create_collection("foo")

    adapter1 = ChromaAdapter.open(tmp_path)
    adapter1.update_collection("foo", new_name="bar")

    adapter2 = ChromaAdapter.open(tmp_path)
    names = [c.name for c in adapter2.list_collections()]
    assert names == ["bar"]


# ---------------------------------------------------------------------------
# query_records (technical-spec §5.6 4, §8.4, roadmap M3-1)
#
# These tests exercise the query_embeddings path exclusively: embeddings are
# supplied explicitly to collection.add()/query() so no embedding function
# (and therefore no model download / network access) is ever invoked. The
# query_text path is only exercised for its error handling (mocked, below);
# a happy-path query_text test would require chromadb's default embedding
# function to download the all-MiniLM-L6-v2 model, which is not viable in a
# network-free test environment.
# ---------------------------------------------------------------------------


def _make_query_collection(tmp_path: Path):
    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.create_collection("foo")
    collection.add(
        ids=["near", "mid", "far"],
        documents=["doc-near", "doc-mid", "doc-far"],
        metadatas=[{"idx": 0}, {"idx": 1}, {"idx": 2}],
        embeddings=[[0.0, 0.0], [1.0, 0.0], [10.0, 0.0]],
    )
    return collection


def test_query_records_orders_by_distance(tmp_path: Path) -> None:
    _make_query_collection(tmp_path)
    adapter = ChromaAdapter.open(tmp_path)

    matches = adapter.query_records("foo", query_embedding=[0.0, 0.0], n_results=3)

    assert [m.id for m in matches] == ["near", "mid", "far"]
    assert matches[0].distance == 0.0
    assert matches[1].distance < matches[2].distance


def test_query_records_n_results_limits_matches(tmp_path: Path) -> None:
    _make_query_collection(tmp_path)
    adapter = ChromaAdapter.open(tmp_path)

    matches = adapter.query_records("foo", query_embedding=[0.0, 0.0], n_results=1)

    assert len(matches) == 1
    assert matches[0].id == "near"


def test_query_records_include_controls_fields(tmp_path: Path) -> None:
    _make_query_collection(tmp_path)
    adapter = ChromaAdapter.open(tmp_path)

    matches = adapter.query_records(
        "foo",
        query_embedding=[0.0, 0.0],
        n_results=3,
        include=("documents", "metadatas", "uris", "distances", "embeddings"),
    )

    by_id = {m.id: m for m in matches}
    assert by_id["near"].document == "doc-near"
    assert by_id["near"].metadata == {"idx": 0}
    assert by_id["near"].embedding_dimension == 2
    assert by_id["near"].embedding_preview == [0.0, 0.0]


def test_query_records_where_narrows_candidates(tmp_path: Path) -> None:
    _make_query_collection(tmp_path)
    adapter = ChromaAdapter.open(tmp_path)

    matches = adapter.query_records(
        "foo", query_embedding=[0.0, 0.0], n_results=3, where={"idx": 2}
    )

    assert [m.id for m in matches] == ["far"]


def test_query_records_where_document_narrows_candidates(tmp_path: Path) -> None:
    _make_query_collection(tmp_path)
    adapter = ChromaAdapter.open(tmp_path)

    matches = adapter.query_records(
        "foo",
        query_embedding=[0.0, 0.0],
        n_results=3,
        where_document={"$contains": "mid"},
    )

    assert [m.id for m in matches] == ["mid"]


def test_query_records_invalid_where_raises_invalid_filter_error(tmp_path: Path) -> None:
    _make_query_collection(tmp_path)
    adapter = ChromaAdapter.open(tmp_path)

    with pytest.raises(InvalidFilterError):
        adapter.query_records(
            "foo", query_embedding=[0.0, 0.0], where={"idx": {"$bogus": 1}}
        )


def test_query_records_wrong_dimension_embedding_raises_invalid_query_embedding(
    tmp_path: Path,
) -> None:
    _make_query_collection(tmp_path)
    adapter = ChromaAdapter.open(tmp_path)

    with pytest.raises(InvalidQueryEmbeddingError):
        adapter.query_records("foo", query_embedding=[0.0, 0.0, 0.0])


def test_query_records_nonexistent_collection_raises(tmp_path: Path) -> None:
    chromadb.PersistentClient(path=str(tmp_path))
    adapter = ChromaAdapter.open(tmp_path)

    with pytest.raises(CollectionNotFoundError):
        adapter.query_records("does-not-exist", query_embedding=[0.0, 0.0])


def test_query_records_query_text_embedding_function_failure_raises(
    tmp_path: Path, monkeypatch
) -> None:
    """query_text delegates embedding to the collection's embedding
    function; a failure there (e.g. unavailable, can't load the model) must
    surface as EmbeddingFunctionUnavailableError rather than
    InvalidQueryEmbeddingError. The underlying chromadb call is mocked here
    to simulate that failure without requiring network access."""
    _make_query_collection(tmp_path)
    adapter = ChromaAdapter.open(tmp_path)

    collection = adapter._client.get_collection("foo")

    def _boom(**kwargs):
        raise ValueError("embedding function unavailable")

    monkeypatch.setattr(collection, "query", _boom)
    monkeypatch.setattr(adapter._client, "get_collection", lambda name: collection)

    with pytest.raises(EmbeddingFunctionUnavailableError):
        adapter.query_records("foo", query_text="hello")


def test_query_records_n_results_exceeds_collection_size_returns_all(
    tmp_path: Path,
) -> None:
    """n_results larger than the collection's total record count should not
    error -- chromadb simply returns as many matches as exist."""
    _make_query_collection(tmp_path)
    adapter = ChromaAdapter.open(tmp_path)

    matches = adapter.query_records("foo", query_embedding=[0.0, 0.0], n_results=500)

    assert len(matches) == 3
    assert [m.id for m in matches] == ["near", "mid", "far"]


def test_query_records_empty_collection_returns_no_matches(tmp_path: Path) -> None:
    client = chromadb.PersistentClient(path=str(tmp_path))
    client.create_collection("empty")
    adapter = ChromaAdapter.open(tmp_path)

    matches = adapter.query_records("empty", query_embedding=[0.0, 0.0], n_results=5)

    assert matches == []


def test_query_records_where_yields_zero_matches(tmp_path: Path) -> None:
    """A well-formed where filter that matches nothing returns an empty
    match list, not an error."""
    _make_query_collection(tmp_path)
    adapter = ChromaAdapter.open(tmp_path)

    matches = adapter.query_records(
        "foo", query_embedding=[0.0, 0.0], n_results=3, where={"idx": 999}
    )

    assert matches == []


def test_query_records_zero_vector_embedding_succeeds(tmp_path: Path) -> None:
    """An all-zero query_embedding is a valid (if degenerate) vector -- it
    must not be rejected, and should still return distance-ordered
    matches."""
    _make_query_collection(tmp_path)
    adapter = ChromaAdapter.open(tmp_path)

    matches = adapter.query_records("foo", query_embedding=[0.0, 0.0], n_results=3)

    assert [m.id for m in matches] == ["near", "mid", "far"]
    assert matches[0].distance == 0.0


def test_query_records_embeddings_excluded_from_include_leaves_fields_none(
    tmp_path: Path,
) -> None:
    """When "embeddings" is absent from include, embedding_dimension and
    embedding_preview must stay None even though other embedding-adjacent
    fields (distances) are populated."""
    _make_query_collection(tmp_path)
    adapter = ChromaAdapter.open(tmp_path)

    matches = adapter.query_records(
        "foo",
        query_embedding=[0.0, 0.0],
        n_results=3,
        include=("documents", "distances"),
    )

    assert len(matches) == 3
    for m in matches:
        assert m.embedding_dimension is None
        assert m.embedding_preview is None
        assert m.distance is not None
