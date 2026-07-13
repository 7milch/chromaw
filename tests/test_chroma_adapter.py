from pathlib import Path

import chromadb
import pytest

from chromaw.chroma_adapter import ChromaAdapter
from chromaw.errors import (
    ChromaEmptyDirectoryError,
    ChromaInvalidDirectoryError,
    ChromaPathNotFoundError,
    CollectionNotFoundError,
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
    records, total = adapter.get_records("foo")

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
    records, total = adapter.get_records("foo", limit=3, offset=0)

    assert total == 10
    assert len(records) == 3


def test_get_records_offset_beyond_count_returns_empty(tmp_path: Path) -> None:
    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.create_collection("foo")
    collection.add(ids=["1", "2"], documents=["a", "b"])

    adapter = ChromaAdapter.open(tmp_path)
    records, total = adapter.get_records("foo", limit=50, offset=100)

    assert total == 2
    assert records == []


def test_get_records_include_embeddings_populates_dimension_and_preview(tmp_path: Path) -> None:
    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.create_collection("foo")
    embedding = [float(i) for i in range(12)]
    collection.add(ids=["1"], embeddings=[embedding])

    adapter = ChromaAdapter.open(tmp_path)
    records, _ = adapter.get_records(
        "foo", include=("documents", "metadatas", "uris", "embeddings")
    )

    assert records[0].embedding_dimension == 12
    assert records[0].embedding_preview == embedding[:8]


def test_get_records_without_embeddings_include_leaves_dimension_none(tmp_path: Path) -> None:
    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.create_collection("foo")
    collection.add(ids=["1"], embeddings=[[0.1, 0.2, 0.3]])

    adapter = ChromaAdapter.open(tmp_path)
    records, _ = adapter.get_records("foo", include=("documents", "metadatas", "uris"))

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
    records, total = adapter.get_records("foo", ids=["1", "3"])

    assert total == 2
    assert {r.id for r in records} == {"1", "3"}


def test_get_records_ids_with_nonexistent_id_returns_empty(tmp_path: Path) -> None:
    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.create_collection("foo")
    collection.add(ids=["1", "2"], documents=["a", "b"])

    adapter = ChromaAdapter.open(tmp_path)
    records, total = adapter.get_records("foo", ids=["does-not-exist"])

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
    records, total = adapter.get_records(
        "foo", ids=["1"], include=("documents", "metadatas", "uris", "embeddings")
    )

    assert total == 1
    assert records[0].embedding_dimension == 10
    assert records[0].embedding_preview == embedding[:8]
