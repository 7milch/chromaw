from pathlib import Path

import chromadb
import pytest

from chromaw.chroma_adapter import ChromaAdapter
from chromaw.errors import (
    ChromaEmptyDirectoryError,
    ChromaInvalidDirectoryError,
    ChromaPathNotFoundError,
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
