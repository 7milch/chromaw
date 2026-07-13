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


def test_open_existing_chroma_dir_lists_collections(tmp_path: Path) -> None:
    _make_real_chroma_dir(tmp_path, ["foo", "bar"])

    adapter = ChromaAdapter.open(tmp_path)

    assert adapter.path == tmp_path
    assert sorted(adapter.list_collections()) == ["bar", "foo"]


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

    assert adapter.list_collections() == ["foo"]


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
