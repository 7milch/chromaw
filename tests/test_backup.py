"""Tests for the pre-first-write backup (technical-spec §9.1, roadmap M2-5).

Unit tests exercise ``BackupManager`` directly; integration tests drive it
through the PATCH endpoint via the ``make_app``/``make_client`` fixtures
(see ``tests/conftest.py``).
"""
from __future__ import annotations

import filecmp
from pathlib import Path
from typing import Callable

import chromadb
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from chromaw.backup import BackupManager
from chromaw.chroma_adapter import ChromaAdapter
from chromaw.errors import BackupFailedError


def _make_real_chroma_dir(path: Path, collection_names: list[str]) -> None:
    client = chromadb.PersistentClient(path=str(path))
    for name in collection_names:
        collection = client.create_collection(name)
        collection.add(ids=["1"], documents=["hello"], metadatas=[{"k": "v"}])


# --- BackupManager unit tests -----------------------------------------


def test_ensure_backup_creates_timestamped_copy(tmp_path: Path) -> None:
    _make_real_chroma_dir(tmp_path, ["foo"])
    manager = BackupManager(tmp_path)

    backup_dir = manager.ensure_backup()

    assert backup_dir is not None
    assert backup_dir.is_dir()
    assert backup_dir.parent == tmp_path / ".chromaw" / "backups"
    assert (backup_dir / "chroma.sqlite3").is_file()


def test_ensure_backup_content_matches_original(tmp_path: Path) -> None:
    _make_real_chroma_dir(tmp_path, ["foo"])
    manager = BackupManager(tmp_path)

    backup_dir = manager.ensure_backup()
    assert backup_dir is not None

    original_sqlite = tmp_path / "chroma.sqlite3"
    backup_sqlite = backup_dir / "chroma.sqlite3"
    assert filecmp.cmp(original_sqlite, backup_sqlite, shallow=False)


def test_ensure_backup_excludes_chromaw_dir(tmp_path: Path) -> None:
    _make_real_chroma_dir(tmp_path, ["foo"])
    manager = BackupManager(tmp_path)

    backup_dir = manager.ensure_backup()
    assert backup_dir is not None

    assert not (backup_dir / ".chromaw").exists()


def test_ensure_backup_only_runs_once(tmp_path: Path) -> None:
    _make_real_chroma_dir(tmp_path, ["foo"])
    manager = BackupManager(tmp_path)

    first = manager.ensure_backup()
    second = manager.ensure_backup()

    assert first == second
    backups_root = tmp_path / ".chromaw" / "backups"
    assert len(list(backups_root.iterdir())) == 1


def test_ensure_backup_raises_and_stays_retryable_on_failure(tmp_path: Path) -> None:
    _make_real_chroma_dir(tmp_path, ["foo"])
    manager = BackupManager(tmp_path)

    # Block the backups directory from being created by putting a *file* at
    # the path .chromaw/backups needs to be a directory.
    chromaw_dir = tmp_path / ".chromaw"
    chromaw_dir.mkdir()
    (chromaw_dir / "backups").write_text("not a directory")

    with pytest.raises(BackupFailedError):
        manager.ensure_backup()

    assert manager.done is False

    # Fixing the obstruction lets a later call succeed (not-done state was
    # preserved across the failed attempt).
    (chromaw_dir / "backups").unlink()
    backup_dir = manager.ensure_backup()
    assert backup_dir is not None
    assert manager.done is True


# --- Integration: PATCH endpoint triggers the backup -------------------


@pytest.fixture
def write_app_and_client(
    tmp_path: Path, make_client: Callable[..., TestClient]
) -> tuple[FastAPI, TestClient, Path]:
    _make_real_chroma_dir(tmp_path, ["foo"])
    adapter = ChromaAdapter.open(tmp_path)
    from chromaw.server import create_app

    app = create_app(
        adapter, write=True, token="test-token", host="127.0.0.1", port=8000
    )
    client = make_client(app)
    return app, client, tmp_path


def test_first_patch_creates_backup(
    write_app_and_client: tuple[FastAPI, TestClient, Path],
) -> None:
    app, client, chroma_path = write_app_and_client

    response = client.patch(
        "/api/collections/foo/records/1", json={"metadata": {"k": "v2"}}
    )

    assert response.status_code == 200
    backups_root = chroma_path / ".chromaw" / "backups"
    assert backups_root.is_dir()
    assert len(list(backups_root.iterdir())) == 1


def test_second_patch_does_not_create_new_backup(
    write_app_and_client: tuple[FastAPI, TestClient, Path],
) -> None:
    app, client, chroma_path = write_app_and_client

    client.patch("/api/collections/foo/records/1", json={"metadata": {"k": "v2"}})
    client.patch("/api/collections/foo/records/1", json={"metadata": {"k": "v3"}})

    backups_root = chroma_path / ".chromaw" / "backups"
    assert len(list(backups_root.iterdir())) == 1


def test_read_only_mode_never_creates_backup(
    tmp_path: Path, make_client: Callable[..., TestClient]
) -> None:
    _make_real_chroma_dir(tmp_path, ["foo"])
    adapter = ChromaAdapter.open(tmp_path)
    from chromaw.server import create_app

    app = create_app(adapter, write=False, token="test-token", host="127.0.0.1", port=8000)
    client = make_client(app)

    response = client.patch(
        "/api/collections/foo/records/1", json={"metadata": {"k": "v2"}}
    )

    assert response.status_code == 403
    assert not (tmp_path / ".chromaw").exists()


def test_bulk_delete_creates_backup(
    write_app_and_client: tuple[FastAPI, TestClient, Path],
) -> None:
    """POST .../records/bulk-delete must trigger the same first-write backup
    as PATCH (roadmap M4-2, matching M2-5's "backup before first write")."""
    app, client, chroma_path = write_app_and_client

    response = client.post(
        "/api/collections/foo/records/bulk-delete",
        json={"ids": ["1"], "confirm": "foo"},
    )

    assert response.status_code == 200
    backups_root = chroma_path / ".chromaw" / "backups"
    assert backups_root.is_dir()
    assert len(list(backups_root.iterdir())) == 1


def test_backup_failure_blocks_write_and_returns_500(
    write_app_and_client: tuple[FastAPI, TestClient, Path],
) -> None:
    app, client, chroma_path = write_app_and_client

    # Obstruct the backup destination the same way as the unit test above.
    chromaw_dir = chroma_path / ".chromaw"
    chromaw_dir.mkdir()
    (chromaw_dir / "backups").write_text("not a directory")

    response = client.patch(
        "/api/collections/foo/records/1", json={"metadata": {"k": "v2"}}
    )

    assert response.status_code == 500

    # The write must not have gone through: fetch the record back and check
    # metadata wasn't updated.
    (chromaw_dir / "backups").unlink()
    adapter: ChromaAdapter = app.state.adapter
    records, _, _ = adapter.get_records(
        "foo", ids=["1"], include=("metadatas",)
    )
    assert records[0].metadata == {"k": "v"}
