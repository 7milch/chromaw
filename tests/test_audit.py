"""Tests for the write-operation audit log (technical-spec §9.2, roadmap
M2-6).

Unit tests exercise ``AuditLogger`` directly; integration tests drive it
through the PATCH endpoint via the ``make_app``/``make_client`` fixtures
(see ``tests/conftest.py``), mirroring ``tests/test_backup.py``.
"""
from __future__ import annotations

import json
from pathlib import Path

import chromadb
import pytest

from chromaw.audit import AuditLogger
from chromaw.errors import AuditWriteFailedError

AUDIT_RELPATH = Path(".chromaw") / "audit.jsonl"


def _add_records(tmp_path: Path, name: str, count: int):
    client_lib = chromadb.PersistentClient(path=str(tmp_path))
    collection = client_lib.create_collection(name)
    ids = [str(i) for i in range(count)]
    documents = [f"doc-{i}" for i in range(count)]
    metadatas = [{"idx": i} for i in range(count)]
    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    return collection


def _read_audit_lines(tmp_path: Path) -> list[dict]:
    audit_path = tmp_path / AUDIT_RELPATH
    if not audit_path.is_file():
        return []
    lines = audit_path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line]


# --- AuditLogger unit tests --------------------------------------------


def test_log_update_appends_single_line(tmp_path: Path) -> None:
    logger = AuditLogger(tmp_path)

    logger.log_update(
        collection="foo",
        record_id="1",
        changes={"metadata": {"before": {"a": 1}, "after": {"a": 2}}},
        embedding_stale=False,
    )

    entries = _read_audit_lines(tmp_path)
    assert len(entries) == 1
    entry = entries[0]
    assert entry["operation"] == "record.update"
    assert entry["collection"] == "foo"
    assert entry["id"] == "1"
    assert entry["changes"] == {"metadata": {"before": {"a": 1}, "after": {"a": 2}}}
    assert entry["embedding_stale"] is False
    assert entry["timestamp"].endswith("Z")
    assert entry["user_agent"].startswith("chromaw/")


def test_log_update_creates_chromaw_dir_if_missing(tmp_path: Path) -> None:
    logger = AuditLogger(tmp_path)
    assert not (tmp_path / ".chromaw").exists()

    logger.log_update(
        collection="foo", record_id="1", changes={}, embedding_stale=False
    )

    assert (tmp_path / AUDIT_RELPATH).is_file()


def test_log_update_appends_multiple_entries(tmp_path: Path) -> None:
    logger = AuditLogger(tmp_path)

    logger.log_update(collection="foo", record_id="1", changes={}, embedding_stale=False)
    logger.log_update(collection="foo", record_id="2", changes={}, embedding_stale=True)

    entries = _read_audit_lines(tmp_path)
    assert len(entries) == 2
    assert entries[0]["id"] == "1"
    assert entries[1]["id"] == "2"
    assert entries[1]["embedding_stale"] is True


def test_log_update_raises_on_write_failure(tmp_path: Path) -> None:
    # Block .chromaw from being a directory.
    (tmp_path / ".chromaw").write_text("not a directory")
    logger = AuditLogger(tmp_path)

    with pytest.raises(AuditWriteFailedError):
        logger.log_update(collection="foo", record_id="1", changes={}, embedding_stale=False)


# --- PATCH integration tests --------------------------------------------


def test_patch_metadata_appends_audit_entry(tmp_path: Path, make_app, make_client) -> None:
    _add_records(tmp_path, "foo", 1)
    app = make_app(tmp_path, write=True)
    client = make_client(app)

    response = client.patch(
        "/api/collections/foo/records/0", json={"metadata": {"idx": 99}}
    )
    assert response.status_code == 200

    entries = _read_audit_lines(tmp_path)
    assert len(entries) == 1
    entry = entries[0]
    assert entry["operation"] == "record.update"
    assert entry["collection"] == "foo"
    assert entry["id"] == "0"
    assert entry["changes"]["metadata"]["before"] == {"idx": 0}
    assert entry["changes"]["metadata"]["after"] == {"idx": 99}
    assert "uri" not in entry["changes"]
    assert "document" not in entry["changes"]
    assert entry["embedding_stale"] is False


def test_patch_uri_appends_audit_entry(tmp_path: Path, make_app, make_client) -> None:
    _add_records(tmp_path, "foo", 1)
    app = make_app(tmp_path, write=True)
    client = make_client(app)

    response = client.patch(
        "/api/collections/foo/records/0", json={"uri": "file:///new-uri"}
    )
    assert response.status_code == 200

    entries = _read_audit_lines(tmp_path)
    assert len(entries) == 1
    changes = entries[0]["changes"]
    assert changes["uri"] == {"before": None, "after": "file:///new-uri"}
    assert "metadata" not in changes


def test_patch_document_appends_audit_entry_with_stale_flag(
    tmp_path: Path, make_app, make_client
) -> None:
    _add_records(tmp_path, "foo", 1)
    app = make_app(tmp_path, write=True)
    client = make_client(app)

    response = client.patch(
        "/api/collections/foo/records/0",
        json={"document": "new document text", "embedding_mode": "keep"},
    )
    assert response.status_code == 200

    entries = _read_audit_lines(tmp_path)
    assert len(entries) == 1
    entry = entries[0]
    assert entry["changes"]["document"] == {
        "before": "doc-0",
        "after": "new document text",
    }
    assert entry["embedding_stale"] is True


def test_patch_twice_appends_two_lines(tmp_path: Path, make_app, make_client) -> None:
    _add_records(tmp_path, "foo", 1)
    app = make_app(tmp_path, write=True)
    client = make_client(app)

    client.patch("/api/collections/foo/records/0", json={"metadata": {"idx": 1}})
    client.patch("/api/collections/foo/records/0", json={"metadata": {"idx": 2}})

    entries = _read_audit_lines(tmp_path)
    assert len(entries) == 2
    assert entries[0]["changes"]["metadata"]["after"] == {"idx": 1}
    assert entries[1]["changes"]["metadata"]["before"] == {"idx": 1}
    assert entries[1]["changes"]["metadata"]["after"] == {"idx": 2}


def test_patch_read_only_writes_no_audit_entry(tmp_path: Path, make_app, make_client) -> None:
    _add_records(tmp_path, "foo", 1)
    app = make_app(tmp_path, write=False)
    client = make_client(app)

    response = client.patch("/api/collections/foo/records/0", json={"metadata": {"a": 1}})

    assert response.status_code == 403
    assert not (tmp_path / AUDIT_RELPATH).exists()


def test_patch_nonexistent_record_writes_no_audit_entry(
    tmp_path: Path, make_app, make_client
) -> None:
    _add_records(tmp_path, "foo", 1)
    app = make_app(tmp_path, write=True)
    client = make_client(app)

    response = client.patch(
        "/api/collections/foo/records/does-not-exist", json={"metadata": {"a": 1}}
    )

    assert response.status_code == 404
    assert not (tmp_path / AUDIT_RELPATH).exists()


def test_patch_backup_failure_returns_500_and_no_audit_entry(
    tmp_path: Path, make_app, make_client
) -> None:
    _add_records(tmp_path, "foo", 1)
    app = make_app(tmp_path, write=True)
    client = make_client(app)

    # Obstruct .chromaw before the very first write: the pre-first-write
    # backup (technical-spec §9.1) runs before the mutation and fails
    # closed, so neither the update nor an audit entry happens.
    (tmp_path / ".chromaw").write_text("not a directory")

    response = client.patch("/api/collections/foo/records/0", json={"metadata": {"a": 1}})

    assert response.status_code == 500
    get_response = client.get("/api/collections/foo/records?limit=10")
    updated = next(r for r in get_response.json()["records"] if r["id"] == "0")
    assert updated["metadata"] == {"idx": 0}


def test_patch_audit_write_failure_returns_500(tmp_path: Path, make_app, make_client) -> None:
    _add_records(tmp_path, "foo", 1)
    app = make_app(tmp_path, write=True)
    client = make_client(app)

    # First PATCH succeeds normally, completing the one-time pre-first-write
    # backup so it won't touch .chromaw again on the second call.
    first = client.patch("/api/collections/foo/records/0", json={"metadata": {"idx": 1}})
    assert first.status_code == 200
    assert len(_read_audit_lines(tmp_path)) == 1

    # Now obstruct .chromaw/audit.jsonl's parent so only the post-update
    # audit append fails; the backup step above is already done and is a
    # no-op that never touches .chromaw again.
    audit_dir = tmp_path / ".chromaw"
    import shutil

    shutil.rmtree(audit_dir)
    audit_dir.write_text("not a directory")

    response = client.patch("/api/collections/foo/records/0", json={"metadata": {"idx": 2}})

    assert response.status_code == 500
    # The update itself already went through (chromadb has no notion of
    # "undo" once .update() returns) -- fail-closed here means the client
    # is told the operation failed, not that the mutation is reverted.
    get_response = client.get("/api/collections/foo/records?limit=10")
    updated = next(r for r in get_response.json()["records"] if r["id"] == "0")
    assert updated["metadata"] == {"idx": 2}


# --- Edge cases -----------------------------------------------------------


def test_log_update_non_ascii_round_trips(tmp_path: Path) -> None:
    # ensure_ascii=False is used in AuditLogger._append; verify the file is
    # written as real UTF-8 (not \uXXXX escapes) and round-trips exactly.
    logger = AuditLogger(tmp_path)
    ja_before = "こんにちは世界"
    ja_after = "さようなら世界 🌏"

    logger.log_update(
        collection="コレクション",
        record_id="1",
        changes={"document": {"before": ja_before, "after": ja_after}},
        embedding_stale=True,
    )

    audit_path = tmp_path / AUDIT_RELPATH
    raw = audit_path.read_text(encoding="utf-8")
    assert "\\u" not in raw
    assert ja_before in raw
    assert ja_after in raw

    entries = _read_audit_lines(tmp_path)
    assert entries[0]["collection"] == "コレクション"
    assert entries[0]["changes"]["document"]["before"] == ja_before
    assert entries[0]["changes"]["document"]["after"] == ja_after


def test_patch_non_ascii_metadata_and_document_audit_round_trip(
    tmp_path: Path, make_app, make_client
) -> None:
    _add_records(tmp_path, "foo", 1)
    app = make_app(tmp_path, write=True)
    client = make_client(app)

    response = client.patch(
        "/api/collections/foo/records/0",
        json={
            "metadata": {"idx": 0, "note": "日本語のメタデータ"},
            "document": "新しいドキュメント本文です。",
            "embedding_mode": "keep",
        },
    )
    assert response.status_code == 200

    entries = _read_audit_lines(tmp_path)
    assert len(entries) == 1
    changes = entries[0]["changes"]
    # document updates always mark_stale=True, which the adapter reflects
    # into metadata as chromaw_embedding_status -- accounted for here.
    assert changes["metadata"]["after"] == {
        "idx": 0,
        "note": "日本語のメタデータ",
        "chromaw_embedding_status": "stale",
    }
    assert changes["document"]["after"] == "新しいドキュメント本文です。"


def test_audit_jsonl_lines_are_all_individually_valid_json(
    tmp_path: Path, make_app, make_client
) -> None:
    _add_records(tmp_path, "foo", 1)
    app = make_app(tmp_path, write=True)
    client = make_client(app)

    for i in range(5):
        response = client.patch(
            "/api/collections/foo/records/0", json={"metadata": {"idx": i}}
        )
        assert response.status_code == 200

    audit_path = tmp_path / AUDIT_RELPATH
    lines = audit_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 5
    for line in lines:
        # Each line must itself be a complete, standalone JSON document
        # (JSONL contract) -- not just that the whole file parses.
        parsed = json.loads(line)
        assert parsed["operation"] == "record.update"


def test_log_update_large_document_is_single_line(tmp_path: Path) -> None:
    logger = AuditLogger(tmp_path)
    big_before = "a" * 200_000
    big_after = "b" * 200_000

    logger.log_update(
        collection="foo",
        record_id="1",
        changes={"document": {"before": big_before, "after": big_after}},
        embedding_stale=True,
    )
    # A second, small entry to confirm the large one didn't leak a stray
    # newline that would corrupt line-based JSONL parsing.
    logger.log_update(
        collection="foo", record_id="2", changes={}, embedding_stale=False
    )

    audit_path = tmp_path / AUDIT_RELPATH
    lines = audit_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    entries = _read_audit_lines(tmp_path)
    assert entries[0]["changes"]["document"]["before"] == big_before
    assert entries[0]["changes"]["document"]["after"] == big_after
    assert entries[1]["id"] == "2"


def test_patch_before_snapshot_reflects_prior_state_not_mutated_state(
    tmp_path: Path, make_app, make_client
) -> None:
    # Guards against a read-then-update race within the handler itself:
    # the "before" value recorded in the audit entry must be the record's
    # state as it existed prior to this specific PATCH, even across
    # multiple sequential updates to the same field.
    _add_records(tmp_path, "foo", 1)
    app = make_app(tmp_path, write=True)
    client = make_client(app)

    values = [{"idx": 1}, {"idx": 2}, {"idx": 3}]
    for v in values:
        resp = client.patch("/api/collections/foo/records/0", json={"metadata": v})
        assert resp.status_code == 200

    entries = _read_audit_lines(tmp_path)
    assert len(entries) == 3
    assert entries[0]["changes"]["metadata"]["before"] == {"idx": 0}
    assert entries[0]["changes"]["metadata"]["after"] == {"idx": 1}
    assert entries[1]["changes"]["metadata"]["before"] == {"idx": 1}
    assert entries[1]["changes"]["metadata"]["after"] == {"idx": 2}
    assert entries[2]["changes"]["metadata"]["before"] == {"idx": 2}
    assert entries[2]["changes"]["metadata"]["after"] == {"idx": 3}
