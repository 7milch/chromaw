from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from chromaw import __version__
from chromaw.errors import AuditWriteFailedError

_CHROMAW_DIRNAME = ".chromaw"
_AUDIT_FILENAME = "audit.jsonl"


@dataclass
class AuditLogger:
    """Appends one JSON line per write operation to
    ``{chroma_path}/.chromaw/audit.jsonl`` (technical-spec §9.2, roadmap
    M2-6).

    Each entry records enough to reconstruct what changed without needing to
    diff the backup: the collection/record touched, a per-field
    before/after ``changes`` map, and whether the update left the record's
    embedding stale (technical-spec §3.1 "keep" mode). This is a superset of
    the minimal example in technical-spec §9.2 (which only shows
    ``before_hash``/``after_hash``) -- full before/after values are recorded
    instead of hashes so the audit log is directly useful for review/undo
    without needing the original values on hand to verify a hash match.

    Audit logging is fail-closed, matching ``BackupManager``'s posture
    (technical-spec §9.1): if the append can't be written (e.g. permissions,
    disk full, ``.chromaw`` obstructed by a file), ``AuditWriteFailedError``
    is raised rather than silently dropping the record. Callers must treat a
    failed audit write as meaning the operation as a whole did not complete
    successfully -- the spec's "safe-by-default" principle extends to
    "every write is recorded, or the write didn't happen."
    """

    chroma_path: Path
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def log_update(
        self,
        *,
        collection: str,
        record_id: str,
        changes: dict[str, dict[str, Any]],
        embedding_stale: bool,
        embedding_mode: str | None = None,
    ) -> None:
        """Append a ``record.update`` entry for a single PATCH operation.

        ``changes`` maps each updated field name (``metadata``, ``uri``,
        ``document``) to ``{"before": ..., "after": ...}``. Only fields that
        were actually part of the request should be included -- untouched
        fields are omitted entirely rather than recorded as a no-op change.

        ``embedding_mode`` (M3-3) records the caller's requested mode
        (``"keep"``/``"reembed"``) verbatim when a ``document`` update was
        involved, ``None`` otherwise (e.g. a metadata/uri-only PATCH) --
        distinct from ``embedding_stale``, which is the *resulting* stale
        status of the record's vector.
        """

        entry = {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "operation": "record.update",
            "collection": collection,
            "id": record_id,
            "changes": changes,
            "embedding_stale": embedding_stale,
            "embedding_mode": embedding_mode,
            "user_agent": f"chromaw/{__version__}",
        }
        self._append(entry)

    def log_delete_record(
        self,
        *,
        collection: str,
        record_id: str,
        before: dict[str, Any],
    ) -> None:
        """Append a ``record.delete`` entry (roadmap M2-7).

        ``before`` is the full record snapshot (document/metadata/uri) as it
        existed immediately before deletion, so the audit log alone is
        enough to see what was lost without needing the backup on hand.
        """

        entry = {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "operation": "record.delete",
            "collection": collection,
            "id": record_id,
            "before": before,
            "user_agent": f"chromaw/{__version__}",
        }
        self._append(entry)

    def log_delete_collection(
        self,
        *,
        collection: str,
        before: dict[str, Any],
    ) -> None:
        """Append a ``collection.delete`` entry (roadmap M2-7).

        ``before`` is the collection's summary info (id/count/metadata/
        dimension) immediately before deletion.
        """

        entry = {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "operation": "collection.delete",
            "collection": collection,
            "before": before,
            "user_agent": f"chromaw/{__version__}",
        }
        self._append(entry)

    def log_rename_collection(
        self,
        *,
        before_name: str,
        after_name: str,
    ) -> None:
        """Append a ``collection.rename`` entry (roadmap M2-7)."""

        entry = {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "operation": "collection.rename",
            "collection": before_name,
            "changes": {"name": {"before": before_name, "after": after_name}},
            "user_agent": f"chromaw/{__version__}",
        }
        self._append(entry)

    def log_update_collection_metadata(
        self,
        *,
        collection: str,
        before: dict[str, Any] | None,
        after: dict[str, Any] | None,
    ) -> None:
        """Append a ``collection.update`` entry for a metadata-only PATCH
        (roadmap M2-7)."""

        entry = {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "operation": "collection.update",
            "collection": collection,
            "changes": {"metadata": {"before": before, "after": after}},
            "user_agent": f"chromaw/{__version__}",
        }
        self._append(entry)

    def _append(self, entry: dict[str, Any]) -> None:
        audit_dir = self.chroma_path / _CHROMAW_DIRNAME
        audit_path = audit_dir / _AUDIT_FILENAME
        line = json.dumps(entry, ensure_ascii=False)

        with self._lock:
            try:
                audit_dir.mkdir(parents=True, exist_ok=True)
                with audit_path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except Exception as exc:
                raise AuditWriteFailedError(
                    f"failed to append audit entry to {audit_path}: {exc}"
                ) from exc
