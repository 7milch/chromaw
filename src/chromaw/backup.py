from __future__ import annotations

import shutil
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from chromaw.errors import BackupFailedError

_CHROMAW_DIRNAME = ".chromaw"
_BACKUPS_DIRNAME = "backups"


@dataclass
class BackupManager:
    """Creates a one-time, pre-first-write backup of a ChromaDB persistent
    directory (technical-spec §9.1, roadmap M2-5).

    Before the very first write operation performed by a given chromaw
    process, the whole ``chroma_path`` directory is copied to
    ``{chroma_path}/.chromaw/backups/{YYYYMMDD-HHMMSS}/`` so a user can
    recover pre-edit state. Only a directory copy is implemented for now
    (technical-spec §9.1 explicitly defers "export changed records only" /
    filesystem snapshot integration to later, size-driven work).

    ``ensure_backup()`` is idempotent for the lifetime of this instance: the
    first call performs the copy, every subsequent call is a no-op. This is
    process-local, single-``chromaw``-process state (technical-spec §9.3
    concurrency model), not a durable "has this path ever been backed up"
    marker -- restarting chromaw against the same directory will back it up
    again on its own first write, which is intentional (the directory may
    have changed since the last run).

    Backup failures raise ``BackupFailedError`` and must not be swallowed by
    callers: chromaw is fail-closed here -- if the safety copy can't be
    made, the write it was meant to protect must not proceed either.
    """

    chroma_path: Path
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _done: bool = field(default=False, repr=False)
    _backup_dir: Optional[Path] = field(default=None, repr=False)

    def ensure_backup(self) -> Optional[Path]:
        """Perform the one-time pre-first-write backup if it hasn't run yet.

        Returns the backup directory path (whether just created by this call
        or a prior one), so tests/callers can inspect it. Raises
        ``BackupFailedError`` (without mutating ``_done``, so a later call
        can retry) if the copy fails for any reason.
        """
        with self._lock:
            if self._done:
                return self._backup_dir

            backups_root = self.chroma_path / _CHROMAW_DIRNAME / _BACKUPS_DIRNAME
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            destination = backups_root / timestamp

            try:
                backups_root.mkdir(parents=True, exist_ok=True)
                shutil.copytree(
                    self.chroma_path,
                    destination,
                    ignore=shutil.ignore_patterns(_CHROMAW_DIRNAME),
                )
            except Exception as exc:
                raise BackupFailedError(
                    f"failed to back up {self.chroma_path} to {destination}: {exc}"
                ) from exc

            self._backup_dir = destination
            self._done = True
            return destination

    @property
    def done(self) -> bool:
        """Whether ``ensure_backup`` has successfully completed at least once."""
        return self._done
