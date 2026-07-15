from __future__ import annotations

import json
import os
import socket
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from chromaw.errors import LockHeldError

_CHROMAW_DIRNAME = ".chromaw"
_LOCK_FILENAME = "lock"


def _pid_alive_windows(pid: int) -> bool:
    """Windows implementation of :func:`_pid_alive`.

    ``os.kill(pid, 0)`` is unsafe on Windows: signal 0 is not special-cased
    there and ``os.kill`` actually calls ``TerminateProcess``, which can kill
    the target process outright. Instead we use the Win32 API directly:
    ``OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION)`` + ``GetExitCodeProcess``,
    checking for the ``STILL_ACTIVE`` (259) sentinel.
    """
    import ctypes
    from ctypes import wintypes

    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    STILL_ACTIVE = 259
    ERROR_ACCESS_DENIED = 5

    # use_last_error=True is required for ctypes.get_last_error() below to
    # return an accurate value: without it, ctypes does not preserve the
    # thread-local last-error code across the ctypes call boundary, so
    # ERROR_ACCESS_DENIED (5) detection would be unreliable.
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        # ERROR_INVALID_PARAMETER (and similar) means no such process --
        # ERROR_ACCESS_DENIED means the process exists but we can't query it,
        # which we treat as "alive" (same policy as the POSIX PermissionError
        # branch below).
        err = ctypes.get_last_error()
        return err == ERROR_ACCESS_DENIED
    try:
        exit_code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return False
        return exit_code.value == STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


def _pid_alive(pid: int) -> bool:
    """Return whether ``pid`` refers to a currently-running process.

    On POSIX, uses signal 0 (no-op signal) which only checks
    existence/permission without actually sending a signal. A
    ``PermissionError`` means the process exists but is owned by another
    user -- still alive from our point of view. Any other value means it's
    dead or the pid is bogus.

    On Windows, ``os.kill(pid, 0)`` is not safe (see
    :func:`_pid_alive_windows`), so we use the Win32 API instead.
    """
    if pid <= 0:
        return False
    if sys.platform == "win32":
        return _pid_alive_windows(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


@dataclass
class ChromawLock:
    """Guards against multiple ``chromaw`` processes operating on the same
    ChromaDB persistent directory at once (technical-spec §9.3, roadmap
    M2-8).

    On ``acquire()``, atomically creates ``{chroma_path}/.chromaw/lock``
    already containing its JSON payload describing the owning process
    (``pid``, ``created_at``, ``host``, ``port``). This is done by first
    writing the payload to a unique temporary file
    (``lock.tmp-{pid}-{uuid4().hex}``) in the same directory and then
    hard-linking it into place with ``os.link()``: ``os.link`` fails with
    ``FileExistsError`` if the destination already exists, so two processes
    racing to create the lock can't both succeed, and -- unlike
    ``O_CREAT | O_EXCL`` followed by a separate write -- there is no window
    where a concurrent reader can observe an empty lock file and misjudge it
    as corrupt/stale. (This relies on the ``.chromaw`` directory living on a
    filesystem that supports hard links, which is assumed to hold for the
    local/network filesystems chromaw targets.) The temporary file is
    unlinked again once the hard link is in place. If a lock file already
    exists:

    - if the lock's ``hostname`` differs from ours, the lock is always
      treated as held/alive (``LockHeldError``), regardless of whether the
      recorded pid looks alive locally -- pids are only meaningful on the
      host that created them, so a pid check across hosts is meaningless
      (and would falsely reclaim a live remote lock). This matters when
      ``chroma_path`` lives on a network filesystem shared between hosts;
      note that liveness checks in that scenario are still inherently
      racy/best-effort (e.g. clock skew, stale NFS caches), and this lock
      is not a substitute for a real distributed lock service;
    - otherwise, if the recorded ``pid`` is no longer alive, the lock is
      treated as stale (left behind by a crashed/killed process) and is
      removed and re-acquired automatically;
    - otherwise, if the recorded ``pid`` is alive, ``LockHeldError`` is
      raised with the holder's pid and creation time in the message so the
      user knows which process to stop;
    - if the lock file exists but is not valid JSON (or is missing the
      ``pid`` field), it's treated the same as a stale lock and reclaimed --
      a corrupt lock file must not be able to permanently wedge the
      directory.

    Reclaiming a stale lock is done via an atomic rename (to a unique
    temporary name) followed by a content check and unlink, rather than an
    in-place unlink, to avoid a TOCTOU race where two processes both decide
    the same lock is stale and both proceed to treat themselves as the sole
    reclaimer (see ``acquire()`` for details).

    This lock is taken for both read-only and write chromaw sessions: the
    concern (two chromaw UIs on the same directory racing each other, or a
    write session running unaware of another live session) applies
    regardless of write mode, and technical-spec §9.3 does not carve out an
    exception for read-only.

    ``release()`` only removes the lock file if it still contains *this*
    process's pid, so a process never deletes a lock it doesn't own (e.g.
    after a stale lock was already reclaimed by someone else). Supports use
    as a context manager.
    """

    chroma_path: Path
    host: Optional[str] = None
    port: Optional[int] = None
    _acquired: bool = field(default=False, repr=False, init=False)

    @property
    def lock_path(self) -> Path:
        return self.chroma_path / _CHROMAW_DIRNAME / _LOCK_FILENAME

    def _read_lock_info(self, *, _retried: bool = False) -> Optional[dict[str, Any]]:
        try:
            raw = self.lock_path.read_text()
        except OSError:
            return None
        if not raw and not _retried:
            # Defensive retry: even though acquire() now hard-links a fully
            # written temp file into place (so a genuinely empty lock file
            # should no longer be observable), guard against any other
            # source of a transient empty read (e.g. a filesystem/caching
            # quirk) by giving a concurrent writer a brief moment to finish
            # before treating this as corrupt/stale.
            time.sleep(0.1)
            return self._read_lock_info(_retried=True)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if not isinstance(data, dict) or not isinstance(data.get("pid"), int):
            return None
        return data

    def _is_stale(self, info: Optional[dict[str, Any]]) -> bool:
        if info is None:
            return True
        lock_hostname = info.get("hostname")
        if lock_hostname is not None and lock_hostname != socket.gethostname():
            # Pids are not comparable across hosts -- a lock created
            # elsewhere is always treated as live.
            return False
        return not _pid_alive(info["pid"])

    def _reclaim_stale_lock(self, observed_info: Optional[dict[str, Any]]) -> None:
        """Remove a lock file previously judged stale, guarding against a
        TOCTOU race where two processes both observe the same stale lock and
        both try to reclaim it.

        A plain ``unlink()`` here would be unsafe: if process A unlinks the
        stale lock and then re-creates it (winning the O_CREAT|O_EXCL race),
        and process B is still mid-way through its own "it's stale, remove
        it" logic, B could end up unlinking A's brand-new, live lock instead
        of the stale one it actually observed.

        Instead we atomically rename the lock file aside to a unique
        temporary name first. ``os.rename`` on the same filesystem is atomic,
        so at most one process can successfully claim the file this way; a
        second, concurrent reclaimer sees ``FileNotFoundError`` and simply
        retries the whole acquire loop (by returning normally here -- the
        caller's ``continue`` will re-attempt ``O_CREAT | O_EXCL`` and either
        succeed or observe whatever is there now). After winning the rename,
        we double-check the moved-aside file's contents still match what we
        originally judged stale (paranoia against an even weirder
        interleaving) before deleting it.
        """
        tmp_path = self.lock_path.with_name(
            f"{_LOCK_FILENAME}.reclaim-{os.getpid()}-{uuid.uuid4().hex}"
        )
        try:
            os.rename(self.lock_path, tmp_path)
        except FileNotFoundError:
            # Someone else already reclaimed (or released) it first.
            return

        try:
            try:
                raw = tmp_path.read_text()
            except OSError:
                raw = None
            try:
                data: Optional[dict[str, Any]] = json.loads(raw) if raw is not None else None
            except json.JSONDecodeError:
                data = None
            if observed_info is not None and data != observed_info:
                # Contents changed out from under us between our staleness
                # check and the rename -- don't trust our earlier judgment;
                # let the caller retry from scratch.
                return
        finally:
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass

    def acquire(self) -> None:
        """Acquire the lock, raising ``LockHeldError`` if another live
        chromaw process already holds it. Reclaims stale/corrupt locks
        automatically."""
        lock_dir = self.lock_path.parent
        lock_dir.mkdir(parents=True, exist_ok=True)

        payload = json.dumps(
            {
                "pid": os.getpid(),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "host": self.host,
                "port": self.port,
                "hostname": socket.gethostname(),
            }
        )

        while True:
            tmp_path = self.lock_path.with_name(
                f"{_LOCK_FILENAME}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
            )
            fd = os.open(tmp_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            try:
                with os.fdopen(fd, "w") as fh:
                    fh.write(payload)
                try:
                    os.link(tmp_path, self.lock_path)
                except FileExistsError:
                    existing = self._read_lock_info()
                    if self._is_stale(existing):
                        self._reclaim_stale_lock(existing)
                        continue
                    pid = existing["pid"] if existing else "unknown"
                    created_at = (
                        existing.get("created_at", "unknown") if existing else "unknown"
                    )
                    raise LockHeldError(
                        f"chromaw is already running against {self.chroma_path} "
                        f"(pid={pid}, started at {created_at}). "
                        "Stop that process first, or remove "
                        f"{self.lock_path} if you're sure it's stale."
                    )
                else:
                    self._acquired = True
                    return
            finally:
                try:
                    tmp_path.unlink()
                except FileNotFoundError:
                    pass

    def release(self) -> None:
        """Remove the lock file, but only if it still records this
        process's pid (never delete a lock we don't own)."""
        if not self._acquired:
            return
        info = self._read_lock_info()
        if info is not None and info.get("pid") == os.getpid():
            try:
                self.lock_path.unlink()
            except FileNotFoundError:
                pass
        self._acquired = False

    def __enter__(self) -> "ChromawLock":
        self.acquire()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.release()
