"""Tests for the ``.chromaw/lock`` multi-instance guard (technical-spec
§9.3, roadmap M2-8).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import chromadb
import pytest

from chromaw.errors import LockHeldError
from chromaw.lock import ChromawLock


def _dead_pid() -> int:
    """Return a pid that (almost certainly) does not currently exist.

    Spawns and immediately waits on a short-lived subprocess, then returns
    its pid -- guaranteed dead the moment ``wait()`` returns, and unlikely
    to be recycled to another live process within the test run.
    """
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def test_acquire_creates_lock_file_with_expected_contents(tmp_path: Path) -> None:
    lock = ChromawLock(tmp_path, host="127.0.0.1", port=1234)
    lock.acquire()
    try:
        assert lock.lock_path.is_file()
        data = json.loads(lock.lock_path.read_text())
        assert data["pid"] == os.getpid()
        assert data["host"] == "127.0.0.1"
        assert data["port"] == 1234
        assert "created_at" in data
    finally:
        lock.release()


def test_second_acquire_raises_lock_held_error(tmp_path: Path) -> None:
    first = ChromawLock(tmp_path)
    first.acquire()
    try:
        second = ChromawLock(tmp_path)
        with pytest.raises(LockHeldError) as excinfo:
            second.acquire()
        assert str(os.getpid()) in str(excinfo.value)
    finally:
        first.release()


def test_release_allows_reacquire(tmp_path: Path) -> None:
    first = ChromawLock(tmp_path)
    first.acquire()
    first.release()

    assert not first.lock_path.exists()

    second = ChromawLock(tmp_path)
    second.acquire()
    try:
        assert second.lock_path.is_file()
    finally:
        second.release()


def test_release_without_acquire_is_a_no_op(tmp_path: Path) -> None:
    lock = ChromawLock(tmp_path)
    lock.release()  # must not raise
    assert not lock.lock_path.exists()


def test_release_does_not_delete_lock_it_does_not_own(tmp_path: Path) -> None:
    lock = ChromawLock(tmp_path)
    lock.acquire()

    # Simulate another process having reclaimed the (now-stale) lock after
    # this instance already thinks it holds it.
    other_payload = json.dumps({"pid": 999999999, "created_at": "x"})
    lock.lock_path.write_text(other_payload)

    lock.release()

    # The file (owned by "someone else") must survive.
    assert lock.lock_path.exists()


def test_release_when_lock_file_already_removed_is_a_no_op(tmp_path: Path) -> None:
    lock = ChromawLock(tmp_path)
    lock.acquire()

    # Simulate the lock file having vanished out from under us (e.g. someone
    # manually cleaned up .chromaw/).
    lock.lock_path.unlink()

    lock.release()  # must not raise
    assert not lock.lock_path.exists()


def test_stale_lock_with_dead_pid_is_reclaimed(tmp_path: Path) -> None:
    lock_dir = tmp_path / ".chromaw"
    lock_dir.mkdir()
    stale_pid = _dead_pid()
    (lock_dir / "lock").write_text(
        json.dumps({"pid": stale_pid, "created_at": "2020-01-01T00:00:00Z"})
    )

    lock = ChromawLock(tmp_path)
    lock.acquire()  # must not raise
    try:
        data = json.loads(lock.lock_path.read_text())
        assert data["pid"] == os.getpid()
    finally:
        lock.release()


def test_stale_lock_reclaim_race_only_one_instance_wins_the_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Simulate two ChromawLock instances concurrently reclaiming the same
    stale lock. This is sequential (no real threads), but forces the
    interleaving: instance A's ``_reclaim_stale_lock`` performs the rename
    first, then instance B's is invoked against a lock file that has
    already vanished -- exercising the ``os.rename`` -> FileNotFoundError
    -> retry path, and confirming exactly one instance ends up holding the
    lock.
    """
    lock_dir = tmp_path / ".chromaw"
    lock_dir.mkdir()
    stale_pid = _dead_pid()
    stale_info = {"pid": stale_pid, "created_at": "2020-01-01T00:00:00Z"}
    (lock_dir / "lock").write_text(json.dumps(stale_info))

    lock_a = ChromawLock(tmp_path)
    lock_b = ChromawLock(tmp_path)

    real_reclaim_a = lock_a._reclaim_stale_lock
    real_reclaim_b = lock_b._reclaim_stale_lock

    reclaim_calls: list[str] = []

    def reclaim_a(observed_info):
        reclaim_calls.append("a")
        # Before A performs its own rename, let B race in and reclaim the
        # lock first -- simulating B "winning" the TOCTOU race.
        if len(reclaim_calls) == 1:
            real_reclaim_b(observed_info)
        return real_reclaim_a(observed_info)

    monkeypatch.setattr(lock_a, "_reclaim_stale_lock", reclaim_a)

    # A observes the stale lock, but by the time it tries to rename it away,
    # B has already done so (FileNotFoundError -> A's reclaim is a no-op and
    # it retries the acquire loop, successfully creating its own lock).
    lock_a.acquire()
    try:
        assert lock_a.lock_path.is_file()
        data = json.loads(lock_a.lock_path.read_text())
        assert data["pid"] == os.getpid()
        # The original stale-lock content must not linger, and no
        # leftover "lock.reclaim-*" temp file should remain.
        leftover = [p for p in lock_dir.iterdir() if p.name != "lock"]
        assert leftover == []
    finally:
        lock_a.release()


def test_live_pid_lock_blocks_acquire(tmp_path: Path) -> None:
    lock_dir = tmp_path / ".chromaw"
    lock_dir.mkdir()
    # Our own pid is trivially "alive".
    (lock_dir / "lock").write_text(
        json.dumps({"pid": os.getpid(), "created_at": "2020-01-01T00:00:00Z"})
    )

    lock = ChromawLock(tmp_path)
    with pytest.raises(LockHeldError):
        lock.acquire()


def test_corrupt_lock_file_is_treated_as_stale(tmp_path: Path) -> None:
    lock_dir = tmp_path / ".chromaw"
    lock_dir.mkdir()
    (lock_dir / "lock").write_text("not valid json{{{")

    lock = ChromawLock(tmp_path)
    lock.acquire()  # must not raise
    try:
        data = json.loads(lock.lock_path.read_text())
        assert data["pid"] == os.getpid()
    finally:
        lock.release()


def test_lock_missing_pid_field_is_treated_as_stale(tmp_path: Path) -> None:
    lock_dir = tmp_path / ".chromaw"
    lock_dir.mkdir()
    (lock_dir / "lock").write_text(json.dumps({"created_at": "2020-01-01T00:00:00Z"}))

    lock = ChromawLock(tmp_path)
    lock.acquire()
    lock.release()


def test_context_manager_acquires_and_releases(tmp_path: Path) -> None:
    lock = ChromawLock(tmp_path)
    with lock:
        assert lock.lock_path.is_file()
    assert not lock.lock_path.exists()


def test_context_manager_releases_on_exception(tmp_path: Path) -> None:
    lock = ChromawLock(tmp_path)
    with pytest.raises(RuntimeError):
        with lock:
            assert lock.lock_path.is_file()
            raise RuntimeError("boom")
    assert not lock.lock_path.exists()


def test_context_manager_second_lock_raises(tmp_path: Path) -> None:
    with ChromawLock(tmp_path):
        with pytest.raises(LockHeldError):
            with ChromawLock(tmp_path):
                pass


# --- CLI-level: real second process is rejected -------------------------


def _wait_for_port_line(proc: subprocess.Popen, timeout: float = 15.0) -> int:
    deadline = time.time() + timeout
    buf = ""
    assert proc.stdout is not None
    while time.time() < deadline:
        line = proc.stdout.readline()
        if not line:
            if proc.poll() is not None:
                raise RuntimeError(f"process exited early: {buf}")
            continue
        buf += line
        match = re.search(r"chromaw is running at http://[^:]+:(\d+)", line)
        if match:
            return int(match.group(1))
    raise TimeoutError(f"server did not report a running URL in time. output so far:\n{buf}")


@pytest.mark.e2e
def test_second_cli_instance_exits_with_error(tmp_path: Path) -> None:
    chromadb.PersistentClient(path=str(tmp_path))

    chromaw_bin = Path(sys.executable).with_name("chromaw")
    assert chromaw_bin.exists(), f"chromaw console script not found next to {sys.executable}"

    first = subprocess.Popen(
        [str(chromaw_bin), str(tmp_path), "--no-open", "--port", "0"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    try:
        _wait_for_port_line(first)

        second = subprocess.run(
            [str(chromaw_bin), str(tmp_path), "--no-open", "--port", "0"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=15,
        )
        assert second.returncode != 0
        assert "already running" in second.stdout.lower() or "lock" in second.stdout.lower()
    finally:
        first.terminate()
        try:
            first.wait(timeout=5)
        except subprocess.TimeoutExpired:
            first.kill()
            first.wait(timeout=5)
