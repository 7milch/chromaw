"""End-to-end test: actually launch the chromaw CLI as a subprocess and
verify it starts a real HTTP server that responds, then tear it down.
"""
from __future__ import annotations

import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import chromadb


def _wait_for_port_line(proc: subprocess.Popen, timeout: float = 15.0) -> int:
    """Read stdout lines until we see the 'chromaw is running at' line, return port."""
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


def test_real_server_starts_and_responds_over_http(tmp_path: Path) -> None:
    chromadb.PersistentClient(path=str(tmp_path))

    chromaw_bin = Path(sys.executable).with_name("chromaw")
    assert chromaw_bin.exists(), f"chromaw console script not found next to {sys.executable}"

    proc = subprocess.Popen(
        [str(chromaw_bin), str(tmp_path), "--no-open", "--port", "0"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    try:
        port = _wait_for_port_line(proc)

        # Give uvicorn a brief moment to finish binding after the CLI prints
        # the URL (the print happens just before uvicorn.run()).
        deadline = time.time() + 10.0
        last_exc: Exception | None = None
        response = None
        while time.time() < deadline:
            try:
                response = urllib.request.urlopen(f"http://127.0.0.1:{port}/api/does-not-exist", timeout=1)
                break
            except urllib.error.HTTPError as exc:
                # Server responded (even with an error status) -> it's up.
                response = exc
                break
            except (urllib.error.URLError, ConnectionError) as exc:
                last_exc = exc
                time.sleep(0.2)

        assert response is not None, f"server never accepted connections: {last_exc}"
        assert response.code == 404
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
