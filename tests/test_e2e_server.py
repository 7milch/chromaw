"""End-to-end test: actually launch the chromaw CLI as a subprocess and
verify it starts a real HTTP server that responds, then tear it down.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import chromadb
import pytest

pytestmark = pytest.mark.e2e


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
                # Unauthenticated /api requests are rejected with 401 by the
                # security middleware (technical-spec §10.2) before routing
                # even gets a chance to 404.
                response = exc
                break
            except (urllib.error.URLError, ConnectionError) as exc:
                last_exc = exc
                time.sleep(0.2)

        assert response is not None, f"server never accepted connections: {last_exc}"
        assert response.code == 401
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def _extract_token(index_html: str) -> str:
    match = re.search(r'<meta name="chromaw-token" content="([^"]+)">', index_html)
    assert match, f"no chromaw-token meta tag found in index.html:\n{index_html[:500]}"
    return match.group(1)


def test_real_server_collections_endpoint_with_token(tmp_path: Path) -> None:
    """Full E2E: real subprocess server + real ChromaDB directory containing a
    collection with actual embeddings. Verifies the bearer-token flow end to
    end and that the estimated embedding dimension in the response matches
    the real embedding length written to disk."""
    chroma_client = chromadb.PersistentClient(path=str(tmp_path))
    collection = chroma_client.create_collection("real_memory", metadata={"purpose": "e2e"})
    embedding_dim = 5
    collection.add(
        ids=["a", "b"],
        embeddings=[[0.1] * embedding_dim, [0.2] * embedding_dim],
        metadatas=[{"tag": "x"}, {"tag": "y"}],
    )

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
        base = f"http://127.0.0.1:{port}"

        # Fetch "/" (with retries while uvicorn finishes binding) to obtain
        # the bearer token injected into index.html, exactly as the real
        # frontend does.
        deadline = time.time() + 10.0
        index_html = None
        last_exc: Exception | None = None
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(f"{base}/", timeout=1) as resp:
                    index_html = resp.read().decode("utf-8")
                break
            except (urllib.error.URLError, ConnectionError) as exc:
                last_exc = exc
                time.sleep(0.2)
        assert index_html is not None, f"server never served index.html: {last_exc}"
        token = _extract_token(index_html)

        # Unauthenticated request must be rejected.
        req = urllib.request.Request(f"{base}/api/collections")
        try:
            urllib.request.urlopen(req, timeout=5)
            assert False, "expected 401 without a bearer token"
        except urllib.error.HTTPError as exc:
            assert exc.code == 401

        # Authenticated request must return the real collection data.
        req = urllib.request.Request(
            f"{base}/api/collections", headers={"Authorization": f"Bearer {token}"}
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            assert resp.code == 200
            body = json.loads(resp.read().decode("utf-8"))

        assert len(body["collections"]) == 1
        info = body["collections"][0]
        assert info["name"] == "real_memory"
        assert info["count"] == 2
        assert info["metadata"] == {"purpose": "e2e"}
        assert info["dimension"] == embedding_dim
        assert isinstance(info["id"], str) and info["id"]
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
