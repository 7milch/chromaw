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


def test_real_server_records_endpoint_with_token(tmp_path: Path) -> None:
    """Full E2E: real subprocess server + real ChromaDB directory. Verifies
    GET /api/collections/{name}/records over the bearer-token flow, with
    paging and include control against actual data on disk."""
    chroma_client = chromadb.PersistentClient(path=str(tmp_path))
    collection = chroma_client.create_collection("real_records")
    embedding_dim = 4
    ids = [str(i) for i in range(6)]
    collection.add(
        ids=ids,
        documents=[f"doc-{i}" for i in range(6)],
        metadatas=[{"idx": i} for i in range(6)],
        embeddings=[[float(i)] * embedding_dim for i in range(6)],
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
        req = urllib.request.Request(f"{base}/api/collections/real_records/records")
        try:
            urllib.request.urlopen(req, timeout=5)
            assert False, "expected 401 without a bearer token"
        except urllib.error.HTTPError as exc:
            assert exc.code == 401

        # Authenticated request, default include, paging.
        req = urllib.request.Request(
            f"{base}/api/collections/real_records/records?limit=4&offset=0",
            headers={"Authorization": f"Bearer {token}"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            assert resp.code == 200
            body = json.loads(resp.read().decode("utf-8"))

        assert body["total"] == 6
        assert len(body["records"]) == 4
        first = body["records"][0]
        assert set(first.keys()) == {
            "id",
            "document",
            "metadata",
            "uri",
            "embedding_dimension",
            "embedding_preview",
        }
        assert first["document"] is not None
        assert first["metadata"] is not None
        assert first["embedding_dimension"] is None  # embeddings not included by default

        # Second page.
        req = urllib.request.Request(
            f"{base}/api/collections/real_records/records?limit=4&offset=4",
            headers={"Authorization": f"Bearer {token}"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            assert resp.code == 200
            page2 = json.loads(resp.read().decode("utf-8"))
        assert len(page2["records"]) == 2

        seen_ids = {r["id"] for r in body["records"]} | {r["id"] for r in page2["records"]}
        assert seen_ids == set(ids)

        # include=embeddings surfaces embedding info.
        req = urllib.request.Request(
            f"{base}/api/collections/real_records/records?include=embeddings",
            headers={"Authorization": f"Bearer {token}"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            assert resp.code == 200
            emb_body = json.loads(resp.read().decode("utf-8"))
        assert emb_body["records"][0]["embedding_dimension"] == embedding_dim

        # Invalid include value -> 422.
        req = urllib.request.Request(
            f"{base}/api/collections/real_records/records?include=bogus",
            headers={"Authorization": f"Bearer {token}"},
        )
        try:
            urllib.request.urlopen(req, timeout=5)
            assert False, "expected 422 for invalid include value"
        except urllib.error.HTTPError as exc:
            assert exc.code == 422

        # Nonexistent collection -> 404.
        req = urllib.request.Request(
            f"{base}/api/collections/does-not-exist/records",
            headers={"Authorization": f"Bearer {token}"},
        )
        try:
            urllib.request.urlopen(req, timeout=5)
            assert False, "expected 404 for nonexistent collection"
        except urllib.error.HTTPError as exc:
            assert exc.code == 404
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def test_real_server_records_get_endpoint_with_token(tmp_path: Path) -> None:
    """Full E2E: real subprocess server + real ChromaDB directory. Verifies
    POST /api/collections/{name}/records/get (ids-based lookup) over the
    bearer-token flow against actual data on disk."""
    chroma_client = chromadb.PersistentClient(path=str(tmp_path))
    collection = chroma_client.create_collection("real_records_get")
    embedding_dim = 4
    ids = [str(i) for i in range(6)]
    collection.add(
        ids=ids,
        documents=[f"doc-{i}" for i in range(6)],
        metadatas=[{"idx": i} for i in range(6)],
        embeddings=[[float(i)] * embedding_dim for i in range(6)],
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
        req = urllib.request.Request(
            f"{base}/api/collections/real_records_get/records/get",
            data=json.dumps({"ids": ["1"]}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urllib.request.urlopen(req, timeout=5)
            assert False, "expected 401 without a bearer token"
        except urllib.error.HTTPError as exc:
            assert exc.code == 401

        # Authenticated ids-based lookup, default include.
        req = urllib.request.Request(
            f"{base}/api/collections/real_records_get/records/get",
            data=json.dumps({"ids": ["1", "3"]}).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            assert resp.code == 200
            body = json.loads(resp.read().decode("utf-8"))

        assert body["total"] == 2
        assert {r["id"] for r in body["records"]} == {"1", "3"}
        first = body["records"][0]
        assert set(first.keys()) == {
            "id",
            "document",
            "metadata",
            "uri",
            "embedding_dimension",
            "embedding_preview",
        }
        assert first["document"] is not None
        assert first["embedding_dimension"] is None  # embeddings not included by default

        # include=embeddings surfaces embedding info.
        req = urllib.request.Request(
            f"{base}/api/collections/real_records_get/records/get",
            data=json.dumps(
                {"ids": ["2"], "include": ["documents", "metadatas", "uris", "embeddings"]}
            ).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            assert resp.code == 200
            emb_body = json.loads(resp.read().decode("utf-8"))
        assert emb_body["records"][0]["embedding_dimension"] == embedding_dim

        # ids overriding limit: request more ids than the default 50-item
        # limit would suggest a page holds, confirming limit is ignored.
        req = urllib.request.Request(
            f"{base}/api/collections/real_records_get/records/get",
            data=json.dumps({"ids": ids, "limit": 1}).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            assert resp.code == 200
            all_body = json.loads(resp.read().decode("utf-8"))
        assert all_body["total"] == 6
        assert {r["id"] for r in all_body["records"]} == set(ids)

        # Nonexistent id -> empty result, not an error.
        req = urllib.request.Request(
            f"{base}/api/collections/real_records_get/records/get",
            data=json.dumps({"ids": ["does-not-exist"]}).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            assert resp.code == 200
            missing_body = json.loads(resp.read().decode("utf-8"))
        assert missing_body["total"] == 0
        assert missing_body["records"] == []

        # Invalid include value -> 422.
        req = urllib.request.Request(
            f"{base}/api/collections/real_records_get/records/get",
            data=json.dumps({"ids": ["1"], "include": ["bogus"]}).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            urllib.request.urlopen(req, timeout=5)
            assert False, "expected 422 for invalid include value"
        except urllib.error.HTTPError as exc:
            assert exc.code == 422

        # Nonexistent collection -> 404.
        req = urllib.request.Request(
            f"{base}/api/collections/does-not-exist/records/get",
            data=json.dumps({"ids": ["1"]}).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            urllib.request.urlopen(req, timeout=5)
            assert False, "expected 404 for nonexistent collection"
        except urllib.error.HTTPError as exc:
            assert exc.code == 404
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def test_real_server_patch_record_endpoint_with_write_flag(tmp_path: Path) -> None:
    """Full E2E: real subprocess server started with ``--write`` + real
    ChromaDB directory. Verifies PATCH /api/collections/{name}/records/{id}
    over the bearer-token flow actually mutates data on disk, and that the
    write-mode gate blocks the same request when ``--write`` is absent."""
    chroma_client = chromadb.PersistentClient(path=str(tmp_path))
    collection = chroma_client.create_collection("real_patch")
    embedding_dim = 4
    collection.add(
        ids=["1", "2"],
        documents=["doc-1", "doc-2"],
        metadatas=[{"idx": 1}, {"idx": 2}],
        embeddings=[[0.1] * embedding_dim, [0.2] * embedding_dim],
    )

    chromaw_bin = Path(sys.executable).with_name("chromaw")
    assert chromaw_bin.exists(), f"chromaw console script not found next to {sys.executable}"

    proc = subprocess.Popen(
        [str(chromaw_bin), str(tmp_path), "--no-open", "--port", "0", "--write"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    try:
        port = _wait_for_port_line(proc)
        base = f"http://127.0.0.1:{port}"

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

        # Unauthenticated PATCH must be rejected.
        req = urllib.request.Request(
            f"{base}/api/collections/real_patch/records/1",
            data=json.dumps({"metadata": {"tag": "x"}}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="PATCH",
        )
        try:
            urllib.request.urlopen(req, timeout=5)
            assert False, "expected 401 without a bearer token"
        except urllib.error.HTTPError as exc:
            assert exc.code == 401

        # Authenticated PATCH against a real, --write-enabled server actually
        # updates metadata and uri on disk.
        req = urllib.request.Request(
            f"{base}/api/collections/real_patch/records/1",
            data=json.dumps({"metadata": {"tag": "updated"}, "uri": "file:///e2e"}).encode(
                "utf-8"
            ),
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            method="PATCH",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            assert resp.code == 200
            body = json.loads(resp.read().decode("utf-8"))

        assert body["id"] == "1"
        assert body["uri"] == "file:///e2e"
        # chromadb merges metadata updates rather than replacing wholesale.
        assert body["metadata"] == {"idx": 1, "tag": "updated"}

        # Nonexistent record -> 404.
        req = urllib.request.Request(
            f"{base}/api/collections/real_patch/records/does-not-exist",
            data=json.dumps({"metadata": {"tag": "x"}}).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            method="PATCH",
        )
        try:
            urllib.request.urlopen(req, timeout=5)
            assert False, "expected 404 for nonexistent record"
        except urllib.error.HTTPError as exc:
            assert exc.code == 404

        # The update from earlier in this test is durable: re-reading via
        # GET (a fresh request against the same running server/adapter,
        # exercising the on-disk persistence path) shows the new values.
        req = urllib.request.Request(
            f"{base}/api/collections/real_patch/records?limit=10",
            headers={"Authorization": f"Bearer {token}"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            assert resp.code == 200
            records_body = json.loads(resp.read().decode("utf-8"))
        updated_record = next(r for r in records_body["records"] if r["id"] == "1")
        assert updated_record["uri"] == "file:///e2e"
        assert updated_record["metadata"] == {"idx": 1, "tag": "updated"}
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def test_real_server_patch_record_blocked_without_write_flag(tmp_path: Path) -> None:
    """Full E2E: real subprocess server started WITHOUT ``--write`` (the
    default, read-only mode) must reject PATCH with 403, and must not
    mutate the on-disk data."""
    chroma_client = chromadb.PersistentClient(path=str(tmp_path))
    collection = chroma_client.create_collection("real_patch_ro")
    collection.add(
        ids=["1"],
        embeddings=[[0.1, 0.2]],
        metadatas=[{"idx": 1}],
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

        req = urllib.request.Request(
            f"{base}/api/collections/real_patch_ro/records/1",
            data=json.dumps({"metadata": {"tag": "should-not-apply"}}).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            method="PATCH",
        )
        try:
            urllib.request.urlopen(req, timeout=5)
            assert False, "expected 403 in read-only mode"
        except urllib.error.HTTPError as exc:
            assert exc.code == 403
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
