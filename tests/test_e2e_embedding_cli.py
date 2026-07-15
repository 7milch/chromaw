"""End-to-end test: --embedding-config CLI validation (Refs #24).

Confirms the CLI fails fast (exit 1, no server started) when
--embedding-config points at an invalid/missing file, and starts normally
when given a valid local-provider config.
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

pytestmark = pytest.mark.e2e


def _chromaw_bin() -> Path:
    chromaw_bin = Path(sys.executable).with_name("chromaw")
    assert chromaw_bin.exists(), f"chromaw console script not found next to {sys.executable}"
    return chromaw_bin


def test_missing_embedding_config_path_exits_1(tmp_path: Path) -> None:
    chromadb.PersistentClient(path=str(tmp_path))

    proc = subprocess.run(
        [
            str(_chromaw_bin()),
            str(tmp_path),
            "--no-open",
            "--port",
            "0",
            "--embedding-config",
            str(tmp_path / "does-not-exist.json"),
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert proc.returncode == 1
    combined = proc.stdout + proc.stderr
    assert "error" in combined.lower()
    assert "embedding-config" in combined or "embedding_config" in combined


def test_invalid_json_embedding_config_exits_1(tmp_path: Path) -> None:
    chromadb.PersistentClient(path=str(tmp_path))
    config_path = tmp_path / "bad.json"
    config_path.write_text("{not valid json")

    proc = subprocess.run(
        [
            str(_chromaw_bin()),
            str(tmp_path),
            "--no-open",
            "--port",
            "0",
            "--embedding-config",
            str(config_path),
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert proc.returncode == 1
    assert "error" in (proc.stdout + proc.stderr).lower()


def test_hosted_provider_missing_api_key_env_exits_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A hosted-provider config whose api_key_env is unset must fail fast
    at startup (exit 1) rather than only surfacing on the first query."""
    monkeypatch.delenv("CHROMAW_TEST_MISSING_KEY", raising=False)
    chromadb.PersistentClient(path=str(tmp_path))
    config_path = tmp_path / "embedding-config.json"
    config_path.write_text(
        json.dumps(
            {
                "provider": "openai",
                "model": "text-embedding-3-small",
                "api_key_env": "CHROMAW_TEST_MISSING_KEY",
            }
        )
    )

    proc = subprocess.run(
        [
            str(_chromaw_bin()),
            str(tmp_path),
            "--no-open",
            "--port",
            "0",
            "--embedding-config",
            str(config_path),
        ],
        capture_output=True,
        text=True,
        timeout=15,
        env=os.environ.copy(),
    )

    assert proc.returncode == 1
    combined = proc.stdout + proc.stderr
    assert "error" in combined.lower()
    assert "CHROMAW_TEST_MISSING_KEY" in combined


def test_valid_default_provider_embedding_config_starts_server(tmp_path: Path) -> None:
    chromadb.PersistentClient(path=str(tmp_path))
    config_path = tmp_path / "embedding-config.json"
    config_path.write_text(json.dumps({"provider": "default"}))

    proc = subprocess.Popen(
        [
            str(_chromaw_bin()),
            str(tmp_path),
            "--no-open",
            "--port",
            "0",
            "--embedding-config",
            str(config_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    try:
        deadline = time.time() + 15.0
        buf = ""
        found = False
        assert proc.stdout is not None
        while time.time() < deadline:
            line = proc.stdout.readline()
            if not line:
                if proc.poll() is not None:
                    break
                continue
            buf += line
            if re.search(r"chromaw is running at http://[^:]+:(\d+)", line):
                found = True
                break
        assert found, f"server did not start; output so far:\n{buf}"
        assert "provider=default" in buf
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
