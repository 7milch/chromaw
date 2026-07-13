"""Shared pytest fixtures for the chromaw test suite.

Consolidates the setup that was duplicated across test_server.py,
test_security.py, and test_static.py: creating a temporary ChromaDB
directory, opening it as a ChromaAdapter, building a chromaw FastAPI app
from it, and constructing a TestClient with the Host header / bearer token
the security middleware expects.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import chromadb
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from chromaw.chroma_adapter import ChromaAdapter
from chromaw.server import create_app

TOKEN = "test-token"
HOST = "127.0.0.1"
PORT = 8000


@pytest.fixture
def make_adapter() -> Callable[[Path], ChromaAdapter]:
    """Factory fixture: initialize an empty ChromaDB dir at ``path`` and open
    it as a ChromaAdapter."""

    def _make(path: Path) -> ChromaAdapter:
        chromadb.PersistentClient(path=str(path))
        return ChromaAdapter.open(path)

    return _make


@pytest.fixture
def adapter(tmp_path: Path, make_adapter: Callable[[Path], ChromaAdapter]) -> ChromaAdapter:
    """A ready-to-use ChromaAdapter backed by an empty temporary ChromaDB dir."""
    return make_adapter(tmp_path)


@pytest.fixture
def make_app(
    make_adapter: Callable[[Path], ChromaAdapter],
) -> Callable[..., FastAPI]:
    """Factory fixture: build a chromaw app from a fresh ChromaDB dir at
    ``path``, with test-friendly defaults for token/host/port."""

    def _make(
        path: Path,
        *,
        write: bool = False,
        token: str | None = TOKEN,
        host: str = HOST,
        port: int = PORT,
        on_startup: Any = None,
    ) -> FastAPI:
        adapter = make_adapter(path)
        return create_app(
            adapter, write=write, token=token, host=host, port=port, on_startup=on_startup
        )

    return _make


@pytest.fixture
def make_client() -> Callable[..., TestClient]:
    """Factory fixture: a TestClient whose Host header matches HOST:PORT by
    default (satisfying the security middleware's allow-list) and that
    optionally carries a bearer token. Pass ``token=None`` to omit the
    Authorization header (e.g. to exercise the security middleware itself),
    or ``base_url`` to test a mismatched Host header."""

    def _make(
        app: FastAPI,
        *,
        token: str | None = TOKEN,
        base_url: str | None = None,
    ) -> TestClient:
        client = TestClient(app, base_url=base_url or f"http://{HOST}:{PORT}")
        if token is not None:
            client.headers["Authorization"] = f"Bearer {token}"
        return client

    return _make
