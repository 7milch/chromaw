from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb

from chromaw.errors import (
    ChromaEmptyDirectoryError,
    ChromaInvalidDirectoryError,
    ChromaPathNotFoundError,
)

_SQLITE_FILENAME = "chroma.sqlite3"


@dataclass
class ChromaAdapter:
    """Thin wrapper around ``chromadb.PersistentClient``.

    All interaction with a ChromaDB persistent directory must go through this
    class (and, transitively, the official chromadb client). Internal Chroma
    files (``chroma.sqlite3``, segment directories, ...) must never be
    written to directly; the only internal file this class inspects is the
    presence of ``chroma.sqlite3``, used purely as a read-only heuristic to
    tell an existing ChromaDB directory apart from an unrelated one.
    """

    path: Path
    _client: Any

    @classmethod
    def open(cls, path: Path, create: bool = False) -> "ChromaAdapter":
        """Open (and optionally create) a ChromaDB persistent directory.

        Raises:
            ChromaPathNotFoundError: path does not exist and create is False.
            ChromaEmptyDirectoryError: path is an empty directory and create is False.
            ChromaInvalidDirectoryError: path exists but does not look like (or
                cannot be opened as) a ChromaDB persistent directory.
        """
        path = Path(path)

        if not path.exists():
            if not create:
                raise ChromaPathNotFoundError(
                    f"path does not exist: {path} (pass --create to create it)"
                )
            path.mkdir(parents=True, exist_ok=True)
        elif not path.is_dir():
            raise ChromaInvalidDirectoryError(f"path is not a directory: {path}")

        sqlite_path = path / _SQLITE_FILENAME
        is_empty = not any(path.iterdir())

        if is_empty:
            if not create:
                raise ChromaEmptyDirectoryError(
                    f"directory is empty: {path} "
                    "(pass --create to initialize a new ChromaDB store here)"
                )
        elif not sqlite_path.exists():
            # Non-empty directory without chroma.sqlite3: refuse to touch it.
            # Handing this to PersistentClient would silently create a new
            # sqlite3 file inside a directory we don't recognize as Chroma's.
            raise ChromaInvalidDirectoryError(
                f"directory does not look like a ChromaDB persistent directory "
                f"(missing {_SQLITE_FILENAME}): {path}"
            )

        try:
            client = chromadb.PersistentClient(path=str(path))
        except Exception as exc:
            chromadb_version = getattr(chromadb, "__version__", "unknown")
            raise ChromaInvalidDirectoryError(
                f"failed to open ChromaDB directory (possibly corrupted, or created by "
                f"an incompatible chromadb version; this chromaw uses chromadb=="
                f"{chromadb_version}): {path} ({exc})"
            ) from exc

        return cls(path=path, _client=client)

    def list_collections(self) -> list[str]:
        """Return the names of all collections in this ChromaDB directory."""
        collections = self._client.list_collections()
        return [c.name if hasattr(c, "name") else str(c) for c in collections]
