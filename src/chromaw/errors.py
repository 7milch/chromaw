from __future__ import annotations


class ChromawError(Exception):
    """Base class for all chromaw-specific errors."""


class ChromaPathNotFoundError(ChromawError):
    """Raised when the given ChromaDB path does not exist and --create was not given."""


class ChromaEmptyDirectoryError(ChromawError):
    """Raised when the given path is an empty directory and --create was not given."""


class ChromaInvalidDirectoryError(ChromawError):
    """Raised when the given path exists but is not a valid/readable ChromaDB directory.

    This covers both "missing chroma.sqlite3" (not a ChromaDB directory) and
    "chroma.sqlite3 present but unreadable" (corrupted, or created by an
    incompatible chromadb version) cases.
    """


class CollectionNotFoundError(ChromawError):
    """Raised when the requested collection does not exist in the ChromaDB directory."""


class InvalidFilterError(ChromawError):
    """Raised when ``where`` / ``where_document`` filter given to ``get_records``
    is rejected by chromadb (e.g. malformed operator, unknown field syntax)."""
