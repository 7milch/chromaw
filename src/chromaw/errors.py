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


class RecordNotFoundError(ChromawError):
    """Raised when the requested record id does not exist in the collection."""


class BackupFailedError(ChromawError):
    """Raised when the pre-first-write backup (technical-spec §9.1) could not
    be created. Callers must treat this as fail-closed: the write the
    backup was meant to protect must not proceed."""


class AuditWriteFailedError(ChromawError):
    """Raised when an audit log entry (technical-spec §9.2) could not be
    appended to ``.chromaw/audit.jsonl``. chromaw treats audit logging as
    fail-closed: if a write operation cannot be recorded, the caller must
    not treat the operation as successfully completed and must surface an
    error rather than silently skip the audit trail."""
