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


class ConfirmationMismatchError(ChromawError):
    """Raised when a destructive request's ``confirm`` field does not match
    the target name (technical-spec §3.2, §6.5): record delete, collection
    delete, and collection rename all require the caller to type the exact
    target name/id before the operation is carried out."""


class InvalidCollectionNameError(ChromawError):
    """Raised when a collection rename's ``new_name`` is rejected by
    chromadb as an invalid collection name (e.g. empty, too long, disallowed
    characters)."""


class CollectionAlreadyExistsError(ChromawError):
    """Raised when a collection rename's ``new_name`` collides with an
    existing collection in the same ChromaDB directory."""


class LockHeldError(ChromawError):
    """Raised when ``.chromaw/lock`` (technical-spec §9.3) is already held
    by another live chromaw process against the same ChromaDB directory."""


class AuditWriteFailedError(ChromawError):
    """Raised when an audit log entry (technical-spec §9.2) could not be
    appended to ``.chromaw/audit.jsonl``. chromaw treats audit logging as
    fail-closed: if a write operation cannot be recorded, the caller must
    not treat the operation as successfully completed and must surface an
    error rather than silently skip the audit trail."""


class EmbeddingFunctionUnavailableError(ChromawError):
    """Raised when ``query_records`` is given ``query_text`` but the
    collection's embedding function is unavailable or fails while embedding
    it (technical-spec §5.6 4, §8.4). This is not the client's fault in the
    "malformed request" sense (unlike ``InvalidFilterError``/
    ``InvalidQueryEmbeddingError``), so it is mapped to 503 rather than 422:
    the request is well-formed but the server currently cannot fulfil a
    text-based query (e.g. no embedding function configured, or the default
    embedding function's model could not be loaded)."""


class InvalidQueryEmbeddingError(ChromawError):
    """Raised when ``query_records`` is given a ``query_embedding`` that
    chromadb rejects (e.g. wrong dimension for the collection, wrong shape).
    Unlike ``EmbeddingFunctionUnavailableError`` this is a genuine client
    error -- the caller supplied a bad vector -- so it is mapped to 422."""


class EmbeddingConfigError(ChromawError):
    """Raised when ``--embedding-config`` (technical-spec §5.6 4) points at a
    file that cannot be read, is not valid JSON, has an unsupported
    ``provider``, or is missing an ``api_key_env`` (or the environment
    variable it names) for a provider that requires an API key. Unlike
    ``EmbeddingFunctionUnavailableError`` (which covers *runtime* embedding
    failures against a query), this is a configuration problem the caller
    can fix by editing the config file or their environment, and is raised
    eagerly at CLI startup so it fails fast rather than surfacing on the
    first text search."""
