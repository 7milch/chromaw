from __future__ import annotations

import difflib

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from chromaw import __version__
from chromaw.errors import (
    CollectionNotFoundError,
    ConfirmationMismatchError,
    RecordNotFoundError,
)
from chromaw.models import (
    CollectionDeleteRequest,
    CollectionInfo,
    CollectionsResponse,
    CollectionUpdateRequest,
    DeleteResponse,
    DiffRequest,
    DiffResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    RecordDeleteRequest,
    RecordInfo,
    RecordsGetRequest,
    RecordsResponse,
    RecordUpdateRequest,
)

router = APIRouter(prefix="/api")

_VALID_INCLUDE_VALUES = {"documents", "metadatas", "uris", "embeddings"}
# Query results additionally support "distances" (chromadb's collection.query()
# accepts it directly as an include value, unlike collection.get()).
_VALID_QUERY_INCLUDE_VALUES = _VALID_INCLUDE_VALUES | {"distances"}


def require_write_mode(request: Request) -> None:
    """FastAPI dependency guarding write endpoints (technical-spec §3.2).

    chromaw is safe-by-default: it starts in read-only mode unless the
    operator passes ``--write``. Any write endpoint (record/collection
    mutations, added from M2-2 onward) should declare
    ``dependencies=[Depends(require_write_mode)]`` so a read-only server
    rejects the request with a 403 before any mutation is attempted,
    instead of relying on each handler to remember the check.
    """

    if request.app.state.mode != "write":
        raise HTTPException(
            status_code=403,
            detail=(
                "chromaw is running in read-only mode; restart with --write "
                "to enable edits."
            ),
        )


def _validate_include(
    include_values: tuple[str, ...], valid: set[str] = _VALID_INCLUDE_VALUES
) -> None:
    """Raise a 422 if any of ``include_values`` isn't a recognized field.

    Shared by the paged ``GET .../records`` endpoint, the ids-based
    ``POST .../records/get`` endpoint, and ``POST .../query`` (which passes
    ``valid=_VALID_QUERY_INCLUDE_VALUES`` to additionally allow
    ``"distances"``) so the whitelist stays consistent.
    """
    invalid = [item for item in include_values if item not in valid]
    if invalid:
        raise HTTPException(
            status_code=422,
            detail=f"invalid include value(s): {', '.join(invalid)}",
        )


@router.get("/health", response_model=HealthResponse)
def get_health(request: Request) -> HealthResponse:
    """Report server liveness plus the mode/path it was started with."""

    return HealthResponse(
        ok=True,
        version=__version__,
        mode=request.app.state.mode,
        path=str(request.app.state.path),
    )


@router.get("/collections", response_model=CollectionsResponse)
def get_collections(request: Request) -> CollectionsResponse:
    """List all collections in the connected ChromaDB directory."""

    adapter = request.app.state.adapter
    return CollectionsResponse(collections=adapter.list_collections())


@router.get("/collections/{name}/records", response_model=RecordsResponse)
def get_records(
    name: str,
    request: Request,
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    include: str = Query(default="documents,metadatas,uris"),
) -> RecordsResponse:
    """List a page of records for a collection (technical-spec §5.3, §8.3)."""

    include_values = tuple(item.strip() for item in include.split(",") if item.strip())
    _validate_include(include_values)

    adapter = request.app.state.adapter
    records, total, has_more = adapter.get_records(
        name, limit=limit, offset=offset, include=include_values
    )
    return RecordsResponse(records=records, total=total, has_more=has_more)


@router.patch(
    "/collections/{name}/records/{record_id}",
    response_model=RecordInfo,
    dependencies=[Depends(require_write_mode)],
)
def patch_record(
    name: str,
    record_id: str,
    request: Request,
    body: RecordUpdateRequest,
) -> RecordInfo:
    """Update ``metadata``, ``uri``, and/or ``document`` for a single record
    (technical-spec §3.3, §5.4, §8.3).

    ``document`` updates always carry ``embedding_mode="keep"`` (enforced by
    ``RecordUpdateRequest`` validation), so they are always forwarded with
    ``mark_stale=True``: the embedding is left untouched and the record's
    metadata is flagged ``chromaw_embedding_status: "stale"``.

    Before the actual mutation, this is the write-endpoint hook for the
    pre-first-write backup (technical-spec §9.1, roadmap M2-5):
    ``backup_manager.ensure_backup()`` is a no-op after its first successful
    call, and raises ``BackupFailedError`` (mapped to a 500 by
    ``create_app``) if the backup couldn't be made -- fail-closed, so
    ``adapter.update_record`` below is never reached in that case.

    After a successful mutation, the change is appended to the audit log
    (technical-spec §9.2, roadmap M2-6) via ``request.app.state.audit_logger``
    -- only fields actually present in the request body are recorded, each
    as a before/after pair. ``AuditWriteFailedError`` propagates (mapped to
    a 500 by ``create_app``) rather than being swallowed: per the audit
    module's fail-closed contract, a write whose audit entry couldn't be
    persisted must not be reported to the client as having succeeded.
    """

    backup_manager = request.app.state.backup_manager
    if backup_manager is not None:
        backup_manager.ensure_backup()

    adapter = request.app.state.adapter

    before_records, _, _ = adapter.get_records(
        name,
        ids=[record_id],
        include=("documents", "metadatas", "uris"),
    )
    if not before_records:
        raise RecordNotFoundError(f"record not found: {record_id!r} in collection {name!r}")
    before = before_records[0]

    mark_stale = body.document is not None
    adapter.update_record(
        name,
        record_id,
        metadata=body.metadata,
        uri=body.uri,
        document=body.document,
        mark_stale=mark_stale,
    )

    records, _, _ = adapter.get_records(
        name,
        ids=[record_id],
        include=("documents", "metadatas", "uris"),
    )
    if not records:
        raise RecordNotFoundError(f"record not found: {record_id!r} in collection {name!r}")
    after = records[0]

    audit_logger = request.app.state.audit_logger
    if audit_logger is not None:
        changes: dict[str, dict[str, object]] = {}
        if body.metadata is not None:
            changes["metadata"] = {"before": before.metadata, "after": after.metadata}
        if body.uri is not None:
            changes["uri"] = {"before": before.uri, "after": after.uri}
        if body.document is not None:
            changes["document"] = {"before": before.document, "after": after.document}
        audit_logger.log_update(
            collection=name,
            record_id=record_id,
            changes=changes,
            embedding_stale=mark_stale,
        )

    return after


def _find_collection(request: Request, name: str) -> CollectionInfo:
    """Look up a collection's summary info by name, or raise
    ``CollectionNotFoundError`` (mapped to 404).

    Used by the DELETE/PATCH collection endpoints below to get a consistent
    404 for an unknown collection *before* checking ``confirm`` -- so a
    confirm mismatch against a nonexistent collection is reported as "not
    found" rather than "confirmation mismatch".
    """
    adapter = request.app.state.adapter
    for collection in adapter.list_collections():
        if collection.name == name:
            return collection
    raise CollectionNotFoundError(f"collection not found: {name}")


@router.delete(
    "/collections/{name}/records/{record_id}",
    response_model=DeleteResponse,
    dependencies=[Depends(require_write_mode)],
)
def delete_record(
    name: str,
    record_id: str,
    request: Request,
    body: RecordDeleteRequest,
) -> DeleteResponse:
    """Delete a single record (technical-spec §3.2, §5.3, §6.5, roadmap
    M2-7).

    ``body.confirm`` must equal ``record_id`` exactly (the frontend's
    confirmation modal requires the user to type the record id); a mismatch
    raises ``ConfirmationMismatchError`` (409) before anything is deleted.
    The record's existence is checked first, so an unknown ``record_id``
    yields 404 rather than 409 even if ``confirm`` also happens to be wrong.

    Follows the same backup-then-mutate-then-audit sequence as
    ``patch_record``; the full pre-deletion record (document/metadata/uri)
    is captured and recorded as the audit entry's ``before`` snapshot, since
    once deleted it can't be reconstructed from the collection itself.
    """

    backup_manager = request.app.state.backup_manager
    if backup_manager is not None:
        backup_manager.ensure_backup()

    adapter = request.app.state.adapter

    before_records, _, _ = adapter.get_records(
        name,
        ids=[record_id],
        include=("documents", "metadatas", "uris"),
    )
    if not before_records:
        raise RecordNotFoundError(f"record not found: {record_id!r} in collection {name!r}")
    before = before_records[0]

    if body.confirm != record_id:
        raise ConfirmationMismatchError(
            f"confirm {body.confirm!r} does not match record id {record_id!r}"
        )

    adapter.delete_record(name, record_id)

    audit_logger = request.app.state.audit_logger
    if audit_logger is not None:
        audit_logger.log_delete_record(
            collection=name,
            record_id=record_id,
            before=before.model_dump(),
        )

    return DeleteResponse(deleted=True, id=record_id)


@router.delete(
    "/collections/{name}",
    response_model=DeleteResponse,
    dependencies=[Depends(require_write_mode)],
)
def delete_collection(
    name: str,
    request: Request,
    body: CollectionDeleteRequest,
) -> DeleteResponse:
    """Delete an entire collection (technical-spec §3.2, §5.2, §6.5, roadmap
    M2-7).

    ``body.confirm`` must equal the collection's name exactly. The
    collection's existence is checked first (via ``_find_collection``, 404
    if missing) before the ``confirm`` comparison, so an unknown collection
    always yields 404 rather than 409.
    """

    backup_manager = request.app.state.backup_manager
    if backup_manager is not None:
        backup_manager.ensure_backup()

    before = _find_collection(request, name)

    if body.confirm != name:
        raise ConfirmationMismatchError(
            f"confirm {body.confirm!r} does not match collection name {name!r}"
        )

    adapter = request.app.state.adapter
    adapter.delete_collection(name)

    audit_logger = request.app.state.audit_logger
    if audit_logger is not None:
        audit_logger.log_delete_collection(collection=name, before=before.model_dump())

    return DeleteResponse(deleted=True, id=name)


@router.patch(
    "/collections/{name}",
    response_model=CollectionInfo,
    dependencies=[Depends(require_write_mode)],
)
def patch_collection(
    name: str,
    request: Request,
    body: CollectionUpdateRequest,
) -> CollectionInfo:
    """Rename and/or update the metadata of a collection (technical-spec
    §5.2, §8.2, roadmap M2-7).

    Renaming (``body.name`` given) is destructive (technical-spec §3.2,
    §6.5): ``body.confirm`` must equal the collection's *current* name
    (checked here against the URL's ``name``, since ``CollectionUpdateRequest``
    already rejects ``name`` without any ``confirm`` at the schema level). A
    metadata-only update (no ``name``) is non-destructive and proceeds
    without a confirm check.
    """

    backup_manager = request.app.state.backup_manager
    if backup_manager is not None:
        backup_manager.ensure_backup()

    before = _find_collection(request, name)

    if body.name is not None and body.confirm != name:
        raise ConfirmationMismatchError(
            f"confirm {body.confirm!r} does not match collection name {name!r}"
        )

    adapter = request.app.state.adapter
    after = adapter.update_collection(name, new_name=body.name, metadata=body.metadata)

    audit_logger = request.app.state.audit_logger
    if audit_logger is not None:
        if body.name is not None:
            audit_logger.log_rename_collection(before_name=name, after_name=body.name)
        if body.metadata is not None:
            audit_logger.log_update_collection_metadata(
                collection=after.name, before=before.metadata, after=after.metadata
            )

    return after


@router.post("/diff", response_model=DiffResponse)
def post_diff(body: DiffRequest) -> DiffResponse:
    """Generate a unified diff between two arbitrary texts (M2-4).

    Used by the frontend's edit-confirmation screens to preview ``document``
    and ``metadata`` (JSON-serialized) changes before a PATCH is sent. Has no
    side effects, so it is available in read-only mode (no
    ``require_write_mode`` dependency) and only needs the bearer-token
    auth already applied to all ``/api`` routes.
    """

    diff_lines = difflib.unified_diff(
        body.before.splitlines(keepends=True),
        body.after.splitlines(keepends=True),
        fromfile=body.before_label,
        tofile=body.after_label,
    )
    return DiffResponse(diff="".join(diff_lines))


@router.post("/collections/{name}/records/get", response_model=RecordsResponse)
def post_records_get(
    name: str,
    request: Request,
    body: RecordsGetRequest,
) -> RecordsResponse:
    """Look up records by id, ``where``, and/or ``where_document``
    (technical-spec §5.4, §5.5, §6.2, §8.3).
    """

    include_values = tuple(body.include)
    _validate_include(include_values)

    adapter = request.app.state.adapter
    records, total, has_more = adapter.get_records(
        name,
        ids=body.ids,
        where=body.where,
        where_document=body.where_document,
        limit=body.limit,
        offset=body.offset,
        include=include_values,
    )
    return RecordsResponse(records=records, total=total, has_more=has_more)


@router.post("/collections/{name}/query", response_model=QueryResponse)
def post_query(
    name: str,
    request: Request,
    body: QueryRequest,
) -> QueryResponse:
    """Run a similarity search against a collection (technical-spec §5.6 4,
    §8.4, roadmap M3-1).

    A read operation like ``post_records_get`` above -- available in
    read-only mode (no ``require_write_mode`` dependency). ``QueryRequest``
    guarantees exactly one of ``query_text``/``query_embedding`` is given;
    the adapter classifies a failure while embedding ``query_text`` as
    ``EmbeddingFunctionUnavailableError`` (503) rather than a client error,
    since the request itself is well-formed.
    """

    include_values = tuple(body.include)
    _validate_include(include_values, valid=_VALID_QUERY_INCLUDE_VALUES)

    adapter = request.app.state.adapter
    matches = adapter.query_records(
        name,
        query_text=body.query_text,
        query_embedding=body.query_embedding,
        n_results=body.n_results,
        where=body.where,
        where_document=body.where_document,
        include=include_values,
    )
    return QueryResponse(matches=matches)
