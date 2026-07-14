from __future__ import annotations

import difflib

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from chromaw import __version__
from chromaw.errors import RecordNotFoundError
from chromaw.models import (
    CollectionsResponse,
    DiffRequest,
    DiffResponse,
    HealthResponse,
    RecordInfo,
    RecordsGetRequest,
    RecordsResponse,
    RecordUpdateRequest,
)

router = APIRouter(prefix="/api")

_VALID_INCLUDE_VALUES = {"documents", "metadatas", "uris", "embeddings"}


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


def _validate_include(include_values: tuple[str, ...]) -> None:
    """Raise a 422 if any of ``include_values`` isn't a recognized field.

    Shared by both the paged ``GET .../records`` endpoint and the ids-based
    ``POST .../records/get`` endpoint so the whitelist stays consistent.
    """
    invalid = [item for item in include_values if item not in _VALID_INCLUDE_VALUES]
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
    """

    backup_manager = request.app.state.backup_manager
    if backup_manager is not None:
        backup_manager.ensure_backup()

    adapter = request.app.state.adapter
    adapter.update_record(
        name,
        record_id,
        metadata=body.metadata,
        uri=body.uri,
        document=body.document,
        mark_stale=body.document is not None,
    )

    records, _, _ = adapter.get_records(
        name,
        ids=[record_id],
        include=("documents", "metadatas", "uris"),
    )
    if not records:
        raise RecordNotFoundError(f"record not found: {record_id!r} in collection {name!r}")
    return records[0]


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
