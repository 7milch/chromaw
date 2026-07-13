from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from chromaw import __version__
from chromaw.errors import RecordNotFoundError
from chromaw.models import (
    CollectionsResponse,
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
    """Update ``metadata`` and/or ``uri`` for a single record
    (technical-spec §5.4, §8.3). M2-2 scope only; ``document`` and
    ``embedding_mode`` are handled in M2-3.
    """

    adapter = request.app.state.adapter
    adapter.update_record(name, record_id, metadata=body.metadata, uri=body.uri)

    records, _, _ = adapter.get_records(
        name,
        ids=[record_id],
        include=("documents", "metadatas", "uris"),
    )
    if not records:
        raise RecordNotFoundError(f"record not found: {record_id!r} in collection {name!r}")
    return records[0]


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
