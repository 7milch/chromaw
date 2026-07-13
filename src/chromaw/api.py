from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request

from chromaw import __version__
from chromaw.models import CollectionsResponse, HealthResponse, RecordsResponse

router = APIRouter(prefix="/api")

_VALID_INCLUDE_VALUES = {"documents", "metadatas", "uris", "embeddings"}


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
    invalid = [item for item in include_values if item not in _VALID_INCLUDE_VALUES]
    if invalid:
        raise HTTPException(
            status_code=422,
            detail=f"invalid include value(s): {', '.join(invalid)}",
        )

    adapter = request.app.state.adapter
    records, total = adapter.get_records(
        name, limit=limit, offset=offset, include=include_values
    )
    return RecordsResponse(records=records, total=total)
