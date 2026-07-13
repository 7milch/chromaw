from __future__ import annotations

from fastapi import APIRouter, Request

from chromaw import __version__
from chromaw.models import CollectionsResponse, HealthResponse

router = APIRouter(prefix="/api")


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
