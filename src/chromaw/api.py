from __future__ import annotations

from fastapi import APIRouter, Request

from chromaw import __version__
from chromaw.models import HealthResponse

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
