from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Response body for ``GET /api/health``."""

    ok: bool
    version: str
    mode: str
    path: str
