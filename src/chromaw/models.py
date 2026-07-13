from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Response body for ``GET /api/health``."""

    ok: bool
    version: str
    mode: str
    path: str


class CollectionInfo(BaseModel):
    """Summary information for a single ChromaDB collection."""

    id: str
    name: str
    count: int
    metadata: dict | None
    dimension: int | None


class CollectionsResponse(BaseModel):
    """Response body for ``GET /api/collections``."""

    collections: list[CollectionInfo]
