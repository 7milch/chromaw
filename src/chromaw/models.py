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


class RecordInfo(BaseModel):
    """A single record within a collection."""

    id: str
    document: str | None
    metadata: dict | None
    uri: str | None
    embedding_dimension: int | None
    embedding_preview: list[float] | None


class RecordsResponse(BaseModel):
    """Response body for ``GET /api/collections/{name}/records``."""

    records: list[RecordInfo]
    total: int
