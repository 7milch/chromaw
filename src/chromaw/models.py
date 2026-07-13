from __future__ import annotations

from pydantic import BaseModel, Field


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
    """Response body for ``GET /api/collections/{name}/records`` and
    ``POST /api/collections/{name}/records/get``.
    """

    records: list[RecordInfo]
    total: int


class RecordsGetRequest(BaseModel):
    """Request body for ``POST /api/collections/{name}/records/get``.

    Minimal ids-based lookup (technical-spec §8.3 later half). ``where`` /
    ``where_document`` filtering is out of scope here and will be added in
    M1-4.
    """

    ids: list[str] | None = None
    limit: int = Field(default=50, ge=1, le=500)
    offset: int = Field(default=0, ge=0)
    include: list[str] = Field(
        default_factory=lambda: ["documents", "metadatas", "uris"]
    )
