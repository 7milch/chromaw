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
    has_more: bool


class RecordsGetRequest(BaseModel):
    """Request body for ``POST /api/collections/{name}/records/get``.

    Supports ids-based lookup as well as ``where`` (metadata equality/operator
    filter) and ``where_document`` (document content filter, e.g.
    ``{"$contains": "..."}``) filtering (technical-spec §8.3, §5.5 1-3). All
    of ``ids``/``where``/``where_document`` may be combined; chromadb applies
    them as a conjunction (``collection.get`` semantics).
    """

    ids: list[str] | None = None
    where: dict | None = None
    where_document: dict | None = None
    limit: int = Field(default=50, ge=1, le=500)
    offset: int = Field(default=0, ge=0)
    include: list[str] = Field(
        default_factory=lambda: ["documents", "metadatas", "uris"]
    )
