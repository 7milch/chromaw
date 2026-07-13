from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


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


class RecordUpdateRequest(BaseModel):
    """Request body for ``PATCH /api/collections/{name}/records/{id}``
    (technical-spec §5.4, §8.3).

    M2-2 accepts only ``metadata`` and ``uri``; ``document``/``embedding_mode``
    are deferred to M2-3. At least one of ``metadata``/``uri`` must be given
    (both ``None`` is rejected with 422, since it would be a no-op PATCH).

    chromadb metadata values must be a flat mapping of ``str``/``int``/
    ``float``/``bool`` -- nested dicts/lists and ``None`` values are rejected
    with 422 rather than silently reaching chromadb and failing there.

    chromadb's ``collection.update`` merges the given ``metadata`` into the
    existing metadata rather than replacing it, so an empty ``{}`` would be a
    silent no-op; it is rejected with 422 requiring a non-empty mapping.
    """

    metadata: dict | None = None
    uri: str | None = None

    @model_validator(mode="after")
    def _check_at_least_one_field(self) -> "RecordUpdateRequest":
        if self.metadata is None and self.uri is None:
            raise ValueError("at least one of metadata/uri must be given")
        return self

    @model_validator(mode="after")
    def _check_metadata_is_flat_scalar_dict(self) -> "RecordUpdateRequest":
        if self.metadata is None:
            return self
        if not self.metadata:
            raise ValueError("metadata must be a non-empty mapping if given")
        for key, value in self.metadata.items():
            if isinstance(value, bool):
                continue
            if isinstance(value, (str, int, float)):
                continue
            raise ValueError(
                f"metadata value for key {key!r} must be a str, int, float, "
                f"or bool (got {type(value).__name__})"
            )
        return self
