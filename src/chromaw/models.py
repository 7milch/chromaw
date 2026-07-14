from __future__ import annotations

from typing import Literal

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

    M2-3 adds ``document`` and ``embedding_mode``. Only ``embedding_mode:
    "keep"`` is supported in the MVP (technical-spec §3.3); "reembed"/
    "manual" are deferred to M3. At least one of ``metadata``/``uri``/
    ``document`` must be given (all ``None`` is rejected with 422, since it
    would be a no-op PATCH).

    Since Chroma has no way to recompute the embedding for a changed
    ``document`` without an embedding function configured, chromaw requires
    the caller to explicitly acknowledge that the embedding will *not* be
    recomputed by passing ``embedding_mode="keep"`` alongside ``document``;
    omitting it is rejected with 422 rather than silently leaving document
    and vector inconsistent.

    chromadb metadata values must be a flat mapping of ``str``/``int``/
    ``float``/``bool`` -- nested dicts/lists and ``None`` values are rejected
    with 422 rather than silently reaching chromadb and failing there.

    chromadb's ``collection.update`` merges the given ``metadata`` into the
    existing metadata rather than replacing it, so an empty ``{}`` would be a
    silent no-op; it is rejected with 422 requiring a non-empty mapping.
    """

    metadata: dict | None = None
    uri: str | None = None
    document: str | None = None
    embedding_mode: Literal["keep"] | None = None

    @model_validator(mode="after")
    def _check_at_least_one_field(self) -> "RecordUpdateRequest":
        if self.metadata is None and self.uri is None and self.document is None:
            raise ValueError(
                "at least one of metadata/uri/document must be given"
            )
        return self

    @model_validator(mode="after")
    def _check_document_requires_embedding_mode(self) -> "RecordUpdateRequest":
        if self.document is not None and self.embedding_mode is None:
            raise ValueError(
                "embedding_mode must be given when document is updated "
                "(technical-spec §3.3): document edits do not recompute the "
                "embedding, so the caller must explicitly acknowledge this by "
                "passing embedding_mode=\"keep\""
            )
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


class DiffRequest(BaseModel):
    """Request body for ``POST /api/diff`` (technical-spec §8, M2-4).

    Generic unified-diff endpoint: the frontend supplies arbitrary
    before/after text (e.g. raw ``document`` text, or ``metadata`` rendered
    via ``JSON.stringify(obj, null, 2)``) plus optional labels, and gets back
    a unified diff string. This has no side effects, so unlike the PATCH
    endpoint it does not require ``require_write_mode``.
    """

    before: str
    after: str
    before_label: str = "before"
    after_label: str = "after"


class DiffResponse(BaseModel):
    """Response body for ``POST /api/diff``."""

    diff: str
