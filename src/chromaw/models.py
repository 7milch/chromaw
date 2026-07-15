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


class RecordDeleteRequest(BaseModel):
    """Request body for ``DELETE /api/collections/{name}/records/{id}``
    (technical-spec §3.2, §6.5, roadmap M2-7).

    Deleting a record is destructive, so ``confirm`` must be given and is
    checked against the record id in the URL by the endpoint (client error
    -- ``ConfirmationMismatchError``, mapped to 409 -- if it doesn't match).
    """

    confirm: str


class CollectionDeleteRequest(BaseModel):
    """Request body for ``DELETE /api/collections/{name}`` (technical-spec
    §3.2, §5.2, §6.5, roadmap M2-7).

    ``confirm`` must match the collection name in the URL, checked by the
    endpoint the same way as ``RecordDeleteRequest.confirm``.
    """

    confirm: str


class CollectionUpdateRequest(BaseModel):
    """Request body for ``PATCH /api/collections/{name}`` (technical-spec
    §5.2, §8.2, roadmap M2-7).

    ``name`` renames the collection; ``metadata`` updates its metadata.
    chromadb's ``collection.modify(metadata=...)`` replaces the collection's
    metadata wholesale, so ``ChromaAdapter.update_collection`` merges
    ``metadata`` into the collection's current metadata itself before
    calling ``collection.modify()``, giving the same merge semantics as
    record metadata updates. At least one of
    ``name``/``metadata`` must be given (422 otherwise, matching
    ``RecordUpdateRequest``'s no-op guard).

    Renaming is a destructive operation (technical-spec §3.2, §6.5): when
    ``name`` is given, ``confirm`` must also be given and is checked by the
    endpoint against the collection's *current* name (client error --
    ``ConfirmationMismatchError``, mapped to 409 -- if it doesn't match).
    Renaming is not itself the source of truth for that comparison since the
    request body has no access to the URL path parameter; the endpoint
    performs the actual check. A metadata-only update (no ``name``) is
    non-destructive and does not require ``confirm``.
    """

    name: str | None = None
    metadata: dict | None = None
    confirm: str | None = None

    @model_validator(mode="after")
    def _check_at_least_one_field(self) -> "CollectionUpdateRequest":
        if self.name is None and self.metadata is None:
            raise ValueError("at least one of name/metadata must be given")
        return self

    @model_validator(mode="after")
    def _check_rename_requires_confirm(self) -> "CollectionUpdateRequest":
        if self.name is not None and self.confirm is None:
            raise ValueError(
                "confirm must be given when renaming a collection "
                "(technical-spec §3.2, §6.5): pass the collection's current "
                "name as confirm to acknowledge the rename"
            )
        return self


class DeleteResponse(BaseModel):
    """Response body for the record/collection DELETE endpoints."""

    deleted: bool
    id: str


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
