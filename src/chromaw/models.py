from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class HealthResponse(BaseModel):
    """Response body for ``GET /api/health``.

    ``embedding_available`` (M3-3, roadmap "document edit + re-embed") is
    ``True`` iff an explicit ``--embedding-config`` was given at startup
    (``EmbeddingResolver.has_explicit_config``); it drives whether the
    frontend offers a "Re-embed" option for document edits, since without an
    explicit config chromaw can only opportunistically use a *record's own*
    collection's embedding function -- unknown to the frontend ahead of a
    save attempt -- so this flag is a conservative "definitely available"
    signal rather than an exhaustive one.
    """

    ok: bool
    version: str
    mode: str
    path: str
    embedding_available: bool = False


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


class RecordMatchInfo(BaseModel):
    """A single similarity-search result within a collection
    (technical-spec §5.6 4, §8.4).

    Same shape as ``RecordInfo`` plus ``distance`` (chromadb's raw distance
    for the collection's configured space -- lower is more similar; ``None``
    only if ``"distances"`` was omitted from the request's ``include``).
    """

    id: str
    document: str | None
    metadata: dict | None
    uri: str | None
    distance: float | None
    embedding_dimension: int | None
    embedding_preview: list[float] | None


class QueryResponse(BaseModel):
    """Response body for ``POST /api/collections/{name}/query``."""

    matches: list[RecordMatchInfo]


class QueryRequest(BaseModel):
    """Request body for ``POST /api/collections/{name}/query``
    (technical-spec §5.6 4, §8.4).

    Exactly one of ``query_text`` (embedded via the collection's embedding
    function -- technical-spec §5.6 4's priority order) or
    ``query_embedding`` (a precomputed vector, bypassing the embedding
    function entirely) must be given; neither or both is rejected with 422.
    ``where``/``where_document`` may additionally narrow the candidate set,
    same semantics as ``RecordsGetRequest``.
    """

    query_text: str | None = None
    query_embedding: list[float] | None = None
    n_results: int = Field(default=10, ge=1, le=500)
    where: dict | None = None
    where_document: dict | None = None
    include: list[str] = Field(
        default_factory=lambda: ["documents", "metadatas", "uris", "distances"]
    )

    @model_validator(mode="after")
    def _check_exactly_one_query_kind(self) -> "QueryRequest":
        if (self.query_text is None) == (self.query_embedding is None):
            raise ValueError(
                "exactly one of query_text/query_embedding must be given"
            )
        return self

    @model_validator(mode="after")
    def _check_query_embedding_non_empty(self) -> "QueryRequest":
        if self.query_embedding is not None and len(self.query_embedding) == 0:
            raise ValueError("query_embedding must be a non-empty list of floats")
        return self


class RecordUpdateRequest(BaseModel):
    """Request body for ``PATCH /api/collections/{name}/records/{id}``
    (technical-spec §5.4, §8.3).

    M2-3 added ``document`` and ``embedding_mode`` with only
    ``embedding_mode: "keep"`` supported. M3-3 adds ``"reembed"``
    (technical-spec §3.3 roadmap "document edit + re-embed"): ``"manual"``
    (caller supplies a precomputed vector) remains deferred. At least one of
    ``metadata``/``uri``/``document`` must be given (all ``None`` is
    rejected with 422, since it would be a no-op PATCH).

    Since Chroma has no way to recompute the embedding for a changed
    ``document`` without an embedding function configured, chromaw requires
    the caller to explicitly choose whether the embedding should be
    recomputed (``"reembed"``) or intentionally left stale (``"keep"``) by
    passing ``embedding_mode`` alongside ``document``; omitting it is
    rejected with 422 rather than silently leaving document and vector
    inconsistent. A ``"reembed"`` request that can't actually be fulfilled
    (no embedding function available) is rejected by the API layer with 503
    rather than a validation error, since it's a runtime/environment
    condition rather than a malformed request.

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
    embedding_mode: Literal["keep", "reembed"] | None = None

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
                "passing embedding_mode=\"keep\" or \"reembed\""
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


class BulkDeleteRequest(BaseModel):
    """Request body for ``POST /api/collections/{name}/records/bulk-delete``
    (technical-spec §3.2, §6.5, roadmap M4-2).

    Bulk deletion is destructive, so ``confirm`` must be given -- like
    ``CollectionDeleteRequest``, it is checked against the *collection's*
    name (not a per-record id, since there are many), matching the M2-7
    "type the target name" confirmation principle. ``ids`` must be
    non-empty (422 otherwise); duplicates are tolerated (the adapter
    de-duplicates internally, same as ``RecordsGetRequest``'s ``ids``).
    """

    ids: list[str] = Field(min_length=1)
    confirm: str


class BulkDeleteResponse(BaseModel):
    """Response body for ``POST /api/collections/{name}/records/bulk-delete``.

    ``deleted`` lists the ids that actually existed and were removed;
    ``skipped`` lists the requested ids that did not exist in the
    collection (and were therefore left untouched) so the caller can
    reconcile its selection against what really happened.
    """

    deleted: list[str]
    skipped: list[str]


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
