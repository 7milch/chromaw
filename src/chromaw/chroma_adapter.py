from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chromadb
from chromadb.errors import ChromaError

from chromaw.embedding import EmbeddingResolver
from chromaw.errors import (
    ChromaEmptyDirectoryError,
    ChromaInvalidDirectoryError,
    ChromaPathNotFoundError,
    CollectionAlreadyExistsError,
    CollectionNotFoundError,
    EmbeddingFunctionUnavailableError,
    InvalidCollectionNameError,
    InvalidFilterError,
    InvalidQueryEmbeddingError,
    RecordNotFoundError,
)
from chromaw.models import CollectionInfo, RecordInfo, RecordMatchInfo

_SQLITE_FILENAME = "chroma.sqlite3"
_DEFAULT_MAX_BATCH_SIZE = 1000


@dataclass
class ImportEntry:
    """A single parsed JSONL import row (roadmap M4-3, technical-spec §8).

    Produced by the API layer's line-by-line parsing of the uploaded JSONL
    body (``POST .../import``) -- everything id/JSON-shape related has
    already been validated by the time an ``ImportEntry`` reaches
    ``ChromaAdapter.import_records``; ``line`` (1-indexed, matching the
    uploaded file) is carried along purely so adapter-level failures (e.g. a
    ``collection.add()``/``collection.upsert()`` call rejected by chromadb)
    can still be reported back to the caller as a line-numbered skip, the
    same shape as a parse-level skip.
    """

    line: int
    id: str
    document: str | None = None
    metadata: dict[str, Any] | None = None
    uri: str | None = None
    embedding: list[float] | None = None


@dataclass
class ChromaAdapter:
    """Thin wrapper around ``chromadb.PersistentClient``.

    All interaction with a ChromaDB persistent directory must go through this
    class (and, transitively, the official chromadb client). Internal Chroma
    files (``chroma.sqlite3``, segment directories, ...) must never be
    written to directly; the only internal file this class inspects is the
    presence of ``chroma.sqlite3``, used purely as a read-only heuristic to
    tell an existing ChromaDB directory apart from an unrelated one.
    """

    path: Path
    _client: Any
    embedding_resolver: EmbeddingResolver = field(default_factory=EmbeddingResolver)

    @classmethod
    def open(cls, path: Path, create: bool = False) -> "ChromaAdapter":
        """Open (and optionally create) a ChromaDB persistent directory.

        ``embedding_resolver`` (technical-spec §5.6 4, M3-2) is left at its
        default (no explicit ``--embedding-config``) here; callers that want
        one set ``adapter.embedding_resolver`` afterwards -- it has nothing
        to do with opening the ChromaDB directory itself.

        Raises:
            ChromaPathNotFoundError: path does not exist and create is False.
            ChromaEmptyDirectoryError: path is an empty directory and create is False.
            ChromaInvalidDirectoryError: path exists but does not look like (or
                cannot be opened as) a ChromaDB persistent directory.
        """
        path = Path(path)

        if not path.exists():
            if not create:
                raise ChromaPathNotFoundError(
                    f"path does not exist: {path} (pass --create to create it)"
                )
            path.mkdir(parents=True, exist_ok=True)
        elif not path.is_dir():
            raise ChromaInvalidDirectoryError(f"path is not a directory: {path}")

        sqlite_path = path / _SQLITE_FILENAME
        is_empty = not any(path.iterdir())

        if is_empty:
            if not create:
                raise ChromaEmptyDirectoryError(
                    f"directory is empty: {path} "
                    "(pass --create to initialize a new ChromaDB store here)"
                )
        elif not sqlite_path.exists():
            # Non-empty directory without chroma.sqlite3: refuse to touch it.
            # Handing this to PersistentClient would silently create a new
            # sqlite3 file inside a directory we don't recognize as Chroma's.
            raise ChromaInvalidDirectoryError(
                f"directory does not look like a ChromaDB persistent directory "
                f"(missing {_SQLITE_FILENAME}): {path}"
            )

        try:
            client = chromadb.PersistentClient(path=str(path))
        except Exception as exc:
            chromadb_version = getattr(chromadb, "__version__", "unknown")
            raise ChromaInvalidDirectoryError(
                f"failed to open ChromaDB directory (possibly corrupted, or created by "
                f"an incompatible chromadb version; this chromaw uses chromadb=="
                f"{chromadb_version}): {path} ({exc})"
            ) from exc

        return cls(path=path, _client=client)

    def list_collections(self) -> list[CollectionInfo]:
        """Return summary info for all collections in this ChromaDB directory.

        ``dimension`` is estimated by fetching a single record's embedding
        from the collection (``get(limit=1, include=["embeddings"])``) and
        reading its length; collections with zero records yield ``None``.
        """
        collections = self._client.list_collections()
        result: list[CollectionInfo] = []
        for collection in collections:
            count = collection.count()
            dimension: int | None = None
            if count > 0:
                sample = collection.get(limit=1, include=["embeddings"])
                embeddings = sample.get("embeddings")
                if embeddings is not None and len(embeddings) > 0:
                    dimension = len(embeddings[0])
            result.append(
                CollectionInfo(
                    id=str(collection.id),
                    name=collection.name,
                    count=count,
                    metadata=collection.metadata,
                    dimension=dimension,
                )
            )
        return result

    def get_records(
        self,
        name: str,
        *,
        ids: list[str] | None = None,
        where: dict | None = None,
        where_document: dict | None = None,
        limit: int = 50,
        offset: int = 0,
        include: tuple[str, ...] = ("documents", "metadatas", "uris"),
    ) -> tuple[list[RecordInfo], int, bool]:
        """Return a page of records for the collection named ``name``.

        When ``ids`` is given (and neither ``where`` nor ``where_document``
        is), results are restricted to those ids (``collection.get(ids=...)``)
        and paging (``limit``/``offset``) is not applied by Chroma to the ids
        list itself -- all matching records are returned in one page, and
        ``has_more`` is always ``False`` in that case. Returns a
        ``(records, total, has_more)`` tuple where ``total`` is the number of
        records returned (i.e. matching the given ``ids``) rather than
        ``collection.count()`` in that case, since "the collection's full
        record count" is not a meaningful notion of total for an ids-scoped
        lookup; without ``ids``, ``total`` remains ``collection.count()`` as
        before.

        ``where`` / ``where_document`` (technical-spec §5.5 1-3, §8.3) are
        passed straight through to ``collection.get()`` and may be combined
        with ``ids`` (chromadb ANDs all given filters together). chromadb
        has no way to cheaply count "records matching this filter" (its
        ``count()`` ignores filters), so when either is given ``total`` uses
        the same approximation as the ``ids`` case: the number of records
        actually returned by this call (i.e. ``offset + len(records)``,
        not the true total match count) rather than ``collection.count()``.
        Callers driving "next page" purely off whether the current page is
        full-sized still work correctly; only the displayed grand total is
        approximate while filtering.

        To determine ``has_more`` while filtering (where ``total`` is only
        an approximation and can't answer "is there another page"), this
        method internally requests ``limit + 1`` records from chromadb:
        ``has_more`` is ``True`` iff that internal fetch returned more than
        ``limit`` records, and the extra record (if any) is trimmed before
        returning so callers still see at most ``limit`` records. Without
        filtering, ``has_more`` is derived directly from ``collection.count()``
        instead: ``offset + len(records) < total``.

        Any ``ChromaError`` (or ``ValueError``/``TypeError``) raised by
        chromadb while a ``where``/``where_document`` filter is given is
        treated as a client error and reraised as ``InvalidFilterError``
        (422), rather than being distinguished from genuine internal/server
        failures. This is a deliberate simplification: chromadb does not
        give this adapter a reliable way to tell "malformed filter" apart
        from other errors that happen to surface while a filter is active,
        so all such errors are attributed to the filter. A malformed filter
        will correctly produce a 422; an unrelated internal error that
        happens to occur only when filtering is present would incorrectly
        also surface as a 422 instead of a 5xx, which is the accepted
        trade-off.

        ``embedding_dimension``/``embedding_preview`` (first 8 values) are
        only populated when ``"embeddings"`` is present in ``include``;
        callers that don't need them can omit it to avoid the extra cost of
        fetching embedding vectors.

        Raises:
            CollectionNotFoundError: no collection named ``name`` exists.
            InvalidFilterError: ``where``/``where_document`` is malformed and
                rejected by chromadb.
        """
        try:
            collection = self._client.get_collection(name)
        except Exception as exc:
            raise CollectionNotFoundError(f"collection not found: {name}") from exc

        if ids is not None and len(ids) == 0:
            # chromadb raises a ValueError for collection.get(ids=[]); an
            # empty ids list unambiguously matches zero records, so return
            # an empty page without calling into chromadb at all.
            return [], 0, False

        if ids is not None:
            # chromadb raises DuplicateIDError if the same id appears more
            # than once in the ids list; de-duplicate while preserving
            # order so each requested id still yields exactly one record.
            seen: set[str] = set()
            deduped_ids: list[str] = []
            for record_id in ids:
                if record_id not in seen:
                    seen.add(record_id)
                    deduped_ids.append(record_id)
            ids = deduped_ids

        is_filtered = where is not None or where_document is not None
        get_include = [item for item in include if item != "ids"]
        get_kwargs: dict[str, Any] = {"include": get_include}
        if ids is not None:
            get_kwargs["ids"] = ids
        if where is not None:
            get_kwargs["where"] = where
        if where_document is not None:
            get_kwargs["where_document"] = where_document
        # Chroma applies limit/offset regardless of whether ids/where/
        # where_document are also given; only the ids-only (no where/
        # where_document) case above intentionally omits them, to preserve
        # existing "ids ignores paging" behavior. When filtering, fetch one
        # extra record (limit + 1) so has_more can be derived from whether
        # the extra record came back, without an additional round trip; it
        # is trimmed off below before records are returned.
        if ids is None or is_filtered:
            get_kwargs["limit"] = limit + 1 if is_filtered else limit
            get_kwargs["offset"] = offset

        try:
            result = collection.get(**get_kwargs)
        except (ValueError, TypeError, ChromaError) as exc:
            if where is not None or where_document is not None:
                # A malformed where/where_document filter is a client error,
                # not an internal fallback situation; surface it as such
                # rather than retrying the "uris" fallback below (which
                # wouldn't help and would just obscure the real cause).
                raise InvalidFilterError(str(exc)) from exc
            # Defensive fallback: some chromadb versions reject "uris" (or
            # other include values) in collection.get(), raising either a
            # plain ValueError/TypeError or a chromadb-specific error (e.g.
            # InvalidArgumentError, which subclasses ChromaError). Retry
            # without "uris" rather than failing the whole request; if the
            # fallback also fails, surface the original error instead of the
            # fallback's (usually less informative) one.
            fallback_include = [item for item in get_include if item != "uris"]
            if fallback_include == get_include:
                # "uris" wasn't even requested; the retry can't help.
                raise
            try:
                result = collection.get(**{**get_kwargs, "include": fallback_include})
            except Exception:
                raise exc from None

        result_ids = result.get("ids") or []
        if is_filtered:
            # Fetched limit + 1 above; more than limit rows coming back
            # means there is at least one further page.
            has_more = len(result_ids) > limit
            if has_more:
                result_ids = result_ids[:limit]
            result_count = len(result_ids)
            # Paging (limit/offset) is applied by chromadb here, so "total"
            # can only be approximated by how far paging has progressed
            # (using the trimmed, limit-sized count); see docstring above.
            total = offset + result_count
        elif ids is not None:
            # ids-only: paging isn't applied, so the result count *is* the
            # total (matching however many of the given ids exist), and all
            # matches are always returned in this single call.
            result_count = len(result_ids)
            total = result_count
            has_more = False
        else:
            result_count = len(result_ids)
            total = collection.count()
            has_more = offset + result_count < total

        ids = result_ids
        # Slice documents/metadatas/uris/embeddings to match the (possibly
        # trimmed) ids list above so indices stay aligned.
        documents = result.get("documents")
        if documents is not None:
            documents = documents[: len(ids)]
        metadatas = result.get("metadatas")
        if metadatas is not None:
            metadatas = metadatas[: len(ids)]
        uris = result.get("uris")
        if uris is not None:
            uris = uris[: len(ids)]
        embeddings = result.get("embeddings") if "embeddings" in include else None
        if embeddings is not None:
            embeddings = embeddings[: len(ids)]

        records: list[RecordInfo] = []
        for i, record_id in enumerate(ids):
            document = documents[i] if documents is not None else None
            metadata = metadatas[i] if metadatas is not None else None
            uri = uris[i] if uris is not None else None

            embedding_dimension: int | None = None
            embedding_preview: list[float] | None = None
            if embeddings is not None and embeddings[i] is not None:
                embedding = list(embeddings[i])
                embedding_dimension = len(embedding)
                embedding_preview = embedding[:8]

            records.append(
                RecordInfo(
                    id=str(record_id),
                    document=document,
                    metadata=metadata,
                    uri=uri,
                    embedding_dimension=embedding_dimension,
                    embedding_preview=embedding_preview,
                )
            )

        return records, total, has_more

    def query_records(
        self,
        name: str,
        *,
        query_text: str | None = None,
        query_embedding: list[float] | None = None,
        n_results: int = 10,
        where: dict | None = None,
        where_document: dict | None = None,
        include: tuple[str, ...] = ("documents", "metadatas", "uris", "distances"),
    ) -> list[RecordMatchInfo]:
        """Run a similarity search against the collection named ``name``
        (technical-spec §5.6 4, §8.4).

        Exactly one of ``query_text``/``query_embedding`` must be given by
        the caller (enforced by ``QueryRequest`` at the API layer):
        ``query_text`` is embedded via the collection's configured embedding
        function (technical-spec §5.6 4's priority order --
        ``--embedding-config`` if given, else chromadb's default embedding
        function); ``query_embedding`` is used as-is, bypassing the
        embedding function entirely.

        Unlike ``get_records``, chromadb's ``collection.query()`` supports
        ``"distances"`` directly as an ``include`` value (there is no
        "uris"-style fallback dance needed here).

        Any ``ChromaError``/``ValueError``/``TypeError`` raised by
        ``collection.query()`` is reclassified the same way ``get_records``
        does for filters, with two additions specific to querying:

        - a malformed ``where``/``where_document`` filter (checked first,
          same priority as ``get_records``) raises ``InvalidFilterError``
          (422);
        - otherwise, for a ``query_text`` query, the failure is attributed to
          the embedding function (unavailable, failed to load/run) and
          raises ``EmbeddingFunctionUnavailableError`` (503) -- the request
          itself is well-formed, the server just currently can't fulfil a
          text-based query;
        - otherwise (a ``query_embedding`` query), the failure is attributed
          to the caller-supplied vector (e.g. wrong dimension for the
          collection) and raises ``InvalidQueryEmbeddingError`` (422).

        ``embedding_dimension``/``embedding_preview`` (first 8 values) are
        only populated when ``"embeddings"`` is present in ``include``, same
        as ``get_records``.

        Raises:
            CollectionNotFoundError: no collection named ``name`` exists.
            InvalidFilterError: ``where``/``where_document`` is malformed.
            EmbeddingFunctionUnavailableError: ``query_text`` was given but
                the collection's embedding function is unavailable or failed.
            InvalidQueryEmbeddingError: ``query_embedding`` was given but
                rejected by chromadb (e.g. wrong dimension).
        """
        try:
            collection = self._client.get_collection(name)
        except Exception as exc:
            raise CollectionNotFoundError(f"collection not found: {name}") from exc

        query_kwargs: dict[str, Any] = {
            "n_results": n_results,
            "include": list(include),
        }
        if query_text is not None and self.embedding_resolver.has_explicit_config:
            # Tier 1 (technical-spec §5.6 4): an explicit --embedding-config
            # always wins over the collection's own embedding function, so
            # embed the query text ourselves and query by vector instead of
            # letting collection.query() use its own configured EF.
            embedded = self.embedding_resolver.embed_query(query_text)
            query_kwargs["query_embeddings"] = [embedded]
        elif query_text is not None:
            # Tiers 2/3: no explicit config, so defer to chromadb -- it uses
            # the collection's own embedding function if configured, else
            # its default embedding function, else raises (caught below).
            query_kwargs["query_texts"] = [query_text]
        else:
            query_kwargs["query_embeddings"] = [query_embedding]
        if where is not None:
            query_kwargs["where"] = where
        if where_document is not None:
            query_kwargs["where_document"] = where_document

        try:
            result = collection.query(**query_kwargs)
        except (ValueError, TypeError, ChromaError) as exc:
            if where is not None or where_document is not None:
                raise InvalidFilterError(str(exc)) from exc
            if query_text is not None:
                raise EmbeddingFunctionUnavailableError(str(exc)) from exc
            raise InvalidQueryEmbeddingError(str(exc)) from exc

        result_ids = (result.get("ids") or [[]])[0]
        documents = result.get("documents")
        documents = documents[0] if documents else None
        metadatas = result.get("metadatas")
        metadatas = metadatas[0] if metadatas else None
        uris = result.get("uris")
        uris = uris[0] if uris else None
        distances = result.get("distances")
        distances = distances[0] if distances else None
        embeddings = result.get("embeddings")
        embeddings = embeddings[0] if (embeddings is not None and "embeddings" in include) else None

        matches: list[RecordMatchInfo] = []
        for i, record_id in enumerate(result_ids):
            document = documents[i] if documents is not None else None
            metadata = metadatas[i] if metadatas is not None else None
            uri = uris[i] if uris is not None else None
            distance = distances[i] if distances is not None else None

            embedding_dimension: int | None = None
            embedding_preview: list[float] | None = None
            if embeddings is not None and embeddings[i] is not None:
                embedding = list(embeddings[i])
                embedding_dimension = len(embedding)
                embedding_preview = embedding[:8]

            matches.append(
                RecordMatchInfo(
                    id=str(record_id),
                    document=document,
                    metadata=metadata,
                    uri=uri,
                    distance=distance,
                    embedding_dimension=embedding_dimension,
                    embedding_preview=embedding_preview,
                )
            )

        return matches

    def update_record(
        self,
        name: str,
        record_id: str,
        *,
        metadata: dict[str, Any] | None = None,
        uri: str | None = None,
        document: str | None = None,
        embedding_mode: str | None = None,
    ) -> None:
        """Update ``metadata``, ``uri``, and/or ``document`` for a single
        record (technical-spec §3.3, §5.4, §8.3, roadmap M3-3).

        Only fields explicitly given (non-``None``) are passed to
        ``collection.update()``, so omitting one leaves it untouched in
        Chroma. Existence of ``record_id`` is checked up front via
        ``collection.get(ids=[record_id])`` so a missing record raises
        ``RecordNotFoundError`` instead of chromadb's update() silently
        no-op'ing (which is chromadb's actual behavior for unknown ids).

        ``embedding_mode`` is only meaningful when ``document`` is also
        given, and controls what happens to the record's vector
        (technical-spec §3.3):

        - ``None`` (the default): the embedding is left completely
          untouched and the record's metadata is not modified beyond
          whatever ``metadata`` the caller gave -- used internally by tests
          and by any future caller that doesn't need chromaw's stale-
          tracking convention.
        - ``"keep"``: same as ``None`` for the embedding itself, but
          ``chromaw_embedding_status: "stale"`` is merged into the metadata
          sent to ``collection.update()`` -- into the caller-supplied
          ``metadata`` if given, or as a standalone metadata update
          otherwise (chromadb's update merges metadata rather than
          replacing it, so this does not clobber other existing metadata
          keys). This is the mode the API layer uses for
          ``RecordUpdateRequest(embedding_mode="keep")``.
        - ``"reembed"``: ``embedding_resolver.embed_document()`` is called
          *before* ``collection.update()`` (technical-spec §3.1 fail-closed:
          if embedding fails, nothing is written) to compute a fresh vector
          for the new ``document``, which is then sent alongside it.
          ``chromaw_embedding_status`` is set to ``"fresh"`` in the metadata
          (a sentinel value, not merely omitted -- chromadb's metadata merge
          has no way to *delete* a key, so a prior ``"stale"`` from an
          earlier ``"keep"`` edit must be explicitly overwritten to clear
          it).

        For both ``None`` and ``"keep"``, ``document`` requires explicitly
        re-sending the record's *existing* embedding alongside the new text:
        chromadb's ``collection.update()`` otherwise recomputes the
        embedding itself via the collection's embedding function whenever
        ``documents`` is given without an explicit ``embeddings`` (observed
        to raise ``InvalidArgumentError`` for dimension mismatches, or to
        silently replace the vector when dimensions happen to match) --
        exactly what "keep/leave untouched" must prevent.

        Raises:
            CollectionNotFoundError: no collection named ``name`` exists.
            RecordNotFoundError: no record with id ``record_id`` exists in
                the collection.
            EmbeddingFunctionUnavailableError: ``embedding_mode="reembed"``
                was given but no embedding function is available to compute
                the new vector (see ``EmbeddingResolver.embed_document``);
                raised before any write is attempted.
        """
        try:
            collection = self._client.get_collection(name)
        except Exception as exc:
            raise CollectionNotFoundError(f"collection not found: {name}") from exc

        include = ["embeddings"] if document is not None else []
        existing = collection.get(ids=[record_id], include=include)
        if not existing.get("ids"):
            raise RecordNotFoundError(
                f"record not found: {record_id!r} in collection {name!r}"
            )

        # Fail closed: compute the new embedding (if requested) before any
        # write is attempted, so a failure here leaves the record untouched.
        new_embedding: list[float] | None = None
        if document is not None and embedding_mode == "reembed":
            new_embedding = self.embedding_resolver.embed_document(
                document, collection=collection
            )

        final_metadata = dict(metadata) if metadata is not None else None
        if document is not None and embedding_mode == "keep":
            final_metadata = final_metadata or {}
            final_metadata["chromaw_embedding_status"] = "stale"
        elif document is not None and embedding_mode == "reembed":
            final_metadata = final_metadata or {}
            final_metadata["chromaw_embedding_status"] = "fresh"

        update_kwargs: dict[str, Any] = {"ids": [record_id]}
        if final_metadata is not None:
            update_kwargs["metadatas"] = [final_metadata]
        if uri is not None:
            update_kwargs["uris"] = [uri]
        if document is not None:
            update_kwargs["documents"] = [document]
            if embedding_mode == "reembed":
                update_kwargs["embeddings"] = [new_embedding]
            else:
                existing_embeddings = existing.get("embeddings")
                if existing_embeddings is not None and len(existing_embeddings) > 0:
                    # Re-send the current embedding explicitly so chromadb
                    # does not recompute it from the new document via the
                    # collection's embedding function (see docstring above).
                    update_kwargs["embeddings"] = [list(existing_embeddings[0])]

        collection.update(**update_kwargs)

    def delete_record(self, name: str, record_id: str) -> None:
        """Delete a single record from the collection named ``name``
        (technical-spec §3.2, §5.3, §6.5, roadmap M2-7).

        Existence of ``record_id`` is checked up front via
        ``collection.get(ids=[record_id])`` so a missing record raises
        ``RecordNotFoundError`` instead of chromadb's ``delete()`` silently
        no-op'ing for unknown ids.

        Raises:
            CollectionNotFoundError: no collection named ``name`` exists.
            RecordNotFoundError: no record with id ``record_id`` exists in
                the collection.
        """
        try:
            collection = self._client.get_collection(name)
        except Exception as exc:
            raise CollectionNotFoundError(f"collection not found: {name}") from exc

        existing = collection.get(ids=[record_id], include=[])
        if not existing.get("ids"):
            raise RecordNotFoundError(
                f"record not found: {record_id!r} in collection {name!r}"
            )

        collection.delete(ids=[record_id])

    def bulk_delete_records(
        self, name: str, ids: list[str]
    ) -> tuple[list[RecordInfo], list[str]]:
        """Delete multiple records from the collection named ``name`` in one
        call (technical-spec §3.2, §6.5, roadmap M4-2).

        Existence of each requested id is checked up front (mirroring
        ``delete_record``'s single-id existence check) via
        ``collection.get(ids=...)``: ids that exist are captured as full
        ``RecordInfo`` snapshots (document/metadata/uris, used by the caller
        as the audit log's ``before`` values) and passed to
        ``collection.delete()``; ids that don't exist are skipped rather
        than raising, since a bulk operation should still make progress on
        whichever of the requested ids are actually present. Duplicate ids
        are de-duplicated (same as ``get_records``' ``ids`` handling) so
        chromadb's ``DuplicateIDError`` is never hit.

        Returns a ``(deleted_records, skipped_ids)`` tuple: ``deleted_records``
        are the full pre-deletion snapshots of the ids that were actually
        removed (in the order given, deduplicated), and ``skipped_ids`` are
        the requested ids that did not exist in the collection.

        Raises:
            CollectionNotFoundError: no collection named ``name`` exists.
        """
        try:
            collection = self._client.get_collection(name)
        except Exception as exc:
            raise CollectionNotFoundError(f"collection not found: {name}") from exc

        seen: set[str] = set()
        deduped_ids: list[str] = []
        for record_id in ids:
            if record_id not in seen:
                seen.add(record_id)
                deduped_ids.append(record_id)

        existing_records, _, _ = self.get_records(
            name,
            ids=deduped_ids,
            include=("documents", "metadatas", "uris"),
        )
        existing_ids = {record.id for record in existing_records}
        skipped_ids = [rid for rid in deduped_ids if rid not in existing_ids]

        if existing_ids:
            collection.delete(ids=[record.id for record in existing_records])

        return existing_records, skipped_ids

    def bulk_patch_records(
        self, name: str, ids: list[str], metadata: dict[str, Any]
    ) -> tuple[list[tuple[RecordInfo, RecordInfo]], list[str]]:
        """Merge ``metadata`` into multiple records of the collection named
        ``name`` in one call (technical-spec §3.3, §5.4, §6.5, roadmap M4-4).

        Mirrors ``bulk_delete_records``'s existence-checking/de-duplication
        pattern: ids that exist are captured as full ``RecordInfo``
        "before" snapshots (document/metadata/uris) and the given
        ``metadata`` is merged into their metadata via
        ``collection.update()``; ids that don't exist are skipped rather
        than raising, so the operation still makes best-effort progress on
        whichever of the requested ids are actually present. Duplicate ids
        are de-duplicated first, same as ``bulk_delete_records``.

        ``collection.update()`` merges ``metadata`` into *each* record's own
        existing metadata (rather than replacing it), so passing the same
        ``metadata`` dict for every record in the batch is sufficient --
        there is no need to fetch and hand-merge per record. The full "after"
        snapshot is re-fetched post-update (rather than computed locally) so
        the returned/audited values reflect exactly what chromadb now holds.

        Returns a ``(patched_records, skipped_ids)`` tuple: ``patched_records``
        is a list of ``(before, after)`` ``RecordInfo`` pairs for the ids that
        were actually updated (in the order given, deduplicated), and
        ``skipped_ids`` are the requested ids that did not exist in the
        collection.

        Raises:
            CollectionNotFoundError: no collection named ``name`` exists.
        """
        try:
            collection = self._client.get_collection(name)
        except Exception as exc:
            raise CollectionNotFoundError(f"collection not found: {name}") from exc

        seen: set[str] = set()
        deduped_ids: list[str] = []
        for record_id in ids:
            if record_id not in seen:
                seen.add(record_id)
                deduped_ids.append(record_id)

        before_records, _, _ = self.get_records(
            name,
            ids=deduped_ids,
            include=("documents", "metadatas", "uris"),
        )
        existing_ids = [record.id for record in before_records]
        existing_id_set = set(existing_ids)
        skipped_ids = [rid for rid in deduped_ids if rid not in existing_id_set]

        if not existing_ids:
            return [], skipped_ids

        collection.update(
            ids=existing_ids,
            metadatas=[dict(metadata) for _ in existing_ids],
        )

        after_records, _, _ = self.get_records(
            name,
            ids=existing_ids,
            include=("documents", "metadatas", "uris"),
        )
        after_by_id = {record.id: record for record in after_records}

        patched_records = [
            (before, after_by_id[before.id])
            for before in before_records
            if before.id in after_by_id
        ]

        return patched_records, skipped_ids

    def delete_collection(self, name: str) -> None:
        """Delete the collection named ``name`` in its entirety
        (technical-spec §3.2, §5.2, §6.5, roadmap M2-7).

        Existence is checked up front via ``get_collection`` so a missing
        collection raises ``CollectionNotFoundError`` with a consistent
        message, rather than relying on whatever chromadb's
        ``delete_collection`` raises for an unknown name.

        Raises:
            CollectionNotFoundError: no collection named ``name`` exists.
        """
        try:
            self._client.get_collection(name)
        except Exception as exc:
            raise CollectionNotFoundError(f"collection not found: {name}") from exc

        self._client.delete_collection(name)

    def update_collection(
        self,
        name: str,
        *,
        new_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CollectionInfo:
        """Rename and/or update the metadata of the collection named
        ``name`` (technical-spec §5.2, §8.2, roadmap M2-7).

        Unlike ``collection.update()`` for records, chromadb's
        ``collection.modify(metadata=...)`` *replaces* the collection's
        metadata wholesale rather than merging it. To keep the same "merge"
        semantics as record metadata updates (technical-spec §5.4) -- so a
        metadata PATCH here can't silently wipe out unrelated existing
        keys -- this method merges ``metadata`` into the collection's
        current metadata itself before calling ``collection.modify()``.
        Rename and metadata are applied via a single ``collection.modify()``
        call. At least one of ``new_name``/``metadata`` is expected to be
        given by the caller (enforced by ``CollectionUpdateRequest`` at the
        API layer).

        Raises:
            CollectionNotFoundError: no collection named ``name`` exists.
            InvalidCollectionNameError: ``new_name`` is rejected by chromadb
                as malformed (empty, too long, disallowed characters, ...).
            CollectionAlreadyExistsError: ``new_name`` collides with another
                existing collection.
        """
        try:
            collection = self._client.get_collection(name)
        except Exception as exc:
            raise CollectionNotFoundError(f"collection not found: {name}") from exc

        modify_kwargs: dict[str, Any] = {}
        if new_name is not None:
            modify_kwargs["name"] = new_name
        if metadata is not None:
            modify_kwargs["metadata"] = {**(collection.metadata or {}), **metadata}

        try:
            collection.modify(**modify_kwargs)
        except Exception as exc:
            if new_name is None:
                raise
            # chromadb doesn't give this adapter a distinct exception type
            # for "name already taken" vs. "name malformed" -- both surface
            # as generic errors (e.g. ValueError/ChromaError, or a raw
            # sqlite UNIQUE constraint error) from collection.modify().
            # Attribute the error to whichever is more likely from its
            # message, defaulting to "invalid name" so an unrecognized
            # failure still becomes a 422 (client error) rather than a 500.
            message = str(exc).lower()
            if "exist" in message or "unique constraint" in message:
                raise CollectionAlreadyExistsError(
                    f"collection already exists: {new_name!r}"
                ) from exc
            raise InvalidCollectionNameError(
                f"invalid collection name {new_name!r}: {exc}"
            ) from exc

        count = collection.count()
        dimension: int | None = None
        if count > 0:
            sample = collection.get(limit=1, include=["embeddings"])
            embeddings = sample.get("embeddings")
            if embeddings is not None and len(embeddings) > 0:
                dimension = len(embeddings[0])

        return CollectionInfo(
            id=str(collection.id),
            name=collection.name,
            count=count,
            metadata=collection.metadata,
            dimension=dimension,
        )

    def iter_records(
        self, name: str, *, batch_size: int = 500
    ) -> Iterator[dict[str, Any]]:
        """Yield every record of the collection named ``name``, one dict at
        a time, for JSONL export (roadmap M4-3, technical-spec §8).

        Unlike ``get_records``, this fetches records in fixed-size
        ``batch_size`` pages via an internal offset loop and yields them one
        by one, so memory use stays bounded to a single batch regardless of
        collection size -- callers (the ``GET .../export.jsonl`` streaming
        endpoint) can stream the response without holding the whole
        collection in memory. It also deliberately bypasses ``RecordInfo``'s
        ``embedding_preview`` truncation (first 8 values only): each yielded
        dict's ``"embedding"`` is the record's *full* vector, since a
        round-trippable export/import needs the complete embedding, not a
        preview.

        Each yielded dict has keys ``id``, ``document``, ``metadata``,
        ``uri``, ``embedding`` (``list[float] | None``) -- the same fields
        ``import_records``/``ImportEntry`` expect, so an export produced by
        this method can be fed straight back through ``POST .../import``.

        Raises:
            CollectionNotFoundError: no collection named ``name`` exists.
        """
        try:
            collection = self._client.get_collection(name)
        except Exception as exc:
            raise CollectionNotFoundError(f"collection not found: {name}") from exc

        include = ["documents", "metadatas", "uris", "embeddings"]
        offset = 0
        while True:
            try:
                result = collection.get(limit=batch_size, offset=offset, include=include)
            except (ValueError, TypeError, ChromaError):
                # Defensive fallback, mirroring get_records: some chromadb
                # versions reject "uris" in collection.get().
                fallback_include = [item for item in include if item != "uris"]
                result = collection.get(
                    limit=batch_size, offset=offset, include=fallback_include
                )

            ids = result.get("ids") or []
            if not ids:
                break

            documents = result.get("documents")
            metadatas = result.get("metadatas")
            uris = result.get("uris")
            embeddings = result.get("embeddings")

            for i, record_id in enumerate(ids):
                embedding: list[float] | None = None
                if embeddings is not None and embeddings[i] is not None:
                    embedding = [float(v) for v in embeddings[i]]
                yield {
                    "id": str(record_id),
                    "document": documents[i] if documents is not None else None,
                    "metadata": metadatas[i] if metadatas is not None else None,
                    "uri": uris[i] if uris is not None else None,
                    "embedding": embedding,
                }

            if len(ids) < batch_size:
                break
            offset += batch_size

    def import_records(
        self, name: str, entries: list[ImportEntry], *, mode: str = "add"
    ) -> tuple[list[str], list[dict[str, Any]]]:
        """Write parsed JSONL import rows into the collection named ``name``
        (roadmap M4-3, technical-spec §8).

        ``mode="add"`` (default) rejects rows whose ``id`` already exists in
        the collection -- checked up front via a single batched
        ``get_records(ids=..., include=())`` lookup, so those rows never
        reach ``collection.add()`` and are reported back as line-numbered
        errors (``"id already exists"``) rather than aborting the whole
        import. ``mode="upsert"`` skips that check entirely and lets
        ``collection.upsert()`` overwrite any existing id.

        Rows are split into two groups before being written, because
        chromadb's ``collection.add()``/``collection.upsert()`` take a
        single ``embeddings=[...]`` list applying uniformly to the whole
        call (there is no way to say "use the given vector for this row,
        but compute one from the document for that row" in one call):

        - rows that carried an explicit ``embedding`` in the JSONL are
          written with that vector, bypassing the collection's embedding
          function entirely (mirrors ``update_record``'s "keep" mode not
          letting chromadb recompute a vector it wasn't asked to);
        - rows without an ``embedding`` are written with only ``documents``,
          letting chromadb's configured embedding function compute the
          vector -- this fails (surfaced as a per-row error, not a raised
          exception -- see below) if no embedding function is available,
          same underlying cause as ``EmbeddingFunctionUnavailableError``
          elsewhere in this module.

        Each group is additionally chunked to at most ``client.
        get_max_batch_size()`` rows per ``collection.add()``/``collection.
        upsert()`` call (chromadb rejects an entire call outright if it
        exceeds this limit, so a large import would otherwise fail in full
        rather than making partial progress); if the client doesn't expose
        ``get_max_batch_size`` (defensive -- older/alternate chromadb
        clients), a fixed fallback of ``_DEFAULT_MAX_BATCH_SIZE`` (1000) is
        used instead. If a chunk's write call raises, every row in *that
        chunk* (not the one row actually at fault, which chromadb does not
        identify, and not the whole group -- other chunks are written
        independently and are unaffected) is reported as a line-numbered
        error with the exception's message as ``reason`` -- a deliberate
        simplification matching ``get_records``'s ``InvalidFilterError``
        handling: chromadb gives this adapter no reliable way to attribute a
        batch failure to one specific row.

        Returns a ``(imported_ids, errors)`` tuple: ``imported_ids`` are the
        ids of rows actually written (in the order given); ``errors`` are
        ``{"line": int, "reason": str}`` dicts for rows that were rejected
        before or during the write (id-already-exists, or a batch write
        failure).

        Raises:
            CollectionNotFoundError: no collection named ``name`` exists.
        """
        try:
            collection = self._client.get_collection(name)
        except Exception as exc:
            raise CollectionNotFoundError(f"collection not found: {name}") from exc

        if not entries:
            return [], []

        errors: list[dict[str, Any]] = []
        usable = list(entries)

        if mode == "add":
            existing_records, _, _ = self.get_records(
                name, ids=[e.id for e in usable], include=()
            )
            existing_ids = {record.id for record in existing_records}
            remaining: list[ImportEntry] = []
            for entry in usable:
                if entry.id in existing_ids:
                    errors.append(
                        {
                            "line": entry.line,
                            "reason": (
                                f"id already exists: {entry.id!r} "
                                "(use mode=upsert to overwrite)"
                            ),
                        }
                    )
                else:
                    remaining.append(entry)
            usable = remaining

        with_embedding = [e for e in usable if e.embedding is not None]
        without_embedding = [e for e in usable if e.embedding is None]

        try:
            max_batch_size = self._client.get_max_batch_size()
        except AttributeError:
            max_batch_size = _DEFAULT_MAX_BATCH_SIZE
        if not max_batch_size or max_batch_size <= 0:
            max_batch_size = _DEFAULT_MAX_BATCH_SIZE

        imported: list[str] = []

        def _write_chunk(chunk: list[ImportEntry], include_embeddings: bool) -> None:
            write_kwargs: dict[str, Any] = {
                "ids": [e.id for e in chunk],
                "documents": [e.document for e in chunk],
                "metadatas": [e.metadata for e in chunk],
            }
            if any(e.uri is not None for e in chunk):
                write_kwargs["uris"] = [e.uri for e in chunk]
            if include_embeddings:
                write_kwargs["embeddings"] = [e.embedding for e in chunk]

            try:
                if mode == "upsert":
                    collection.upsert(**write_kwargs)
                else:
                    collection.add(**write_kwargs)
            except Exception as exc:
                for e in chunk:
                    errors.append({"line": e.line, "reason": f"import failed: {exc}"})
                return

            imported.extend(e.id for e in chunk)

        def _write(group: list[ImportEntry], include_embeddings: bool) -> None:
            for start in range(0, len(group), max_batch_size):
                chunk = group[start : start + max_batch_size]
                _write_chunk(chunk, include_embeddings)

        _write(with_embedding, include_embeddings=True)
        _write(without_embedding, include_embeddings=False)

        return imported, errors

