from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
from chromadb.errors import ChromaError

from chromaw.errors import (
    ChromaEmptyDirectoryError,
    ChromaInvalidDirectoryError,
    ChromaPathNotFoundError,
    CollectionNotFoundError,
)
from chromaw.models import CollectionInfo, RecordInfo

_SQLITE_FILENAME = "chroma.sqlite3"


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

    @classmethod
    def open(cls, path: Path, create: bool = False) -> "ChromaAdapter":
        """Open (and optionally create) a ChromaDB persistent directory.

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
        limit: int = 50,
        offset: int = 0,
        include: tuple[str, ...] = ("documents", "metadatas", "uris"),
    ) -> tuple[list[RecordInfo], int]:
        """Return a page of records for the collection named ``name``.

        When ``ids`` is given, results are restricted to those ids
        (``collection.get(ids=...)``) and paging (``limit``/``offset``) is
        not applied by Chroma to the ids list itself. Returns a
        ``(records, total)`` tuple where ``total`` is the number of records
        returned (i.e. matching the given ``ids``) rather than
        ``collection.count()`` in that case, since "the collection's full
        record count" is not a meaningful notion of total for an ids-scoped
        lookup; without ``ids``, ``total`` remains ``collection.count()`` as
        before.

        ``embedding_dimension``/``embedding_preview`` (first 8 values) are
        only populated when ``"embeddings"`` is present in ``include``;
        callers that don't need them can omit it to avoid the extra cost of
        fetching embedding vectors.

        Raises:
            CollectionNotFoundError: no collection named ``name`` exists.
        """
        try:
            collection = self._client.get_collection(name)
        except Exception as exc:
            raise CollectionNotFoundError(f"collection not found: {name}") from exc

        if ids is not None and len(ids) == 0:
            # chromadb raises a ValueError for collection.get(ids=[]); an
            # empty ids list unambiguously matches zero records, so return
            # an empty page without calling into chromadb at all.
            return [], 0

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

        get_include = [item for item in include if item != "ids"]
        get_kwargs: dict[str, Any] = {"include": get_include}
        if ids is not None:
            get_kwargs["ids"] = ids
        else:
            get_kwargs["limit"] = limit
            get_kwargs["offset"] = offset

        try:
            result = collection.get(**get_kwargs)
        except (ValueError, TypeError, ChromaError) as exc:
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

        total = len(result.get("ids") or []) if ids is not None else collection.count()

        ids = result.get("ids") or []
        documents = result.get("documents")
        metadatas = result.get("metadatas")
        uris = result.get("uris")
        embeddings = result.get("embeddings") if "embeddings" in include else None

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

        return records, total

