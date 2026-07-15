from __future__ import annotations

from contextlib import asynccontextmanager
from importlib.resources import files
from typing import AsyncIterator, Callable, Optional

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from chromaw.api import router as api_router
from chromaw.audit import AuditLogger
from chromaw.backup import BackupManager
from chromaw.chroma_adapter import ChromaAdapter
from chromaw.errors import (
    AuditWriteFailedError,
    BackupFailedError,
    CollectionAlreadyExistsError,
    CollectionNotFoundError,
    ConfirmationMismatchError,
    EmbeddingFunctionUnavailableError,
    InvalidCollectionNameError,
    InvalidFilterError,
    InvalidQueryEmbeddingError,
    RecordNotFoundError,
)
from chromaw.security import SecurityMiddleware, generate_token


def _inject_token_meta(html: str, token: str) -> str:
    """Insert a ``<meta name="chromaw-token">`` tag before ``</head>``.

    Falls back to prepending the tag when the document has no ``</head>``
    (defensive; the shipped index.html always has one).
    """

    meta_tag = f'<meta name="chromaw-token" content="{token}">'
    if "</head>" in html:
        return html.replace("</head>", f"{meta_tag}\n</head>", 1)
    return f"{meta_tag}\n{html}"


def create_app(
    adapter: ChromaAdapter,
    *,
    write: bool,
    token: Optional[str] = None,
    host: str = "127.0.0.1",
    port: int = 0,
    on_startup: Optional[Callable[[], None]] = None,
) -> FastAPI:
    """Create the chromaw FastAPI application.

    API endpoints live under ``/api`` via APIRouters (see ``chromaw.api``);
    further routers (collections, records, ...) are added in later
    milestones (M0-5+) by including them here alongside ``api_router``.

    ``on_startup``, if given, is invoked once the application has finished
    starting up (e.g. after uvicorn has begun listening), which lets the CLI
    open a browser window in sync with the server actually being reachable
    instead of guessing with a fixed delay.

    Security (technical-spec §10.2): a random bearer ``token`` is required
    for every ``/api/*`` request; ``host``/``port`` are used to build the
    Host/Origin allow-list. If ``token`` is not supplied, one is generated
    automatically. The token is exposed on ``app.state.token`` and injected
    into the served ``index.html`` as a ``<meta name="chromaw-token">`` tag
    so the frontend can pick it up without a separate round trip.
    """

    resolved_token = token if token is not None else generate_token()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if on_startup is not None:
            on_startup()
        yield

    app = FastAPI(title="chromaw", lifespan=lifespan)

    app.state.adapter = adapter
    app.state.mode = "write" if write else "read-only"
    app.state.path = adapter.path
    app.state.token = resolved_token
    # Pre-first-write backup (technical-spec §9.1, roadmap M2-5): only
    # relevant when write endpoints are reachable at all, so it's only
    # created in write mode. Write endpoints look this up via
    # ``request.app.state.backup_manager`` before mutating anything.
    app.state.backup_manager = BackupManager(adapter.path) if write else None
    # Audit log (technical-spec §9.2, roadmap M2-6): same read-only/write
    # gating as the backup manager above -- only reachable write endpoints
    # ever need to append entries, so the logger is only created in write
    # mode. Write endpoints look this up via
    # ``request.app.state.audit_logger`` after a mutation succeeds.
    app.state.audit_logger = AuditLogger(adapter.path) if write else None

    app.add_middleware(SecurityMiddleware, token=resolved_token, host=host, port=port)

    @app.exception_handler(CollectionNotFoundError)
    async def _collection_not_found_handler(
        _: Request, exc: CollectionNotFoundError
    ) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(InvalidFilterError)
    async def _invalid_filter_handler(
        _: Request, exc: InvalidFilterError
    ) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(RecordNotFoundError)
    async def _record_not_found_handler(
        _: Request, exc: RecordNotFoundError
    ) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(ConfirmationMismatchError)
    async def _confirmation_mismatch_handler(
        _: Request, exc: ConfirmationMismatchError
    ) -> JSONResponse:
        # technical-spec §3.2/§6.5: a wrong/missing confirm value means the
        # destructive request conflicts with what the server requires to
        # proceed -- 409, not 422 (the request is well-formed, just not
        # confirmed) and not 400 (nothing about the confirm field's *shape*
        # is invalid).
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(CollectionAlreadyExistsError)
    async def _collection_already_exists_handler(
        _: Request, exc: CollectionAlreadyExistsError
    ) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(InvalidCollectionNameError)
    async def _invalid_collection_name_handler(
        _: Request, exc: InvalidCollectionNameError
    ) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(EmbeddingFunctionUnavailableError)
    async def _embedding_function_unavailable_handler(
        _: Request, exc: EmbeddingFunctionUnavailableError
    ) -> JSONResponse:
        # technical-spec §5.6 4: a query_text query that the embedding
        # function currently can't fulfil (unavailable/failed) is not a
        # malformed request -- 503, not 422.
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.exception_handler(InvalidQueryEmbeddingError)
    async def _invalid_query_embedding_handler(
        _: Request, exc: InvalidQueryEmbeddingError
    ) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(BackupFailedError)
    async def _backup_failed_handler(_: Request, exc: BackupFailedError) -> JSONResponse:
        # Fail-closed (technical-spec §9.1): the write that triggered the
        # backup must not have proceeded; surface this as a server error
        # rather than a 4xx since it's not something the client did wrong.
        return JSONResponse(status_code=500, content={"detail": str(exc)})

    @app.exception_handler(AuditWriteFailedError)
    async def _audit_write_failed_handler(
        _: Request, exc: AuditWriteFailedError
    ) -> JSONResponse:
        # Fail-closed (technical-spec §9.2): the mutation itself already
        # happened by the time this is raised, but the client must not be
        # told the operation succeeded if it couldn't be recorded.
        return JSONResponse(status_code=500, content={"detail": str(exc)})

    app.include_router(api_router)

    static_dir = files("chromaw").joinpath("static")
    index_path = static_dir.joinpath("index.html")
    if index_path.is_file():
        index_html = _inject_token_meta(index_path.read_text(), resolved_token)

        @app.get("/", include_in_schema=False)
        def _index() -> HTMLResponse:
            return HTMLResponse(index_html)

        # Note: this StaticFiles mount serves index.html verbatim from disk
        # (no token meta injection) for any request that resolves to it via
        # the static file router -- most notably a literal "GET /index.html"
        # request, as opposed to "GET /" which is handled by the route above
        # and returns the token-injected `index_html` in memory. This is
        # intentional/harmless: the frontend always fetches "/" to obtain the
        # token, so a client hitting "/index.html" directly simply won't find
        # a <meta name="chromaw-token"> tag and must fall back to loading
        # "/" instead.
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

    return app
