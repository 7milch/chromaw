from __future__ import annotations

from contextlib import asynccontextmanager
from importlib.resources import files
from typing import AsyncIterator, Callable, Optional

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from chromaw.api import router as api_router
from chromaw.chroma_adapter import ChromaAdapter
from chromaw.errors import CollectionNotFoundError, InvalidFilterError
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
