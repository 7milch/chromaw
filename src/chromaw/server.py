from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator, Callable, Optional

from fastapi import FastAPI

from chromaw.api import router as api_router
from chromaw.chroma_adapter import ChromaAdapter


def create_app(
    adapter: ChromaAdapter,
    *,
    write: bool,
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
    """

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if on_startup is not None:
            on_startup()
        yield

    app = FastAPI(title="chromaw", lifespan=lifespan)

    app.state.adapter = adapter
    app.state.mode = "write" if write else "read-only"
    app.state.path = adapter.path

    app.include_router(api_router)

    return app
