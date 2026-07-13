from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator, Callable, Optional

from fastapi import FastAPI

from chromaw.chroma_adapter import ChromaAdapter


def create_app(
    adapter: ChromaAdapter,
    *,
    write: bool,
    on_startup: Optional[Callable[[], None]] = None,
) -> FastAPI:
    """Create the chromaw FastAPI application.

    Endpoints (health, collections, records, ...) are added in later
    milestones (M0-4+). For now the app only exposes its state so that the
    CLI can start a server that carries mode/path/adapter information.

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

    return app
