from __future__ import annotations

import socket
import webbrowser
from pathlib import Path
from typing import Optional

import typer
import uvicorn

from chromaw import __version__
from chromaw.chroma_adapter import ChromaAdapter
from chromaw.errors import ChromawError
from chromaw.server import create_app

app = typer.Typer(
    name="chromaw",
    help="Browse, search, and edit a local ChromaDB persistent directory in your browser.",
    add_completion=False,
)

_LOCAL_HOSTS = {"127.0.0.1", "localhost"}


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"chromaw {__version__}")
        raise typer.Exit()


def _resolve_port(host: str, port: int) -> int:
    """Return ``port`` unchanged, or an auto-assigned free port when ``port`` is 0.

    Binds a socket to determine a free port and immediately releases it. This
    cannot fully eliminate the race against another process grabbing the same
    port before uvicorn binds it, but it minimizes the window.
    """
    if port != 0:
        return port

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return sock.getsockname()[1]


@app.command()
def main(
    path: str = typer.Argument(
        ".",
        help="Path to the ChromaDB persistent directory.",
    ),
    host: str = typer.Option(
        "127.0.0.1",
        "--host",
        help="Host address to bind the server to.",
    ),
    port: int = typer.Option(
        0,
        "--port",
        help="Port to bind the server to. 0 means auto-assign.",
    ),
    no_open: bool = typer.Option(
        False,
        "--no-open",
        help="Do not automatically open a browser window.",
    ),
    write: bool = typer.Option(
        False,
        "--write",
        help="Enable editing. Default is read-only.",
    ),
    create: bool = typer.Option(
        False,
        "--create",
        help="Create the ChromaDB persistent directory if it does not exist or is empty.",
    ),
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show the chromaw version and exit.",
    ),
) -> None:
    """Start the chromaw web UI for the given ChromaDB persistent directory."""
    mode = "write" if write else "read-only"
    typer.echo(f"chromaw: path={path} host={host} port={port} mode={mode} open_browser={not no_open}")

    if host not in _LOCAL_HOSTS:
        typer.echo(
            f"warning: binding to '{host}' exposes chromaw beyond localhost. "
            "Only do this on a trusted network.",
            err=True,
        )

    try:
        adapter = ChromaAdapter.open(Path(path), create=create)
    except ChromawError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1)

    collections = adapter.list_collections()
    typer.echo(f"connected: path={adapter.path} collections={len(collections)}")

    resolved_port = _resolve_port(host, port)
    url = f"http://{host}:{resolved_port}"
    typer.echo(f"chromaw is running at {url}")

    def open_browser() -> None:
        if not no_open:
            webbrowser.open(url)

    fastapi_app = create_app(adapter, write=write, on_startup=open_browser)
    uvicorn.run(fastapi_app, host=host, port=resolved_port)
