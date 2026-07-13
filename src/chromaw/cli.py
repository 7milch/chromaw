from __future__ import annotations

from typing import Optional

import typer

from chromaw import __version__

app = typer.Typer(
    name="chromaw",
    help="Browse, search, and edit a local ChromaDB persistent directory in your browser.",
    add_completion=False,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"chromaw {__version__}")
        raise typer.Exit()


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
    typer.echo("server startup is not implemented yet (M0-3)")
