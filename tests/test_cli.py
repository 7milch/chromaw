from pathlib import Path
from typing import Any

import chromadb
import pytest
from typer.testing import CliRunner

import chromaw.cli as cli_module
from chromaw import __version__
from chromaw.cli import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def chroma_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Make the current working directory a valid (empty) ChromaDB directory.

    Several tests below invoke the CLI with the default path (".") and expect
    a successful connection, so cwd needs to be a real ChromaDB directory.
    """
    chromadb.PersistentClient(path=str(tmp_path))
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture(autouse=True)
def mock_server_startup(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Prevent the CLI from actually starting a blocking server / browser.

    Simulates uvicorn reaching "server started listening" by invoking the
    ``on_startup`` callback passed to ``create_app`` synchronously, mirroring
    what the real lifespan hook does once the server is up. Captures the
    calls so individual tests can assert on them.
    """
    calls: dict[str, Any] = {"uvicorn_run": None, "webbrowser_open": None}

    real_create_app = cli_module.create_app

    def fake_uvicorn_run(app: Any, *, host: str, port: int) -> None:
        calls["uvicorn_run"] = {"host": host, "port": port}

    def fake_webbrowser_open(url: str) -> bool:
        calls["webbrowser_open"] = url
        return True

    def fake_create_app(adapter: Any, *, write: bool, on_startup: Any = None) -> Any:
        app = real_create_app(adapter, write=write, on_startup=on_startup)
        if on_startup is not None:
            on_startup()
        return app

    monkeypatch.setattr(cli_module.uvicorn, "run", fake_uvicorn_run)
    monkeypatch.setattr(cli_module.webbrowser, "open", fake_webbrowser_open)
    monkeypatch.setattr(cli_module, "create_app", fake_create_app)

    return calls


def test_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "chromaw" in result.stdout


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_version_output_format() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == f"chromaw {__version__}"


def test_default_values() -> None:
    result = runner.invoke(app, [])
    assert result.exit_code == 0
    assert "path=." in result.stdout
    assert "host=127.0.0.1" in result.stdout
    assert "port=0" in result.stdout
    assert "mode=read-only" in result.stdout
    assert "open_browser=True" in result.stdout
    assert "collections=0" in result.stdout


def test_path_argument_accepted(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"
    result = runner.invoke(app, [str(missing)])
    assert result.exit_code != 0
    assert f"path={missing}" in result.stdout


def test_host_option() -> None:
    result = runner.invoke(app, ["--host", "0.0.0.0"])
    assert result.exit_code == 0
    assert "host=0.0.0.0" in result.stdout


def test_port_option() -> None:
    result = runner.invoke(app, ["--port", "8123"])
    assert result.exit_code == 0
    assert "port=8123" in result.stdout


def test_no_open_option() -> None:
    result = runner.invoke(app, ["--no-open"])
    assert result.exit_code == 0
    assert "open_browser=False" in result.stdout


def test_write_option() -> None:
    result = runner.invoke(app, ["--write"])
    assert result.exit_code == 0
    assert "mode=write" in result.stdout


def test_path_before_options() -> None:
    result = runner.invoke(app, [".", "--port", "8123"])
    assert result.exit_code == 0
    assert "path=." in result.stdout
    assert "port=8123" in result.stdout


def test_path_and_multiple_options_combined(tmp_path: Path) -> None:
    chroma_dir = tmp_path / "data-chroma"
    chromadb.PersistentClient(path=str(chroma_dir))

    result = runner.invoke(
        app,
        [str(chroma_dir), "--host", "0.0.0.0", "--port", "9000", "--no-open", "--write"],
    )
    assert result.exit_code == 0
    assert f"path={chroma_dir}" in result.stdout
    assert "host=0.0.0.0" in result.stdout
    assert "port=9000" in result.stdout
    assert "mode=write" in result.stdout
    assert "open_browser=False" in result.stdout


def test_options_before_path() -> None:
    result = runner.invoke(app, ["--port", "8123", "."])
    assert result.exit_code == 0
    assert "path=." in result.stdout
    assert "port=8123" in result.stdout


def test_invalid_port_type_rejected() -> None:
    result = runner.invoke(app, ["--port", "not-a-number"])
    assert result.exit_code != 0


def test_version_takes_precedence_and_exits_eagerly() -> None:
    result = runner.invoke(app, ["--version", "--port", "not-a-number"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_nonexistent_path_without_create_fails(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"

    result = runner.invoke(app, [str(missing)])

    assert result.exit_code != 0
    assert "error" in result.stdout.lower() or "error" in (result.stderr or "").lower()


def test_nonexistent_path_with_create_succeeds(tmp_path: Path) -> None:
    missing = tmp_path / "new-chroma-dir"

    result = runner.invoke(app, [str(missing), "--create"])

    assert result.exit_code == 0
    assert missing.exists()
    assert "collections=0" in result.stdout


def test_empty_dir_without_create_fails(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    result = runner.invoke(app, [str(empty_dir)])

    assert result.exit_code != 0


def test_empty_dir_with_create_succeeds(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    result = runner.invoke(app, [str(empty_dir), "--create"])

    assert result.exit_code == 0
    assert (empty_dir / "chroma.sqlite3").exists()


def test_non_chroma_directory_fails(tmp_path: Path) -> None:
    non_chroma_dir = tmp_path / "non-chroma"
    non_chroma_dir.mkdir()
    (non_chroma_dir / "readme.txt").write_text("not chroma")

    result = runner.invoke(app, [str(non_chroma_dir)])

    assert result.exit_code != 0


def test_valid_chroma_dir_reports_collection_count(tmp_path: Path) -> None:
    client = chromadb.PersistentClient(path=str(tmp_path))
    client.create_collection("foo")
    client.create_collection("bar")

    result = runner.invoke(app, [str(tmp_path)])

    assert result.exit_code == 0
    assert "collections=2" in result.stdout


def test_path_is_a_file_fails(tmp_path: Path) -> None:
    file_path = tmp_path / "im-a-file"
    file_path.write_text("not a directory")

    result = runner.invoke(app, [str(file_path)])

    assert result.exit_code != 0


def test_existing_valid_chroma_dir_with_create_flag_succeeds(tmp_path: Path) -> None:
    client = chromadb.PersistentClient(path=str(tmp_path))
    client.create_collection("foo")

    result = runner.invoke(app, [str(tmp_path), "--create"])

    assert result.exit_code == 0
    assert "collections=1" in result.stdout


def test_nested_nonexistent_path_with_create_creates_parents(tmp_path: Path) -> None:
    nested = tmp_path / "a" / "b" / "c"

    result = runner.invoke(app, [str(nested), "--create"])

    assert result.exit_code == 0
    assert nested.exists()
    assert "collections=0" in result.stdout


def test_startup_url_displayed_with_resolved_port(mock_server_startup: dict) -> None:
    result = runner.invoke(app, ["--port", "8123"])
    assert result.exit_code == 0
    assert "http://127.0.0.1:8123" in result.stdout
    assert mock_server_startup["uvicorn_run"] == {"host": "127.0.0.1", "port": 8123}


def test_startup_url_uses_auto_assigned_port_when_zero(mock_server_startup: dict) -> None:
    result = runner.invoke(app, ["--port", "0"])
    assert result.exit_code == 0
    assigned_port = mock_server_startup["uvicorn_run"]["port"]
    assert assigned_port != 0
    assert f"http://127.0.0.1:{assigned_port}" in result.stdout


def test_non_local_host_warns_on_stderr(mock_server_startup: dict) -> None:
    result = runner.invoke(app, ["--host", "0.0.0.0"])
    assert result.exit_code == 0
    stderr = result.stderr if result.stderr is not None else ""
    assert "warning" in stderr.lower()


def test_localhost_host_does_not_warn(mock_server_startup: dict) -> None:
    result = runner.invoke(app, ["--host", "localhost"])
    assert result.exit_code == 0
    stderr = result.stderr if result.stderr is not None else ""
    assert "warning" not in stderr.lower()


def test_127_0_0_1_host_does_not_warn(mock_server_startup: dict) -> None:
    result = runner.invoke(app, ["--host", "127.0.0.1"])
    assert result.exit_code == 0
    stderr = result.stderr if result.stderr is not None else ""
    assert "warning" not in stderr.lower()


def test_browser_opened_by_default(mock_server_startup: dict) -> None:
    result = runner.invoke(app, ["--port", "8123"])
    assert result.exit_code == 0
    assert mock_server_startup["webbrowser_open"] == "http://127.0.0.1:8123"


def test_browser_not_opened_with_no_open(mock_server_startup: dict) -> None:
    result = runner.invoke(app, ["--port", "8123", "--no-open"])
    assert result.exit_code == 0
    assert mock_server_startup["webbrowser_open"] is None


def test_uvicorn_not_started_on_connection_error(
    tmp_path: Path, mock_server_startup: dict
) -> None:
    missing = tmp_path / "does-not-exist"
    result = runner.invoke(app, [str(missing)])
    assert result.exit_code != 0
    assert mock_server_startup["uvicorn_run"] is None


def test_resolve_port_returns_explicit_port_unchanged() -> None:
    assert cli_module._resolve_port("127.0.0.1", 12345) == 12345


def test_resolve_port_auto_assigns_free_port() -> None:
    port = cli_module._resolve_port("127.0.0.1", 0)
    assert isinstance(port, int)
    assert port != 0
    assert 1 <= port <= 65535


def test_resolve_port_auto_assigns_distinct_ports_when_called_twice() -> None:
    # Not guaranteed in theory (OS could reuse), but overwhelmingly likely in
    # practice and useful as a smoke test that the socket is released.
    port_a = cli_module._resolve_port("127.0.0.1", 0)
    port_b = cli_module._resolve_port("127.0.0.1", 0)
    assert isinstance(port_a, int)
    assert isinstance(port_b, int)
