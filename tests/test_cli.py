from pathlib import Path

import chromadb
import pytest
from typer.testing import CliRunner

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


def test_path_argument_accepted() -> None:
    result = runner.invoke(app, ["/tmp/some-chroma-dir"])
    assert result.exit_code != 0
    assert "path=/tmp/some-chroma-dir" in result.stdout


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
