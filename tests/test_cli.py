from typer.testing import CliRunner

from chromaw import __version__
from chromaw.cli import app

runner = CliRunner()


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


def test_path_argument_accepted() -> None:
    result = runner.invoke(app, ["/tmp/some-chroma-dir"])
    assert result.exit_code == 0
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


def test_path_and_multiple_options_combined() -> None:
    result = runner.invoke(
        app,
        ["/data/chroma", "--host", "0.0.0.0", "--port", "9000", "--no-open", "--write"],
    )
    assert result.exit_code == 0
    assert "path=/data/chroma" in result.stdout
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
