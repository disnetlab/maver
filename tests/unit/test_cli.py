"""The command line surface stays a group as subcommands are added."""

from typer.testing import CliRunner

from kira import __version__
from kira.cli import app

runner = CliRunner()


def test_version_subcommand_reports_the_package_version() -> None:
    """Typer collapses a single-command app into that command; the callback prevents it."""
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.output.strip() == __version__


def test_version_flag_reports_the_package_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.output.strip() == __version__


def test_bare_invocation_shows_help_rather_than_running_something() -> None:
    result = runner.invoke(app, [])
    assert "Usage" in result.output
