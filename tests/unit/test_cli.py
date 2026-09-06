"""The command line surface stays a group as subcommands are added."""

import json
import logging
import pathlib
from unittest import mock

import pytest
from typer.testing import CliRunner

from kira import __version__
from kira.cli import app
from kira.memory.factory import build_storage
from kira.memory.ports import ApprovalRecord
from kira.settings.groups import ServerSettings
from kira.settings.root import Settings

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


def test_the_configured_log_level_reaches_the_root_logger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All four shipped deployment artifacts set this, the local overlay to DEBUG, and
    the server hardcoded INFO: an operator raised the level, restarted the pod and got
    the same output with nothing to say why."""
    monkeypatch.setenv("KIRA_OBSERVABILITY__LOG_LEVEL", "DEBUG")
    root = logging.getLogger()
    # basicConfig is a no-op on a root that already has handlers, and the test runner
    # installs one. The server calls it on an untouched interpreter.
    monkeypatch.setattr(root, "handlers", [])
    monkeypatch.setattr(root, "level", root.level)
    with mock.patch("uvicorn.run"):
        assert runner.invoke(app, ["serve"]).exit_code == 0
    assert root.level == logging.DEBUG


def test_approvals_read_and_decide_the_configured_store(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The default path is the store rather than the API, because that is the one that
    works from inside the pod with no port open."""
    monkeypatch.setenv("KIRA_SERVER__WORKDIR", str(tmp_path))
    storage = build_storage(Settings(server=ServerSettings(workdir=tmp_path)))
    storage.approvals.create_request(
        ApprovalRecord(
            request_id="r1",
            session_id="s1",
            kind="tool_call",
            stage="execution",
            status="pending",
            summary='kubectl patch svc user -p \'{"spec":{"ports":[{"targetPort":8080}]}}\'',
            payload={"arguments": {"cmd": "kubectl patch svc user"}, "targets": ["svc/user"]},
            created_at="2026-01-01T00:00:00+00:00",
        )
    )
    storage.close()

    listed = runner.invoke(app, ["approvals", "list", "--json"])
    assert listed.exit_code == 0
    assert [request["request_id"] for request in json.loads(listed.output)] == ["r1"]

    shown = runner.invoke(app, ["approvals", "show", "r1"])
    assert shown.exit_code == 0
    # The whole command and its target, not the six-token summary the prototype showed.
    assert "targetPort" in shown.output
    assert "svc/user" in shown.output

    decided = runner.invoke(app, ["approvals", "approve", "r1", "--by", "an operator"])
    assert decided.exit_code == 0
    assert runner.invoke(app, ["approvals", "approve", "r1"]).exit_code == 1
