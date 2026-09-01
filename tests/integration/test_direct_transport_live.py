"""The direct transport against a real cluster, read-only.

Everything else about this transport is checked at its own I/O seam, which proves the
code does what the stub was told to expect and nothing about whether a real API server
agrees. These are the checks that need the cluster, and each one covers something that
went wrong for real: an enveloped read that silently emptied the namespace snapshot, a
missing namespace mistaken for a present one, and a snapshot that came back blank in
roughly nineteen runs out of twenty without anybody noticing.

Read-only throughout. Every command is written here rather than composed from anything
a model produced, so there is no path from these tests to a write.

Run them with a kubeconfig pointing at a cluster you are willing to have read:

    uv run pytest -m integration tests/integration/test_direct_transport_live.py
"""

import json
import pathlib
import subprocess

import pytest

from kira.audit.journal import Journal, ListSink
from kira.settings.groups import ToolSettings, TransportSettings
from kira.tools.context import ToolContext
from kira.tools.errors import call_failed
from kira.tools.metrics import _namespace_is_missing
from kira.tools.scope import IncidentScope
from kira.tools.snapshot import namespace_snapshot
from kira.tools.transport.direct import direct_transport
from kira.tools.workspace import SessionWorkspace

ABSENT = "kira-definitely-not-a-namespace"

_PROBE_TIMEOUT_S = 5


def _cluster_is_reachable() -> bool:
    """Ask the API server one bounded question before the suite commits to it.

    Without this the suite does not fail when there is no cluster, it *waits*: the
    transport's own bound is two minutes per call, so an unreachable API server turns a
    read-only smoke test into a twenty-minute hang, and `make test-all` never returns.
    A dependency that is absent should say so in seconds, the way the database-backed
    suites do. The skip is loud under `--no-skips`, which is what keeps it from
    quietly reporting green in CI.
    """
    try:
        probe = subprocess.run(
            ["kubectl", "version", "-o", "json", f"--request-timeout={_PROBE_TIMEOUT_S}s"],
            capture_output=True,
            timeout=_PROBE_TIMEOUT_S * 2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return probe.returncode == 0


# Marked rather than skipped at module level: a module-level skip is raised during
# collection, so it never becomes a test report and `--no-skips` cannot see it. The
# probe runs once at import either way.
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _cluster_is_reachable(),
        reason="no reachable cluster: point a kubeconfig at one you are willing to have read",
    ),
]


@pytest.fixture
def transport() -> object:
    return direct_transport(TransportSettings())


async def test_a_read_comes_back_as_the_command_printed_it(transport: object) -> None:
    """The property the shared contract pins, against a real API server. Anything added
    in front of a successful read is invisible until the namespace snapshot silently
    goes blank and trace scoping quietly turns off, because both parse this as JSON."""
    output = await transport.kubectl(["get", "nodes", "-o", "json"])  # type: ignore[attr-defined]

    # That it parses at all is the property. The envelope kind varies with the client
    # version, and pinning it would fail on a cluster this test is meant to accept.
    parsed = json.loads(output)
    assert parsed["items"]
    assert all(item["kind"] == "Node" for item in parsed["items"])


async def test_a_namespace_that_is_gone_is_told_apart_from_one_that_is_there(
    transport: object,
) -> None:
    """What the stale-alert annotation rests on. Only an explicit not-found counts,
    because a timeout says nothing about whether the namespace exists and treating it
    as proof would mark a live incident's own alerts stale."""
    present = await transport.kubectl(["get", "namespace", "default", "-o", "name"])  # type: ignore[attr-defined]
    absent = await transport.kubectl(["get", "namespace", ABSENT, "-o", "name"])  # type: ignore[attr-defined]

    assert not _namespace_is_missing(present)
    assert _namespace_is_missing(absent)
    assert call_failed(absent), "a command that ran and failed has to read as a failure"


async def test_the_snapshot_is_not_empty(transport: object, tmp_path: pathlib.Path) -> None:
    """The prototype's snapshot came back empty in roughly nineteen runs out of twenty,
    in every measured campaign, and the diagnosis it was meant to anchor never noticed.
    An assertion that it has content is the whole point."""
    context = ToolContext(
        transport=transport,  # type: ignore[arg-type]
        workspace=SessionWorkspace(tmp_path),
        role="rca",
        session_id="s1",
        journal=Journal(ListSink(), session_id="s1", stage="rca"),
        settings=ToolSettings(),
        scope=IncidentScope(namespace="default"),
    )

    snapshot = await namespace_snapshot(context, "default")

    assert "NAMESPACE SNAPSHOT: default" in snapshot
    assert "(unavailable)" not in snapshot, "every section a real cluster can answer"
    assert len([line for line in snapshot.splitlines() if line.strip()]) > 5
