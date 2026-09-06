"""The direct transport, at the seams where it is checkable without a cluster.

Nothing here proves the transport talks to Kubernetes or to Prometheus correctly; only a
live cluster proves that. What these do check is the four places where a mistake would be
silent: a capability claimed for a backend that was never configured, a rollback tool
offered by a transport that cannot roll anything back, an alert payload the partitioning
downstream cannot read, and a backend failure that escapes as an exception instead of a
result the loop can recognise.
"""

import stat
from pathlib import Path

import httpx
import respx

from kira.audit.journal import Journal
from kira.settings.groups import TransportSettings
from kira.tools.context import ToolContext
from kira.tools.kubernetes import exec_kubectl_cmd_safely
from kira.tools.metrics import _parse_alerts, partition_incident_alerts
from kira.tools.toolsets import build_toolset
from kira.tools.transport.direct import DirectTransport
from kira.tools.workspace import SessionWorkspace

PROMETHEUS = "http://prometheus.test"
LOKI = "http://loki.test"

# Never resolves, so a test never depends on whether the host running it has kubectl.
ABSENT_KUBECTL = "kubectl-that-is-not-installed"


def transport(**overrides: object) -> DirectTransport:
    settings = {"kubectl_binary": ABSENT_KUBECTL} | overrides
    return DirectTransport(TransportSettings(**settings))


def context(tmp_path: Path, bound: DirectTransport) -> ToolContext:
    return ToolContext(
        transport=bound,
        workspace=SessionWorkspace(tmp_path),
        role="rca",
        session_id="s1",
        journal=Journal(session_id="s1", stage="rca"),
    )


def test_an_unconfigured_backend_costs_its_tools_before_the_run(tmp_path: Path) -> None:
    """A capability claimed without a backend behind it fails at incident time instead.

    The drop is the point: a tool that errors reads to a model as a fault to work
    around, and it will try again, differently, and again.
    """
    bound = transport(prometheus_url="http://prometheus.test")

    assert bound.capabilities == frozenset({"metrics"})

    resolved = build_toolset("rca", context(tmp_path, bound))
    offered = {definition.name for definition in resolved.definitions}
    assert "get_metrics" in offered
    assert "get_logs" not in offered
    assert resolved.dropped["get_logs"] == "transport lacks logs"
    # No kubectl on this host, so the cluster tools go the same way rather than failing
    # on every call.
    assert (
        resolved.dropped["exec_kubectl_cmd_safely"]
        == "transport lacks kubernetes, kubernetes.write"
    )


async def test_rollback_is_never_offered_and_says_why_if_reached() -> None:
    """Undoing a write needs a journal of inverses, which this transport does not keep."""
    bound = transport()

    assert "kubernetes.rollback" not in bound.capabilities
    assert (await bound.rollbackable_commands()).startswith("error: ")
    assert (await bound.rollback_last()).startswith("error: ")


@respx.mock
async def test_alerts_are_returned_in_the_encoding_the_partitioning_reads() -> None:
    """The seam is ``ast.literal_eval``, so a JSON rendering would parse as nothing.

    A payload the partitioning cannot read is passed through byte for byte, which means
    a foreign namespace's firing alert would reach the model as its own evidence with
    nothing to say it had happened.
    """
    respx.get(f"{PROMETHEUS}/api/v1/alerts").mock(
        return_value=httpx.Response(
            200,
            json={
                "status": "success",
                "data": {
                    "alerts": [
                        {
                            "labels": {"alertname": "PodDown", "namespace": "shop"},
                            "annotations": {"summary": "a pod is down"},
                            "state": "firing",
                        },
                        {
                            "labels": {"alertname": "Latency", "namespace": "other"},
                            "annotations": {},
                            "state": "firing",
                        },
                        {
                            "labels": {"alertname": "Warming", "namespace": "shop"},
                            "annotations": {},
                            "state": "pending",
                        },
                    ]
                },
            },
        )
    )

    raw = await transport(prometheus_url=PROMETHEUS).query_alerts()

    parsed = _parse_alerts(raw)
    assert parsed is not None
    assert [alert["labels"]["alertname"] for alert in parsed] == ["PodDown", "Latency"]

    partitioned = partition_incident_alerts("shop", raw)
    assert partitioned.startswith("INCIDENT-SCOPED firing alerts for namespace 'shop':")
    assert "OUT-OF-SCOPE firing alerts from other namespaces" in partitioned


@respx.mock
async def test_a_backend_that_fails_returns_a_result_rather_than_raising() -> None:
    """The loop counts an exception as a successful turn; only the prefix says otherwise.

    A rejected query also carries the backend's own words, because that is what lets a
    model repair a malformed query instead of reissuing it.
    """
    respx.get(f"{PROMETHEUS}/api/v1/query").mock(side_effect=httpx.ConnectError("no route"))
    respx.get(f"{LOKI}/loki/api/v1/query_range").mock(
        return_value=httpx.Response(400, text="parse error at line 1, col 7: syntax error")
    )
    bound = transport(prometheus_url=PROMETHEUS, loki_url=LOKI)

    unreachable = await bound.query_metrics("up")
    rejected = await bound.query_logs("{namespace=}")

    assert unreachable == "error: prometheus request failed: ConnectError: no route"
    assert (
        rejected
        == "error: loki rejected the query (400): parse error at line 1, col 7: syntax error"
    )


async def test_a_blocking_command_is_bounded_under_the_transport_deadline(
    tmp_path: Path,
) -> None:
    """Ten seconds under, so kubectl reports its own timeout before it is killed.

    Also the one check that the argument list arrives as written and that a successful
    command's output comes back with nothing wrapped around it: two tools parse that
    text as JSON, and a prefix in front of it turns both into "unavailable".
    """
    fake = tmp_path / "kubectl"
    fake.write_text('#!/bin/sh\necho "$@"\n')
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    bound = transport(kubectl_binary=str(fake), timeout_s=60)

    result = await exec_kubectl_cmd_safely(
        context(tmp_path, bound), cmd="kubectl wait --for=condition=Ready pod/x"
    )

    assert result == "wait --for=condition=Ready pod/x --timeout=50s\n"
