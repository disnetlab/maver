"""Tests for the namespace snapshot.

Two things here earn a test. The three-state read contract, because losing it is how a
failed read came to print "(no ConfigMap/Secret references found)" — an asserted false
negative, which is worse than a gap, and invisible in a transcript. And the rendered
report itself, because every line of it is model-facing text the prompts and every
published measurement were written against.
"""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from maver.audit.journal import Journal
from maver.tools.context import ToolContext
from maver.tools.snapshot import namespace_snapshot
from maver.tools.workspace import SessionWorkspace

_RESOURCE = {
    "deployments,statefulsets,daemonsets": "workloads",
    "pods": "pods",
    "services": "services",
    "endpoints": "endpoints",
    "configmaps": "configmaps",
    "secrets": "secrets",
    "events": "events",
    "nodes": "nodes",
}


class FakeCluster:
    """Answers each of the eight reads from a canned payload, keyed by resource."""

    name = "fake"
    capabilities = frozenset({"kubernetes"})
    timeout_seconds = 30

    def __init__(self, payloads: dict[str, str], fail_once: frozenset[str] = frozenset()) -> None:
        self.payloads = payloads
        self.fail_once = set(fail_once)
        self.calls: list[str] = []

    async def kubectl(self, argv: list[str]) -> str:
        self.calls.append(" ".join(argv))
        key = _RESOURCE[argv[1]]
        if key in self.fail_once:
            self.fail_once.discard(key)
            return "error: couldn't get current server API group list"
        return self.payloads[key]


def context(tmp_path: Path, transport: Any) -> ToolContext:
    return ToolContext(
        transport=transport,
        workspace=SessionWorkspace(tmp_path),
        role="rca",
        session_id="s1",
        journal=Journal(session_id="s1", stage="rca"),
    )


def listing(items: list[dict[str, Any]]) -> str:
    return json.dumps({"apiVersion": "v1", "kind": "List", "items": items})


def empty_reads() -> dict[str, str]:
    """Every read succeeds and the namespace is genuinely empty."""
    return {key: listing([]) for key in _RESOURCE.values()}


def failed_reads() -> dict[str, str]:
    """Every read fails."""
    return dict.fromkeys(_RESOURCE.values(), "error: timed out after 90.0s")


def populated_reads() -> dict[str, str]:
    """A namespace holding one broken Deployment, its Service, and two nodes."""
    return {
        "workloads": listing(
            [
                {
                    "kind": "Deployment",
                    "metadata": {"name": "frontend", "generation": 2},
                    "spec": {
                        "replicas": 2,
                        "template": {
                            "spec": {
                                "containers": [
                                    {
                                        "image": "reg/web:1.2@sha256:deadbeef",
                                        "ports": [{"containerPort": 8080, "name": "http"}],
                                        "envFrom": [{"configMapRef": {"name": "web-config"}}],
                                    }
                                ],
                                "volumes": [{"configMap": {"name": "missing-cm"}}],
                            }
                        },
                    },
                    "status": {"readyReplicas": 1, "observedGeneration": 1},
                }
            ]
        ),
        "pods": listing(
            [
                {
                    "metadata": {
                        "name": "frontend-5f8-abcde",
                        "labels": {"app": "frontend"},
                        "ownerReferences": [{"kind": "ReplicaSet", "name": "frontend-5f8"}],
                    },
                    "spec": {
                        "nodeName": "node-1",
                        "containers": [{"ports": [{"containerPort": 8080, "name": "http"}]}],
                    },
                    "status": {
                        "phase": "Running",
                        "containerStatuses": [
                            {
                                "ready": False,
                                "restartCount": 3,
                                "state": {
                                    "waiting": {
                                        "reason": "CrashLoopBackOff",
                                        "message": "back-off 5m0s",
                                    }
                                },
                            }
                        ],
                    },
                }
            ]
        ),
        "services": listing(
            [
                {
                    "metadata": {"name": "frontend"},
                    "spec": {
                        "selector": {"app": "frontend"},
                        "type": "ClusterIP",
                        "ports": [{"port": 80, "targetPort": 9999}],
                    },
                }
            ]
        ),
        "endpoints": listing([]),
        "configmaps": listing([{"metadata": {"name": "web-config"}}]),
        "secrets": listing([]),
        "events": listing(
            [
                {
                    "type": "Warning",
                    "reason": "BackOff",
                    "count": 7,
                    "lastTimestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                    "involvedObject": {"kind": "Pod", "name": "frontend-5f8-abcde"},
                    "message": "Back-off restarting failed container",
                }
            ]
        ),
        "nodes": listing(
            [
                {
                    "metadata": {"name": "node-1"},
                    "status": {"conditions": [{"type": "Ready", "status": "True"}]},
                },
                {
                    "metadata": {"name": "node-2"},
                    "status": {
                        "conditions": [
                            {"type": "Ready", "status": "False"},
                            {"type": "MemoryPressure", "status": "True"},
                        ]
                    },
                },
            ]
        ),
    }


# The rendering of `populated_reads()` by the prototype's own renderer
# (`agent/tools/topology.py:_render`), captured verbatim. The multiplication sign in
# the event count is part of that text, not a stray glyph.
EXPECTED = """NAMESPACE SNAPSHOT: shop   (window: last 30m; 4 component(s) flagged [!])
Deterministic read-only facts — generic Kubernetes health flags only, NOT a root-cause verdict. The faulty component is among these; localize it yourself.
WORKLOADS  (kind/name · ready/desired · restarts · age · image)
  Deployment/frontend  1/2  rst=3  ?  reg/web:1.2  [!] 1/2 ready  [!] rollout in progress

POD STATE / RESTART HISTORY  (current · restarts · node · reason and time)
  [!] frontend-5f8-abcde  current=Running/ready=False  rst=3  node=node-1  CrashLoopBackOff: back-off 5m0s

SERVICES / BINDINGS  (selector → endpoints · port→targetPort)
  svc/frontend  app=frontend  eps:0  80→9999  [!] 0 endpoints; targetPort 9999 matches no containerPort [8080]

CONFIG REFERENCES  (ConfigMap/Secret ← referencing workloads)
  [!] cm/missing-cm  NOT FOUND  ← Deployment/frontend
  cm/web-config  ← Deployment/frontend

RECENT CHANGES  (Warning events, last 30m)
  [!] BackOff ×7  Pod/frontend-5f8-abcde  Back-off restarting failed container

NODES  (cluster-scoped context)
  node-1  Ready=True
  node-2  Ready=False  [!] Ready=False, MemoryPressure"""  # noqa: RUF001


async def test_the_report_is_the_one_the_prompts_were_written_against(tmp_path: Path) -> None:
    """Pinned against the prototype's renderer, character for character.

    The header, the six section titles, the flag wording and the caps are all shipped to
    the model, and the eval numbers were taken with this exact text in front of it.
    """
    rendered = await namespace_snapshot(
        context(tmp_path, FakeCluster(populated_reads())), namespace="shop"
    )
    assert rendered == EXPECTED


async def test_a_failed_read_is_a_gap_not_an_asserted_absence(tmp_path: Path) -> None:
    """``None`` and ``[]`` are different answers and must not render the same.

    A namespace that references no ConfigMap and one whose workload read failed both
    produce no references. Saying "none found" for the second is a false negative the
    model has no way to see through, and it shipped that way once.
    """
    empty = await namespace_snapshot(
        context(tmp_path, FakeCluster(empty_reads())), namespace="shop"
    )
    assert "  (no ConfigMap/Secret references found)" in empty
    assert "  (no Warning events in window)" in empty
    # NODES is the one section the prototype renders from `or []`, so a successful read
    # of a cluster with no nodes reads as unavailable. Carried as-is: it is the harmless
    # direction of the conflation, and a nodeless cluster is not a state that occurs.
    assert empty.count("  (unavailable)") == 1

    failed = await namespace_snapshot(
        context(tmp_path, FakeCluster(failed_reads())), namespace="shop"
    )
    assert "no ConfigMap/Secret references found" not in failed
    assert "no Warning events in window" not in failed
    # All six sections. The prototype omitted the pod section entirely here,
    # because its outage notice lived inside the workloads branch — so when both
    # reads failed, nothing anywhere said the pod state was unknown.
    assert failed.count("  (unavailable)") == 6


async def test_every_unusable_read_is_named_in_the_log(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A total read outage stayed invisible for months. "(unavailable)" alone is why."""
    with caplog.at_level(logging.WARNING, logger="maver.tools.snapshot"):
        await namespace_snapshot(context(tmp_path, FakeCluster(failed_reads())), namespace="shop")
    named = {record.args[1] for record in caplog.records if record.args}
    assert named == set(_RESOURCE.values())


async def test_the_eight_reads_fan_out_and_an_unusable_one_is_retried_once(
    tmp_path: Path,
) -> None:
    """A discovery endpoint that 404s mid-rollout is exactly what a snapshot meets."""
    cluster = FakeCluster(populated_reads(), fail_once=frozenset({"nodes"}))
    rendered = await namespace_snapshot(context(tmp_path, cluster), namespace="shop")
    assert cluster.calls == [
        "get deployments,statefulsets,daemonsets -n shop -o json",
        "get pods -n shop -o json",
        "get services -n shop -o json",
        "get endpoints -n shop -o json",
        "get configmaps -n shop -o json",
        "get secrets -n shop -o json",
        "get events -n shop -o json",
        "get nodes -o json",
        "get nodes -o json",
    ]
    assert "node-2  Ready=False" in rendered


async def test_a_failed_endpoints_read_does_not_manufacture_a_finding(tmp_path) -> None:
    """A read that failed is not a count of zero.

    Collapsing the two made every selector-bearing Service render `eps:0` and earn a
    `[!] 0 endpoints` — a fabricated fault, in the section whose whole purpose is
    binding correctness, invented from a gap rather than observed. The rest of the
    module already refuses to judge what it cannot see; this read was the exception.
    """
    reads = dict(populated_reads())
    reads["endpoints"] = "error: timed out after 90.0s"
    report = await namespace_snapshot(context(tmp_path, FakeCluster(reads)), namespace="shop")

    assert "0 endpoints" not in report
    assert "eps:?" in report


def all_reads(**overrides: str) -> dict[str, str]:
    """Every read the snapshot makes, empty unless a test cares about one."""
    return {name: listing([]) for name in _RESOURCE.values()} | overrides


async def test_a_termination_is_dated_against_the_incident_boundary(tmp_path: Path) -> None:
    """The boundary was parsed, validated, carried on the scope — and never passed.

    Everything that uses it was live and unreachable: the header line, the comparison,
    and the label on a restart's termination time. The snapshot is the diagnosing role's
    opening message, so without them every old restart in the namespace reads as evidence
    of the incident, which is the failure the split exists to prevent.
    """
    pod = {
        "metadata": {"name": "payment-7f9d", "namespace": "shop"},
        "status": {
            "phase": "Running",
            "containerStatuses": [
                {
                    "name": "payment",
                    "ready": True,
                    "restartCount": 1,
                    "lastState": {
                        "terminated": {
                            "reason": "Error",
                            "exitCode": 1,
                            "finishedAt": "2026-08-30T11:59:00Z",
                        }
                    },
                }
            ],
        },
    }
    transport = FakeCluster(all_reads(pods=listing([pod])))
    ctx = context(tmp_path, transport)
    ctx.scope.namespace = "shop"
    ctx.scope.observation_window_start = "2026-08-30T12:00:00Z"

    rendered = await namespace_snapshot(ctx, "shop")

    assert "Incident observation boundary: 2026-08-30T12:00:00Z" in rendered
    assert "[BEFORE incident boundary]" in rendered


async def test_another_namespace_is_not_dated_against_this_incident(tmp_path: Path) -> None:
    """A boundary belongs to one incident. Labelling another namespace's terminations
    against it would date them by an event that has nothing to do with them."""
    ctx = context(tmp_path, FakeCluster(all_reads()))
    ctx.scope.namespace = "shop"
    ctx.scope.observation_window_start = "2026-08-30T12:00:00Z"

    assert "Incident observation boundary" not in await namespace_snapshot(ctx, "other")
