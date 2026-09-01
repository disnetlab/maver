"""Incident scoping on the alert query.

The renderings are pinned to `tests/fixtures/prototype_alert_partitioning.json`, produced
by running the prototype's own `_partition_incident_alerts` over each payload. Prompts
were written against these strings, so the expectation is its output rather than this
implementation's.
"""

import json
import pathlib
from typing import Any

import pytest

from kira.audit.journal import Journal, ListSink
from kira.settings.groups import ToolSettings
from kira.tools.context import ToolContext
from kira.tools.metrics import get_alerts, partition_incident_alerts
from kira.tools.scope import IncidentScope
from kira.tools.workspace import SessionWorkspace

FIXTURE = json.loads(
    (
        pathlib.Path(__file__).resolve().parents[1]
        / "fixtures"
        / "prototype_alert_partitioning.json"
    ).read_text(encoding="utf-8")
)

STALE_NS = "gone-ns"
ALERTS = repr(
    [
        {"labels": {"alertname": "HighErrorRate", "namespace": "shop"}},
        {"labels": {"alertname": "OldAlert", "namespace": STALE_NS}},
    ]
)


class FakeTransport:
    """Answers the alert query with a fixed payload and the namespace read as told."""

    name = "fake"
    capabilities = frozenset({"metrics", "kubernetes"})
    timeout_seconds = 30

    def __init__(self, alerts: str = ALERTS, namespace_read: str = "") -> None:
        self.alerts = alerts
        self.namespace_read = namespace_read
        self.reads: list[list[str]] = []

    async def query_alerts(self) -> str:
        return self.alerts

    async def kubectl(self, argv: list[str]) -> str:
        self.reads.append(list(argv))
        return self.namespace_read


def context(tmp_path: pathlib.Path, transport: FakeTransport, *, annotate: bool) -> ToolContext:
    return ToolContext(
        transport=transport,
        workspace=SessionWorkspace(tmp_path),
        role="rca",
        session_id="s1",
        journal=Journal(ListSink(), session_id="s1", stage="rca"),
        scope=IncidentScope(namespace="shop"),
        settings=ToolSettings(annotate_stale_namespaces=annotate),
    )


@pytest.mark.parity
@pytest.mark.parametrize("case", sorted(FIXTURE), ids=str)
def test_the_partitioning_matches_the_prototype(case: str) -> None:
    """Nothing is dropped and every bucket is labelled: a foreign alert can be a real
    dependency signal, so removing it hides evidence, while presenting it unlabelled
    invites the model to adopt it as its own."""
    namespace = None if case == "no_scope" else "shop"
    assert partition_incident_alerts(namespace, FIXTURE[case]["raw"]) == FIXTURE[case]["rendered"]


async def test_a_missing_namespace_marks_its_alerts_stale(tmp_path: pathlib.Path) -> None:
    """An environment that tears its namespaces down between problems leaves alerts
    behind, and the next run inherits them as apparently live evidence."""
    transport = FakeTransport(namespace_read='Error from server (NotFound): namespaces "x"')
    result = await get_alerts(context(tmp_path, transport, annotate=True))

    assert "STALE firing alerts" in result
    assert "is absent from the Kubernetes API" in result


@pytest.mark.parametrize(
    "response",
    [
        "error: timed out waiting for the cluster",
        'Error from server (Forbidden): namespaces "gone-ns" is forbidden',
    ],
)
async def test_only_an_explicit_not_found_proves_a_namespace_is_gone(
    tmp_path: pathlib.Path, response: str
) -> None:
    """A timeout or an RBAC refusal says nothing about whether the namespace exists.
    Treating either as proof would mark a live incident's own alerts stale."""
    transport = FakeTransport(namespace_read=response)
    result = await get_alerts(context(tmp_path, transport, annotate=True))

    assert "STALE firing alerts" not in result


async def test_the_annotation_is_off_by_default_and_costs_nothing(
    tmp_path: pathlib.Path,
) -> None:
    """One cluster read per distinct alert namespace, on every alert query."""
    transport = FakeTransport()
    await get_alerts(context(tmp_path, transport, annotate=False))

    assert transport.reads == []


async def test_an_unreadable_payload_survives_both_stages_unchanged(
    tmp_path: pathlib.Path,
) -> None:
    """A parser that fell back to an empty list would hide every firing alert and
    report a quiet cluster, so an unrecognised shape is passed through instead."""
    raw = "prometheus is unreachable"
    transport = FakeTransport(alerts=raw, namespace_read="NotFound")
    assert await get_alerts(context(tmp_path, transport, annotate=True)) == raw


def test_a_stale_alert_in_the_incident_namespace_is_still_stale() -> None:
    """Bucket order: the staleness test comes before the namespace match, or an alert
    left over from the previous occupant of this namespace reads as this incident's."""
    alerts: list[dict[str, Any]] = [
        {"labels": {"namespace": "shop"}, "stale_namespace": "namespace is gone"}
    ]
    rendered = partition_incident_alerts("shop", repr(alerts))

    assert "No firing alerts scoped to incident namespace 'shop'." in rendered
    assert "STALE firing alerts" in rendered
