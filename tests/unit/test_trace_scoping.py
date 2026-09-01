"""Tests for namespace scoping of the trace tools.

A trace backend is shared across applications and its service names carry no namespace,
so everything that keeps one incident's evidence out of another's lives in this module.
What earns a test is the refusal text, which the prompts were written against; the two
directions the module can fail in, closed on a missing inventory and open on a known-safe
upstream error; the silent-workload line, which exists because its absence let an agent
clear a live incident; and the cache, because a per-call inventory read would put a
kubectl in front of every trace query.
"""

import json
from pathlib import Path
from typing import Any

from kira.audit.journal import Journal
from kira.tools.context import ToolContext
from kira.tools.scope import IncidentScope
from kira.tools.traces import get_operations, get_services, get_traces
from kira.tools.workspace import SessionWorkspace

_GLOBAL_CATALOGUE = json.dumps(["frontend", "other-app-checkout"])
"""What the shared backend answers with: one of this namespace's services, and one that
belongs to another application entirely."""


class FakeBackend:
    """One inventory read and one trace query, both canned, both counted."""

    name = "fake"
    capabilities = frozenset({"kubernetes", "traces"})
    timeout_seconds = 30

    def __init__(self, inventory: str, traces: str = _GLOBAL_CATALOGUE) -> None:
        self.inventory = inventory
        self.traces = traces
        self.kubectl_calls: list[str] = []

    async def kubectl(self, argv: list[str]) -> str:
        self.kubectl_calls.append(" ".join(argv))
        return self.inventory

    async def query_traces(self, operation: str, **kwargs: Any) -> str:
        return self.traces


def context(tmp_path: Path, transport: Any, namespace: str | None = "shop") -> ToolContext:
    return ToolContext(
        transport=transport,
        workspace=SessionWorkspace(tmp_path),
        role="rca",
        session_id="s1",
        journal=Journal(session_id="s1", stage="rca"),
        scope=IncidentScope(namespace=namespace),
    )


def inventory(*names: str) -> str:
    return json.dumps({"items": [{"metadata": {"name": name}} for name in names]})


async def test_a_service_outside_the_namespace_is_refused(tmp_path: Path) -> None:
    """The refusal names the allowed set, so the model can correct itself in one turn."""
    ctx = context(tmp_path, FakeBackend(inventory("frontend", "payments")))

    result = await get_operations(ctx, service="other-app-checkout")

    assert result == (
        "error: Jaeger service 'other-app-checkout' is outside incident namespace "
        "'shop'. Allowed Kubernetes service/workload names: frontend, payments"
    )


async def test_an_unreadable_inventory_closes_rather_than_opens(tmp_path: Path) -> None:
    """Scoping that could not be computed was not enforced, so nothing may be served.

    Falling open here would hand back the backend's global catalogue, which is exactly
    the cross-application evidence the scope exists to withhold.
    """
    backend = FakeBackend("error: couldn't get current server API group list")
    ctx = context(tmp_path, backend)

    result = await get_services(ctx)

    assert result == (
        "error: unable to enforce Jaeger namespace scope for 'shop': "
        "Kubernetes service/workload inventory is unavailable"
    )
    assert "other-app-checkout" not in result


async def test_an_upstream_error_is_not_masked_as_a_scope_error(tmp_path: Path) -> None:
    """A payload that is already a failure has nothing to leak and says what went wrong.

    Masking it made the agent guess at a namespace problem when the truth was a failed
    query.
    """
    backend = FakeBackend(inventory("frontend"), traces="error: Jaeger query failed: 503")
    ctx = context(tmp_path, backend)

    assert await get_traces(ctx, service="frontend", last_n_minutes=5) == (
        "error: Jaeger query failed: 503"
    )

    backend.traces = "<html>gateway timeout</html>"
    assert await get_traces(ctx, service="frontend", last_n_minutes=5) == (
        "error: unable to enforce Jaeger namespace scope: Jaeger returned unparseable traces"
    )


async def test_silent_workloads_are_named_only_when_some_are_silent(tmp_path: Path) -> None:
    """A collapsed reporting list reads as an inventory unless the gap is spelled out.

    An agent read a one-element list for an eight-service namespace as "this namespace
    has one service" and cleared the incident.
    """
    ctx = context(tmp_path, FakeBackend(inventory("frontend", "payments")))

    with_silence = await get_services(ctx)

    assert with_silence == (
        "Services reporting traces in namespace 'shop': ['frontend']\n"
        "Workloads in 'shop' with NO traces in this window: ['payments']. "
        "This is not an inventory of what exists — these workloads are deployed but "
        "silent. A service that stopped serving stops reporting, so treat silence as "
        "a symptom to investigate, not as absence."
    )

    quiet = context(tmp_path, FakeBackend(inventory("frontend")))
    assert await get_services(quiet) == (
        "Services reporting traces in namespace 'shop': ['frontend']"
    )


async def test_the_inventory_is_read_once_per_run(tmp_path: Path) -> None:
    """Every trace call needs the allowed set, and it cannot change within a run."""
    backend = FakeBackend(inventory("frontend"))
    ctx = context(tmp_path, backend)

    await get_services(ctx)
    await get_operations(ctx, service="frontend")
    await get_traces(ctx, service="frontend", last_n_minutes=5)

    assert backend.kubectl_calls == [
        "get services,deployments,statefulsets,daemonsets -n shop -o json"
    ]
    assert ctx.scope.trace_service_candidates == frozenset({"frontend"})
