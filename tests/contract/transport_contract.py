"""The contract every transport has to satisfy, whatever it talks to.

A transport is done when it passes this unmodified. Subclass :class:`ToolTransportContract`
in a ``test_*.py`` module and supply two fixtures:

    class TestMyTransport(ToolTransportContract):
        @pytest.fixture
        def transport(self): ...     # backend answering with the payloads below
        @pytest.fixture
        def unreachable(self): ...   # the same implementation, backend gone

Two objects because the properties split that way: what a transport returns when its
backend answers is a different question from what it returns when the backend is not
there, and no one instance can be both at once. Both stub at that implementation's own
I/O seam, a subprocess or an HTTP client or a client session, so nothing here reaches a
cluster, a network or a database.

The payloads are the contract's rather than each subclass's, and each implementation
serves them in whatever encoding its own backend natively produces. The assertions are
then made through the consumers in ``kira.tools`` that have to read the result, because
the risk this suite exists for is divergence: prompts are written against the strings a
transport returns, so tool output is part of the interface, and a transport that renders
something the layer above cannot read fails silently in the middle of an incident rather
than at import.

What is deliberately not asserted: that two transports return the same bytes. They talk
to different clusters. Only that each returns something the layer above can consume.
"""

import json
from collections.abc import Awaitable, Callable
from typing import Any

from kira.tools.metrics import _parse_alerts, partition_incident_alerts
from kira.tools.snapshot import _load_items
from kira.tools.traces import (
    _PASSTHROUGH_LIMIT,
    _candidate_names_from_inventory,
    _parse_structured,
    _unparseable_scope_error,
)
from kira.tools.transport.base import CAPABILITIES, ToolTransport

__all__ = ["ALERTS", "KUBECTL_JSON", "NAMESPACE", "SERVICE", "SERVICES", "ToolTransportContract"]

NAMESPACE = "shop"
SERVICE = "checkout"

KUBECTL_JSON = json.dumps({"items": [{"metadata": {"name": SERVICE, "labels": {"app": SERVICE}}}]})
"""One namespace read, as ``kubectl get -o json`` writes it.

Carries no surrounding whitespace, so a transport returning the command's own output can
be compared against it byte for byte. A real kubectl ends its output with a newline and
one of the two transports strips it; the comparison is about what was *added*, not about
that.
"""

ALERTS = [
    {
        "labels": {"alertname": "PodCrashLooping", "namespace": NAMESPACE},
        "annotations": {"summary": "checkout is restarting"},
        "state": "firing",
    },
    {
        "labels": {"alertname": "HighLatency", "namespace": "other"},
        "annotations": {"summary": "p99 above budget"},
        "state": "firing",
    },
]
"""Two firing alerts, one of them another namespace's.

The foreign one is what makes the partitioning downstream produce its labelled sections,
so the contract can watch the payload survive all the way to the model rather than only
watching a parser accept it.
"""

SERVICES = [SERVICE, "frontend"]
"""What the trace backend says is reporting spans."""


async def _every_call(transport: ToolTransport) -> list[tuple[str, Any]]:
    """Call every method the protocol declares, keeping whatever each one produced.

    Args:
        transport: The transport under test.

    Returns:
        Each call's label paired with its result, or with the exception it raised. An
        exception is recorded rather than propagated because "it raised" is one of the
        answers the properties here are about.
    """
    calls: list[tuple[str, Callable[[], Awaitable[str]]]] = [
        ("kubectl", lambda: transport.kubectl(["get", "pods", "-n", NAMESPACE])),
        ("rollbackable_commands", transport.rollbackable_commands),
        ("rollback_last", transport.rollback_last),
        ("query_metrics", lambda: transport.query_metrics("up")),
        ("query_alerts", transport.query_alerts),
        ("log_labels", transport.log_labels),
        ("log_label_values", lambda: transport.log_label_values("namespace")),
        ("query_logs", lambda: transport.query_logs(f'{{namespace="{NAMESPACE}"}}')),
        ("query_traces(services)", lambda: transport.query_traces("services")),
        (
            "query_traces(operations)",
            lambda: transport.query_traces("operations", service=SERVICE),
        ),
        (
            "query_traces(traces)",
            lambda: transport.query_traces("traces", service=SERVICE, last_n_minutes=15),
        ),
        (
            "query_traces(dependencies)",
            lambda: transport.query_traces("dependencies", last_n_minutes=30),
        ),
    ]

    produced: list[tuple[str, Any]] = []
    for label, call in calls:
        try:
            produced.append((label, await call()))
        except Exception as exc:
            produced.append((label, exc))
    return produced


class ToolTransportContract:
    """Properties a second transport could plausibly get wrong."""

    def test_it_satisfies_the_protocol(self, transport: ToolTransport) -> None:
        assert isinstance(transport, ToolTransport)
        assert isinstance(transport.name, str) and transport.name

    def test_declared_capabilities_are_in_the_vocabulary(self, transport: ToolTransport) -> None:
        """A misspelt capability silently disables its tool everywhere.

        The tool is dropped from every role's toolset before the run starts, and the
        absence reads in the results exactly like a model that chose not to call it.
        """
        assert transport.capabilities <= CAPABILITIES

    def test_the_timeout_is_a_positive_whole_number_of_seconds(
        self, transport: ToolTransport
    ) -> None:
        """The kubectl tool subtracts from it to bound a blocking wait just under it."""
        timeout = transport.timeout_seconds
        assert isinstance(timeout, int) and not isinstance(timeout, bool)
        assert timeout > 0

    async def test_no_call_raises_when_the_backend_is_gone(
        self, unreachable: ToolTransport
    ) -> None:
        """The layer above turns a string into a tool result, and every caller that
        reads one defends itself against an exception rather than expecting one. Those
        defences are the reason this property exists rather than a substitute for it:
        each of them replaces the backend's own complaint with an exception's repr, and
        the backend's complaint is the one thing that lets a model repair a query
        instead of reissuing it. The auto-injected snapshot records a failure and
        returns nothing at all, and the health gate falls open."""
        raised = [
            (label, result)
            for label, result in await _every_call(unreachable)
            if not isinstance(result, str)
        ]
        assert raised == []

    async def test_a_backend_that_cannot_answer_is_legible_as_a_failure(
        self, unreachable: ToolTransport
    ) -> None:
        """Six things read the ``error:`` prefix and nothing reads anything else.

        The loop's success heuristic, the repeat breaker, the namespace snapshot's
        read-failure detection and the trace tools' passthrough screen are four of them,
        so a failure that arrives without the prefix is counted as a successful turn by
        every one of them.
        """
        results = await _every_call(unreachable)

        unmarked = [
            label
            for label, result in results
            if not (isinstance(result, str) and result.startswith("error:"))
        ]
        assert unmarked == []

        # Read back through a consumer rather than trusting the prefix on its own. The
        # trace tools hand a recognised failure to the model verbatim and mask anything
        # else as a scope error, so a failure that lost its marker reaches the model as
        # a namespace problem it has no way to fix.
        masked = [
            label
            for label, result in results
            if _unparseable_scope_error("an unparseable catalogue", result)
            != result.strip()[:_PASSTHROUGH_LIMIT]
        ]
        assert masked == []

    async def test_a_successful_kubectl_returns_its_own_output_unwrapped(
        self, transport: ToolTransport
    ) -> None:
        """Two consumers parse this text as JSON, and a prefix in front of the payload
        turns both into "unavailable" without a word about why: the namespace snapshot
        comes back empty and trace scoping silently stops being enforced."""
        result = await transport.kubectl(["get", "pods", "-n", NAMESPACE, "-o", "json"])

        assert result == KUBECTL_JSON
        assert _load_items(result) == json.loads(KUBECTL_JSON)["items"]
        assert _candidate_names_from_inventory(result) == frozenset({SERVICE})

    async def test_firing_alerts_reach_the_partitioning_that_scopes_them(
        self, transport: ToolTransport
    ) -> None:
        """A payload the partitioning cannot read is returned byte for byte, so another
        namespace's firing alert reaches the model as this incident's own evidence with
        nothing to say it happened."""
        raw = await transport.query_alerts()

        parsed = _parse_alerts(raw)
        assert parsed is not None
        assert [alert["labels"]["alertname"] for alert in parsed] == [
            "PodCrashLooping",
            "HighLatency",
        ]

        partitioned = partition_incident_alerts(NAMESPACE, raw)
        assert partitioned.startswith(
            f"INCIDENT-SCOPED firing alerts for namespace {NAMESPACE!r}: "
        )
        assert "OUT-OF-SCOPE firing alerts from other namespaces" in partitioned

    async def test_the_service_catalogue_reaches_the_scoping_that_narrows_it(
        self, transport: ToolTransport
    ) -> None:
        """The catalogue is intersected with the incident namespace's own workloads. A
        rendering the trace tool cannot parse is refused rather than passed through, so
        the model loses the trace tools for the rest of the run."""
        raw = await transport.query_traces("services")

        services = _parse_structured(raw)
        assert isinstance(services, list)
        assert all(isinstance(name, str) for name in services)
        assert set(services) == set(SERVICES)
