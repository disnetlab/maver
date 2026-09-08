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
import time
from collections.abc import Awaitable, Callable
from typing import Any

from kira.tools.errors import call_failed
from kira.tools.metrics import _parse_alerts, partition_incident_alerts
from kira.tools.snapshot import _load_items
from kira.tools.traces import (
    _PASSTHROUGH_LIMIT,
    _filter_traces_to_candidates,
    _parse_structured,
    _unparseable_scope_error,
)
from kira.tools.transport.base import CAPABILITIES, ToolTransport

__all__ = [
    "ALERTS",
    "DEPENDENCIES",
    "FOREIGN_SERVICE",
    "KUBECTL_JSON",
    "LOG_COUNTS",
    "LOG_ENTRIES",
    "LOG_METRIC_QUERY",
    "LOG_QUERY",
    "METRIC_QUERY",
    "METRIC_SERIES",
    "NAMESPACE",
    "OPERATIONS",
    "SAMPLE_AT",
    "SAMPLE_YEAR",
    "SERVICE",
    "SERVICES",
    "TRACES",
    "ToolTransportContract",
]

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

FOREIGN_SERVICE = "other-app"
"""A name the same backend knows and this incident does not own."""

SAMPLE_AT = 1_757_000_000
"""One fixed instant every sample and log entry below is stamped with.

A whole second in epoch seconds, so a transport that reads it in the wrong unit lands in
1970 rather than a second away from the right answer.
"""

SAMPLE_YEAR = time.strftime("%Y", time.localtime(SAMPLE_AT))

LOG_QUERY = f'{{namespace="{NAMESPACE}"}}'
LOG_METRIC_QUERY = f'sum by (pod) (count_over_time({{namespace="{NAMESPACE}"}}[1m]))'

LOG_ENTRIES = [
    ("checkout-1", "connection refused talking to cart"),
    ("frontend-2", "upstream request timed out"),
]
"""Two log entries from two pods, as ``(pod, message)``.

Two of them, and from different pods, because the failure this is here for is a
rendering that drops the labels: one entry cannot show that the pods became
indistinguishable.
"""

LOG_COUNTS = [("checkout-1", "17"), ("frontend-2", "3")]
"""The same two pods, counted. LogQL answers ``count_over_time`` over the log endpoint
in a different shape from a log read, and it is a shape the model is invited to ask for."""

METRIC_QUERY = f'up{{namespace="{NAMESPACE}"}}'

METRIC_SERIES = [
    {"metric": {"__name__": "up", "pod": "checkout-1"}, "value": [SAMPLE_AT, "1"]},
    {"metric": {"__name__": "up", "pod": "frontend-2"}, "value": [SAMPLE_AT, "0"]},
]
"""One instant query's series."""

OPERATIONS = ["GET /cart", "POST /checkout"]
"""What the trace backend says one service reports."""

TRACES = [
    {
        "traceID": "abc123",
        "spans": [
            {"spanID": "s1", "operationName": "POST /checkout", "processID": "p1"},
            {"spanID": "s2", "operationName": "GET /elsewhere", "processID": "p2"},
        ],
        "processes": {
            "p1": {"serviceName": SERVICE},
            "p2": {"serviceName": FOREIGN_SERVICE},
        },
    }
]
"""One trace crossing out of the incident, which is the case the span filter exists for."""

DEPENDENCIES = {
    "data": [
        {"parent": SERVICE, "child": "frontend", "callCount": 12},
        {"parent": SERVICE, "child": FOREIGN_SERVICE, "callCount": 3},
    ]
}
"""Two edges, one of them leaving the namespace."""


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
        """The namespace snapshot parses this text as JSON, and a prefix in front of the
        payload turns it into "unavailable" without a word about why: the snapshot comes
        back empty, and the trace scope read through the same path stops being enforced."""
        result = await transport.kubectl(["get", "pods", "-n", NAMESPACE, "-o", "json"])

        assert result == KUBECTL_JSON
        assert _load_items(result) == json.loads(KUBECTL_JSON)["items"]

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

    async def test_log_reads_arrive_one_line_per_entry_with_the_labels_that_place_them(
        self, transport: ToolTransport
    ) -> None:
        """Two readers need that shape, and neither of them says so when it is missing.

        The model attributes an entry from its labels and dates it against the incident
        boundary from its timestamp, and the evidence policy drops a line at a time — so
        a rendering that runs the entries together loses all of them the moment one
        names something the environment hides. The metric form of the same query, which
        the tool's own description invites, is the one that broke: read as a stream it
        rendered bare numbers with the labels gone and every timestamp in 1970, and
        carried no error prefix to say the read had failed.
        """
        for query, expected in ((LOG_QUERY, LOG_ENTRIES), (LOG_METRIC_QUERY, LOG_COUNTS)):
            lines = [line for line in (await transport.query_logs(query)).splitlines() if line]
            assert len(lines) == len(expected)
            for label, body in expected:
                placed = [line for line in lines if label in line and body in line]
                assert len(placed) == 1
                assert SAMPLE_YEAR in placed[0]

    async def test_a_metrics_body_is_structured_enough_to_be_filtered(
        self, transport: ToolTransport
    ) -> None:
        """An evidence policy filters a structured body structurally and any other body
        by dropping the lines that name a hidden term. A metrics rendering that is
        neither JSON nor a Python literal therefore loses every series when one of them
        mentions the load generator, and the model is handed a withheld-result notice
        where the answer to the query it wrote should have been."""
        parsed = _parse_structured(await transport.query_metrics(METRIC_QUERY))

        assert isinstance(parsed, dict)
        series = parsed.get("result")
        assert isinstance(series, list)
        assert [one["metric"]["pod"] for one in series] == [
            one["metric"]["pod"] for one in METRIC_SERIES
        ]

    async def test_traces_reach_the_span_filter_that_narrows_them(
        self, transport: ToolTransport
    ) -> None:
        """A trace crosses application boundaries, so scoping the service it was asked
        for is not enough: the spans inside it are narrowed too. A payload the filter
        cannot parse is refused rather than passed through, so an unreadable rendering
        costs the model the trace tools for the rest of the run — and the operations
        listing, which nothing parses, reaches it exactly as the backend wrote it."""
        listed = await transport.query_traces("operations", service=SERVICE)
        assert all(operation in listed for operation in OPERATIONS)

        raw = await transport.query_traces("traces", service=SERVICE, last_n_minutes=15)
        filtered = _filter_traces_to_candidates(raw, frozenset({SERVICE}))
        assert filtered is not None
        kept = _parse_structured(filtered)
        assert isinstance(kept, list) and len(kept) == 1
        assert [span["operationName"] for span in kept[0]["spans"]] == ["POST /checkout"]
        assert FOREIGN_SERVICE not in str(kept)

    async def test_the_dependency_graph_reaches_the_edge_filter_that_narrows_it(
        self, transport: ToolTransport
    ) -> None:
        """The envelope rather than the edges, because the filter above rewrites the
        edge list in place and hands the whole graph on. An edge with one end outside
        the namespace is dropped there, and a graph that arrives unreadable is refused:
        keeping a half-inside edge would name a service the model has no scoped way to
        look at, so the only thing it could do with the name is follow it out of the
        incident."""
        graph = _parse_structured(await transport.query_traces("dependencies", last_n_minutes=30))

        assert isinstance(graph, dict)
        edges = graph.get("data")
        assert isinstance(edges, list)
        assert [(edge["parent"], edge["child"]) for edge in edges] == [
            (edge["parent"], edge["child"]) for edge in DEPENDENCIES["data"]
        ]

    async def test_a_rollback_answer_agrees_with_the_capability_that_was_declared(
        self, transport: ToolTransport
    ) -> None:
        """The rollback tools are withheld before a run from a transport that does not
        declare the capability, so the only way these are reached at all is a caller
        that ignored it. Answering such a caller with something that reads as success
        would tell an agent a write had been undone by a transport that keeps no journal
        of inverses and cannot undo anything."""
        offered = "kubernetes.rollback" in transport.capabilities

        for answer in (await transport.rollbackable_commands(), await transport.rollback_last()):
            assert answer.strip()
            if not offered:
                assert call_failed(answer)
