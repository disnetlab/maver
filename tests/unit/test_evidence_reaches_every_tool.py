"""The hide list reaches every result, not the one tool that remembered to apply it.

An environment can leak its own fault injection into the telemetry it serves, which turns
a diagnosis into a lookup. Filtering it is the adapter's job, and applying that filter was
each tool's — so it was applied to kubectl, to a shell command and to a host command, and
to none of the nine telemetry tools. The policy's own description names a chaos
injector's labels in the metrics an agent queries as the thing it exists to remove.

Under the benchmark transport the harness proxy filters these paths too, so this is a
second line of defence there. Under the direct transport there is no proxy and it was the
only one.
"""

import pathlib

import pytest

from kira.audit.journal import Journal, ListSink
from kira.tools.context import ToolContext
from kira.tools.logs import get_label_values, get_labels, get_logs
from kira.tools.metrics import get_alerts, get_metrics
from kira.tools.traces import get_dependency_graph, get_operations, get_services, get_traces
from kira.tools.workspace import SessionWorkspace

MARKER = "chaos-mesh-must-not-be-seen"


class ShoutingPolicy:
    """Replaces everything, so a result that was not filtered is obvious."""

    def filter_result(self, content: str) -> str:
        return "[filtered]"

    def filter_query(self, query: str) -> str:
        return query


class Transport:
    name = "fake"
    capabilities = frozenset({"kubernetes", "metrics", "logs", "traces"})
    timeout_seconds = 30

    async def kubectl(self, argv: list[str]) -> str:
        return MARKER

    async def query_metrics(self, query: str, *, window_minutes: int | None = None) -> str:
        return MARKER

    async def query_alerts(self) -> str:
        return MARKER

    async def query_logs(self, query: str, *, last_n_minutes: int = 15) -> str:
        return MARKER

    async def log_labels(self) -> str:
        return MARKER

    async def log_label_values(self, label: str) -> str:
        return MARKER

    async def query_traces(self, operation: str, **kwargs: object) -> str:
        # The service catalogue is what the scope check reads, so it has to parse. The
        # marker rides in a service name, which is exactly where a leak would.
        if operation == "services":
            return f'["user", "{MARKER}"]'
        return MARKER


def context(tmp_path: pathlib.Path) -> ToolContext:
    return ToolContext(
        transport=Transport(),
        workspace=SessionWorkspace(tmp_path),
        role="rca",
        session_id="s1",
        journal=Journal(ListSink(), session_id="s1", stage="rca"),
        evidence=ShoutingPolicy(),
    )


TELEMETRY = [
    ("get_metrics", lambda ctx: get_metrics(ctx, query="up")),
    ("get_alerts", lambda ctx: get_alerts(ctx)),
    ("get_logs", lambda ctx: get_logs(ctx, query='{app="user"}', last_n_minutes=5)),
    ("get_labels", lambda ctx: get_labels(ctx)),
    ("get_label_values", lambda ctx: get_label_values(ctx, label="app")),
    ("get_services", lambda ctx: get_services(ctx)),
    ("get_operations", lambda ctx: get_operations(ctx, service="user")),
    ("get_traces", lambda ctx: get_traces(ctx, service="user", last_n_minutes=5)),
    ("get_dependency_graph", lambda ctx: get_dependency_graph(ctx)),
]


@pytest.mark.parametrize(("name", "call"), TELEMETRY, ids=[n for n, _ in TELEMETRY])
async def test_a_telemetry_result_passes_through_the_evidence_policy(
    tmp_path: pathlib.Path, name: str, call: object
) -> None:
    """A tool that refuses before it reads is fine; one that returns unfiltered is not."""
    result = str(await call(context(tmp_path)))  # type: ignore[operator]

    if result.startswith("error:"):
        pytest.skip(f"{name} refused before reading, so there was nothing to filter")
    assert MARKER not in result


class RewritingPolicy:
    """A policy that actually rewrites, which neither shipped one does."""

    def filter_result(self, content: str) -> str:
        return content

    def filter_query(self, query: str) -> str:
        return query.replace("chaos-mesh", "REDACTED")


class RefusingPolicy(RewritingPolicy):
    def filter_query(self, query: str) -> str:
        raise RuntimeError("the policy could not scope this")


async def test_a_metric_query_is_scoped_before_it_is_sent(tmp_path: pathlib.Path) -> None:
    """The two tools whose *request* can name a hidden place are the log query and the
    metric query, and only the log query was scoped.

    A PromQL selector can name a namespace, and a model that can ask about a hidden one
    learns it exists from the shape of the answer — which no amount of filtering the
    result afterwards undoes. Neither shipped policy rewrites a query, so this is inert
    today; the point is that when one does, it reaches both.
    """
    sent: list[str] = []

    class Recording(Transport):
        async def query_metrics(self, query: str, *, window_minutes: int | None = None) -> str:
            sent.append(query)
            return "{}"

    ctx = context(tmp_path)
    object.__setattr__(ctx, "transport", Recording())
    object.__setattr__(ctx, "evidence", RewritingPolicy())

    await get_metrics(ctx, query='up{namespace="chaos-mesh"}')

    assert sent == ['up{namespace="REDACTED"}']


async def test_a_query_that_could_not_be_scoped_is_refused(tmp_path: pathlib.Path) -> None:
    """Fail closed, as the log query does: an unrewritten query reaches the place the
    environment meant to hide."""
    ctx = context(tmp_path)
    object.__setattr__(ctx, "evidence", RefusingPolicy())

    assert (await get_metrics(ctx, query="up")).startswith("error:")
