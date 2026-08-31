"""Tests for the telemetry tools and the shared error/limit conventions.

Only two things here earn a test. The exact text of a model-facing string, because the
prompts and every published measurement were written against it; and the output limit,
because a tool that pre-truncated would hide the pointer to the saved file.
"""

from pathlib import Path
from typing import Any

import pytest

from kira.audit.journal import Journal
from kira.tools.context import ToolContext
from kira.tools.errors import format_exit_result, tool_error
from kira.tools.logs import get_logs
from kira.tools.metrics import get_metrics
from kira.tools.workspace import SessionWorkspace


class FakeTransport:
    """The transport calls these three modules make, and nothing else."""

    name = "fake"
    capabilities = frozenset({"metrics", "logs", "traces"})
    timeout_seconds = 30

    def __init__(self, payload: str = "") -> None:
        self.payload = payload
        self.log_queries: list[str] = []

    async def query_metrics(self, query: str, *, window_minutes: int | None = None) -> str:
        return self.payload

    async def query_logs(self, query: str, *, last_n_minutes: int = 15) -> str:
        self.log_queries.append(query)
        return self.payload


class BlindingEvidence:
    """An adapter-supplied policy that rewrites the query rather than the result."""

    def filter_query(self, query: str) -> str:
        return f"{query} !chaos-mesh"

    def filter_result(self, content: str) -> str:
        return content


def context(tmp_path: Path, transport: Any, evidence: Any = None) -> ToolContext:
    kwargs = {"evidence": evidence} if evidence is not None else {}
    return ToolContext(
        transport=transport,
        workspace=SessionWorkspace(tmp_path),
        role="rca",
        session_id="s1",
        journal=Journal(session_id="s1", stage="rca"),
        **kwargs,
    )


def test_a_failure_is_a_return_value_the_loop_can_recognise() -> None:
    """The success heuristic, the repeat breaker and the snapshot all read this prefix.

    A tool that raised, or that said "Error:", would be counted as a successful turn.
    """
    assert tool_error("kubectl 'get' is not permitted for the agent").startswith("error: ")


@pytest.mark.parametrize(
    ("returncode", "expected"),
    [
        (
            0,
            "exit_code=0 (no output)\n"
            "The command ran and printed nothing. Re-running it returns this same "
            "empty result. If you expected output, the path, resource, or selector "
            "you named matched nothing — widen the query or check one level up.",
        ),
        (
            1,
            "exit_code=1 (no output)\n"
            "The command failed silently. Re-running it returns this same empty result. "
            "Run it again with stderr shown, or check that the command and its inputs exist.",
        ),
    ],
)
def test_a_silent_command_is_named_as_silent(returncode: int, expected: str) -> None:
    """An empty result gives a greedy decoder nothing to react to, so it re-runs it.

    ``bash_exec`` and ``code_exec`` account for 70% of all repeat-breaker trips, and a
    bare ``exit_code=0`` was the single most repeated tool result in the VM3 A/B.
    """
    assert format_exit_result(returncode, "   \n") == expected
    assert format_exit_result(returncode, "hello") == f"exit_code={returncode}\nhello"


async def test_a_telemetry_result_over_the_limit_keeps_its_pointer(tmp_path: Path) -> None:
    """The tool must not pre-truncate: 16,384 characters, then the saved-file pointer.

    Trimming before the workspace sees the payload would drop the pointer, and the
    model would read a silently shortened result as the whole answer.
    """
    payload = "x" * 20_000
    result = await get_metrics(context(tmp_path, FakeTransport(payload)), query="up")

    assert result.startswith("x" * 16_384)
    assert f"[Output truncated ({len(payload)} chars). Full output saved to " in result
    assert (tmp_path / "tool_outputs" / "prometheus_metrics_0001.log").read_text() == payload


async def test_the_query_is_blinded_before_it_leaves_the_process(tmp_path: Path) -> None:
    """Filtering the result is not enough; the question itself has to be rewritten.

    A model that can ask about a hidden namespace learns it exists from the answer's
    shape, and the injector has leaked into served telemetry before.
    """
    transport = FakeTransport("no logs")
    ctx = context(tmp_path, transport, BlindingEvidence())

    await get_logs(ctx, query='{namespace="astronomy-shop"}')

    assert transport.log_queries == ['{namespace="astronomy-shop"} !chaos-mesh']
