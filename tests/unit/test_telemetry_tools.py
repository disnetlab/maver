"""Tests for the telemetry tools and the shared error/limit conventions.

Only two things here earn a test. The exact text of a model-facing string, because the
prompts and every published measurement were written against it; and the output limit,
because a tool that pre-truncated would hide the pointer to the saved file.
"""

import asyncio
import os
import signal
import stat
from pathlib import Path
from typing import Any

import pytest

from maver.audit.journal import Journal
from maver.settings.groups import ToolSettings
from maver.tools import host
from maver.tools.context import ToolContext
from maver.tools.errors import call_timed_out, format_exit_result, timed_out, tool_error
from maver.tools.host import ssh_exec
from maver.tools.logs import get_logs
from maver.tools.metrics import get_metrics
from maver.tools.workspace import SessionWorkspace


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


async def test_a_bounded_tool_reports_its_timeout_through_the_shared_builder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A timeout is a harness intervention, and the loop finds one by reading the result.

    That is why the marker has one builder. ``ssh_exec`` spelled the sentence out by
    hand instead, so a wording the loop reads and a wording it does not were the same
    edit apart, with nothing between them: the interventions of every run would have
    lost their timeout bucket and no test would have said so.
    """

    class Hangs:
        # A positive sentinel matches the subprocess contract. ``killpg`` is mocked
        # below, so this test never sends a signal to a real process group.
        pid = 424_242

        async def communicate(self) -> tuple[bytes, bytes]:
            await asyncio.sleep(10)
            raise AssertionError("the bound should have fired")

        async def wait(self) -> int:
            """Nothing was started."""
            return -1

    async def spawn(*argv: object, **kwargs: object) -> Hangs:
        return Hangs()

    def no_such_process(pid: int, sig: int) -> None:
        assert (pid, sig) == (Hangs.pid, signal.SIGKILL)
        raise ProcessLookupError

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(os, "killpg", no_such_process)
    monkeypatch.setattr(host, "_TIMEOUT_SECONDS", 0.05)
    ctx = ToolContext(
        transport=FakeTransport(),
        workspace=SessionWorkspace(tmp_path),
        role="execution",
        session_id="s1",
        journal=Journal(session_id="s1", stage="execution"),
        settings=ToolSettings(host_recovery_nodes=frozenset({"worker1"})),
    )

    result = await ssh_exec(ctx, node="worker1", command="systemctl start kubelet")

    assert call_timed_out(result)
    assert result == timed_out("ssh_exec", host._TIMEOUT_SECONDS)


async def test_a_hung_ssh_takes_what_it_started_with_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bound has to end the call's process tree, not only its direct child.

    ssh's own children — a ProxyCommand, a control master, whatever a wrapper on the
    PATH started — inherit the pipes, and killing ssh alone leaves them holding the
    incident's file descriptors and still running after the model has been told the call
    was abandoned on time. This is the treatment the direct transport's kubectl gets,
    and the two subprocess call sites should not differ on it.

    The stand-in for ssh forks before it hangs, waits well past the bound, and leaves
    the evidence in a file, so what is asserted is that the child died with its parent.
    """
    marker = tmp_path / "the-child-outlived-its-parent"
    fake = tmp_path / "ssh"
    fake.write_text(f"#!/bin/sh\n( sleep 1.3; touch {marker} ) &\nsleep 30\n")
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setattr(host, "_TIMEOUT_SECONDS", 0.8)
    ctx = ToolContext(
        transport=FakeTransport(),
        workspace=SessionWorkspace(tmp_path),
        role="execution",
        session_id="s1",
        journal=Journal(session_id="s1", stage="execution"),
        settings=ToolSettings(host_recovery_nodes=frozenset({"worker1"})),
    )

    result = await ssh_exec(ctx, node="worker1", command="systemctl start kubelet")
    await asyncio.sleep(1.2)

    assert call_timed_out(result)
    assert not marker.exists()
