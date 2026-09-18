"""Real bound tools retain honest evidence through the loop's dispatch boundary."""

import asyncio
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from test_loop import SUBMIT, FakeModel, call, spec, turn
from test_snapshot_tool import FakeCluster, empty_reads

from maver.audit.context import current_call
from maver.audit.durable_sink import DurableAuditSink
from maver.audit.journal import Journal, ListSink
from maver.contracts.events import ToolCallEvent, ToolCallStartedEvent
from maver.middleware.authority import DenyAll
from maver.middleware.base import MiddlewareStack
from maver.middleware.write_gate import WriteGate
from maver.roles.loop import run_until_submit
from maver.roles.policy import LoopLimits
from maver.settings.authority import AuthoritySettings
from maver.tools.auditing import ToolCallRecorder
from maver.tools.context import ToolContext
from maver.tools.registry import BoundTool, registered
from maver.tools.workspace import SessionWorkspace


class Transport:
    name = "observed-test"
    capabilities = frozenset({"metrics", "kubernetes"})
    timeout_seconds = 30

    def __init__(self, content: str = "ok") -> None:
        self.content = content
        self.calls: list[Any] = []

    async def kubectl(self, argv: list[str]) -> str:
        self.calls.append(argv)
        return self.content

    async def query_metrics(self, query: str) -> str:
        self.calls.append(query)
        return self.content


def context(tmp_path: Path, transport: Any, *, session: str = "incident", role: str = "rca"):
    sink = ListSink()
    durable = DurableAuditSink(sink, audit_dir=tmp_path / session / "audit")
    ctx = ToolContext(
        transport=transport,
        workspace=SessionWorkspace(tmp_path / session / "workspace"),
        role=role,
        session_id=session,
        journal=Journal(durable, session_id=session, stage=role, run_id=f"{session}-run"),
    )
    return ctx, sink, durable.files


def bound(ctx: ToolContext, name: str) -> BoundTool:
    return BoundTool(registered()[name], ctx)


def pairs(sink: ListSink) -> tuple[list[ToolCallStartedEvent], list[ToolCallEvent]]:
    starts = [
        event.payload for event in sink.events if isinstance(event.payload, ToolCallStartedEvent)
    ]
    ends = [event.payload for event in sink.events if isinstance(event.payload, ToolCallEvent)]
    assert {item.invocation_id for item in starts} == {item.invocation_id for item in ends}
    assert len(starts) == len(ends)
    return starts, ends


async def run(ctx: ToolContext, calls: list[Any], middleware: MiddlewareStack | None = None):
    names = {item.name for item in calls}
    return await run_until_submit(
        model=FakeModel([turn(*calls), turn(call(SUBMIT, {}, id="submit"))]),
        role=ctx.role,
        system="test",
        tools=[spec(name) for name in sorted(names | {SUBMIT})],
        dispatch={name: bound(ctx, name) for name in names if name in registered()},
        submit_tool=SUBMIT,
        messages=[],
        journal=ctx.journal,
        session_id=ctx.session_id,
        middleware=middleware,
        limits=LoopLimits(max_turns=2, finalizer_rescues=0),
    )


async def test_parallel_same_tool_keeps_each_full_raw_and_filtered_output(tmp_path: Path) -> None:
    class Concurrent(Transport):
        async def query_metrics(self, query: str) -> str:
            self.calls.append(query)
            if len(self.calls) == 2:
                ready.set()
            await ready.wait()
            return query + " PRIVATE" * 4_000

    class Evidence:
        def filter_query(self, query: str) -> str:
            return query + " scoped"

        def filter_result(self, content: str) -> str:
            return content.replace("PRIVATE", "HIDDEN")

    ready = asyncio.Event()
    transport = Concurrent()
    ctx, sink, files = context(tmp_path, transport)
    ctx = replace(ctx, evidence=Evidence())
    result = await run(ctx, [call("get_metrics", {"query": key}, id=key) for key in ("a", "b")])
    starts, ends = pairs(sink)
    assert len(starts) == 2
    assert sorted(transport.calls) == ["a scoped", "b scoped"]
    replies = {
        message["tool_call_id"]: message["content"]
        for message in result.transcript
        if message["role"] == "tool"
    }
    for event in ends:
        raw = event.call_id + " scoped" + " PRIVATE" * 4_000
        filtered = raw.replace("PRIVATE", "HIDDEN")
        assert event.status == "completed" and event.succeeded and event.truncated
        assert event.transport == transport.name and event.full_result_chars == len(filtered)
        assert event.duration_ms is not None and event.duration_ms >= 0
        assert files.resolve(event.result)["content"] == replies[event.call_id]
        assert replies[event.call_id].startswith(filtered[:16_384])
        observations = [files.resolve(ref) for ref in event.output_artifacts]
        assert any(value.get("result") == raw for value in observations)
        assert any(value.get("content") == filtered for value in observations)
        assert "PRIVATE" not in replies[event.call_id]
    assert current_call() is None


async def test_partial_snapshot_keeps_unknown_coverage_and_each_failed_read(tmp_path: Path) -> None:
    reads = empty_reads()
    reads["pods"] = "error: Kubernetes API unavailable"
    ctx, sink, files = context(tmp_path, FakeCluster(reads))
    returned = await bound(ctx, "namespace_snapshot")(namespace="shop")
    starts, ends = pairs(sink)
    assert len(starts) == 1
    event = ends[0]
    assert event.status == "incomplete" and not event.succeeded
    assert event.incomplete and event.unavailable == ["pods"]
    assert "Health is unknown" in str(returned)
    assert event.errors == ["pods: error: Kubernetes API unavailable"]
    assert files.resolve(event.result)["content"] == returned
    observations = [files.resolve(ref) for ref in event.output_artifacts]
    assert sum(value.get("result") == reads["pods"] for value in observations) == 2


async def test_real_write_gate_refusal_is_not_a_successful_dispatch(tmp_path: Path) -> None:
    transport = Transport()
    ctx, sink, files = context(tmp_path, transport, role="execution")
    gate = WriteGate(
        authorities={"deny": DenyAll()},
        settings=AuthoritySettings(mode="deny"),
        tools=ctx.settings,
        journal=ctx.journal,
    )
    result = await run(
        ctx,
        [call("exec_kubectl_cmd_safely", {"cmd": "kubectl delete pod web -n shop"})],
        MiddlewareStack([gate]),
    )
    _, ends = pairs(sink)
    assert len(ends) == 1 and transport.calls == []
    assert ends[0].status == "refused" and not ends[0].succeeded
    assert ends[0].transport is None
    reply = next(message["content"] for message in result.transcript if message["role"] == "tool")
    assert reply == "denied by HITL gate: writes are disabled"
    assert files.resolve(ends[0].result)["content"] == reply


async def test_named_output_preserves_failed_backend_status_under_saved_file_wrapper(
    tmp_path: Path,
) -> None:
    raw = "error: backend unavailable\n" + "details" * 5_000
    ctx, sink, files = context(tmp_path, Transport(raw))
    returned = await bound(ctx, "exec_kubectl_cmd_safely")(
        cmd="kubectl get pods -n shop", save_as="failed-read.log"
    )
    _, ends = pairs(sink)
    assert str(returned).startswith("[Saved ")
    assert ends[0].status == "failed" and not ends[0].succeeded
    assert ends[0].truncated and ends[0].full_result_chars == len(raw)
    assert any(files.resolve(ref).get("content") == raw for ref in ends[0].output_artifacts)
    assert (ctx.workspace.outputs / "failed-read.log").read_text() == raw


async def test_cancelled_real_dispatch_has_one_terminal_and_transport_error(tmp_path: Path) -> None:
    class Waiting(Transport):
        async def query_metrics(self, query: str) -> str:
            entered.set()
            await asyncio.Future()
            raise AssertionError("cancelled call resumed")

    entered = asyncio.Event()
    ctx, sink, files = context(tmp_path, Waiting())
    task = asyncio.create_task(run(ctx, [call("get_metrics", {"query": "up"})]))
    await asyncio.wait_for(entered.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    _, ends = pairs(sink)
    assert len(ends) == 1 and ends[0].status == "cancelled" and not ends[0].succeeded
    assert "CancelledError" in ends[0].error
    observations = [files.resolve(ref) for ref in ends[0].output_artifacts]
    assert any(value.get("error", {}).get("type") == "CancelledError" for value in observations)
    assert current_call() is None


async def test_invalid_and_unknown_calls_are_recorded_without_transport_dispatch(
    tmp_path: Path,
) -> None:
    transport = Transport()
    ctx, sink, files = context(tmp_path, transport)
    result = await run(
        ctx,
        [
            call("get_metrics", {}, id="bad-json", parse_error="unterminated string"),
            call("missing_tool", {}, id="unknown"),
        ],
    )
    starts, ends = pairs(sink)
    assert len(starts) == 2 and transport.calls == [] and result.submitted
    assert {item.call_id for item in ends} == {"bad-json", "unknown"}
    assert all(
        item.status == "failed" and not item.succeeded and item.transport is None for item in ends
    )
    assert all(files.resolve(item.result)["content"].startswith("error:") for item in ends)


async def test_same_tool_nested_under_another_incident_gets_its_own_pair(tmp_path: Path) -> None:
    outer_ctx, outer_sink, _ = context(tmp_path, Transport("outer"), session="outer")
    inner_ctx, inner_sink, _ = context(tmp_path, Transport("inner"), session="inner")
    recorder = ToolCallRecorder(outer_ctx.journal, "get_metrics", {"query": "outer"})
    with recorder.active():
        assert await bound(inner_ctx, "get_metrics")(query="inner") == "inner"
        recorder.finish("outer")
    assert len(pairs(outer_sink)[0]) == 1
    assert len(pairs(inner_sink)[0]) == 1
    assert current_call() is None
