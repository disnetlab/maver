"""One measured tool attempt, including refusals and interruptions."""

import asyncio
import inspect
import time
import uuid
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, cast

from maver.audit.context import CallContext, ToolObservation, bind_call, current_call
from maver.audit.journal import Journal
from maver.contracts.events import ToolCallEvent, ToolCallStartedEvent
from maver.tools.errors import call_failed

if TYPE_CHECKING:
    from maver.tools.context import ToolContext
    from maver.tools.transport.base import ToolTransport


@dataclass
class ToolCallRecorder:
    """Collect a started/terminal pair without changing model-visible tool text."""

    journal: Journal
    tool: str
    arguments: dict[str, Any]
    call_id: str | None = None
    rationale: str | None = None
    observation: ToolObservation = field(default_factory=ToolObservation)
    invocation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    _started: float = field(default_factory=time.perf_counter)
    _finished: bool = False

    def finish(self, content: str, *, status: Any = None, error: str | None = None) -> None:
        """Persist the actual reply and every available observation exactly once."""
        if self._finished:
            return
        self._finished = True
        obs = self.observation
        failed = call_failed(content)
        if status is None:
            status = obs.status or (
                "failed"
                if failed
                else "incomplete"
                if obs.incomplete or obs.upstream_truncated
                else "completed"
            )
        self.journal.record(
            ToolCallEvent(
                tool=self.tool,
                call_id=self.call_id,
                invocation_id=self.invocation_id,
                arguments=self.arguments,
                rationale=self.rationale,
                succeeded=status == "completed",
                status=status,
                result_chars=len(content),
                full_result_chars=obs.full_result_chars
                if obs.full_result_chars is not None
                else len(content),
                duration_ms=round((time.perf_counter() - self._started) * 1000),
                transport=obs.transport,
                truncated=obs.truncated,
                upstream_truncated=obs.upstream_truncated,
                incomplete=obs.incomplete or obs.upstream_truncated,
                unavailable=obs.unavailable,
                errors=obs.errors,
                error=error
                or (
                    obs.errors[0]
                    if status == "failed" and obs.errors
                    else content
                    if status in {"failed", "refused"}
                    else None
                ),
                result=self.journal.capture("tool_result", {"content": content}),
                output_artifacts=obs.output_artifacts,
                retrieval=obs.retrieval,
            )
        )

    @contextmanager
    def active(self) -> Iterator["ToolCallRecorder"]:
        """Record an attempt before running it and preserve exceptional exits."""
        self._started = time.perf_counter()
        self.journal.record(
            ToolCallStartedEvent(
                tool=self.tool,
                call_id=self.call_id,
                invocation_id=self.invocation_id,
                arguments=self.arguments,
                rationale=self.rationale,
            )
        )
        context = CallContext(
            invocation_id=self.invocation_id,
            call_id=self.call_id,
            run_id=self.journal.run_id,
            stage=self.journal.stage,
            session_id=self.journal.session_id,
            observation=self.observation,
            journal=self.journal,
            tool=self.tool,
        )
        with bind_call(context):
            try:
                yield self
            except BaseException as exc:
                status = "cancelled" if isinstance(exc, asyncio.CancelledError) else "failed"
                self.finish("", status=status, error=f"{type(exc).__name__}: {exc}")
                raise


def capture_output(
    content: str, *, label: str, path: str | None = None, truncated: bool = False
) -> None:
    """Retain a complete filtered tool result before its bounded prompt representation."""
    call = current_call()
    if call is None:
        return
    obs = call.observation
    ref = call.journal.capture("tool_output", {"content": content})
    obs.output_artifacts.append(
        {**ref, "label": label, **({"workspace_path": path} if path else {})}
    )
    obs.full_result_chars = len(content)
    obs.truncated = obs.truncated or truncated
    if call_failed(content):
        obs.status = "failed"
        obs.errors.append(content)


class ObservedTransport:
    """Capture backend observations through the existing transport protocol."""

    def __init__(self, transport: "ToolTransport") -> None:
        """Delegate to the incident's transport without changing its operations."""
        self._transport = transport

    def __getattr__(self, name: str) -> Any:
        """Preserve properties and observe callable transport operations."""
        operation = getattr(self._transport, name)
        if not callable(operation):
            return operation

        def observed(*args: Any, **kwargs: Any) -> Any:
            call = current_call()
            if call is None:
                return operation(*args, **kwargs)
            call.observation.transport = self._transport.name
            # Freeze inputs before a backend can mutate them. The terminal observation
            # links this request to the complete returned value or raised error.
            request = call.journal.capture(
                "transport_request",
                {
                    "transport": self._transport.name,
                    "operation": name,
                    "args": args,
                    "kwargs": kwargs,
                },
            )
            started = time.perf_counter()

            def finish(*, result: Any = None, error: BaseException | None = None) -> None:
                payload: dict[str, Any] = {
                    "transport": self._transport.name,
                    "operation": name,
                    "request": request,
                    "duration_ms": round((time.perf_counter() - started) * 1000),
                }
                if error is not None:
                    payload["error"] = {"type": type(error).__name__, "message": str(error)}
                else:
                    payload["result"] = result
                call.observation.output_artifacts.append(
                    call.journal.capture("transport_observation", payload)
                )

            try:
                returned = operation(*args, **kwargs)
            except BaseException as exc:
                finish(error=exc)
                raise
            if inspect.isawaitable(returned):

                async def receive() -> Any:
                    try:
                        result = await returned
                    except BaseException as exc:
                        finish(error=exc)
                        raise
                    finish(result=result)
                    return result

                return receive()
            finish(result=returned)
            return returned

        return observed


async def observe_tool[ResultT](
    ctx: "ToolContext",
    name: str,
    arguments: dict[str, Any],
    handler: Callable[..., Awaitable[ResultT]],
) -> ResultT:
    """Measure dispatch or a harness read, preserving its result and arguments."""
    active = current_call()
    if (
        active is None
        or active.dispatching
        or active.tool != name
        or active.session_id != ctx.journal.session_id
        or active.run_id != ctx.journal.run_id
        or active.stage != ctx.journal.stage
    ):
        recorder = ToolCallRecorder(ctx.journal, name, arguments)
        with recorder.active():
            returned = await observe_tool(ctx, name, arguments, handler)
            recorder.finish(str(returned))
            return returned
    transport = (
        ctx.transport
        if isinstance(ctx.transport, ObservedTransport)
        else cast("ToolTransport", ObservedTransport(ctx.transport))
    )
    context = replace(ctx, observation=active.observation, transport=transport)
    with bind_call(replace(active, dispatching=True)):
        returned = await handler(context, **arguments)
    if hasattr(returned, "retrieval"):
        active.observation.retrieval = list(returned.retrieval)
    if active.observation.transport is None and name in {"bash_exec", "code_exec", "ssh_exec"}:
        active.observation.transport = "local"
    if active.observation.transport is None and name.startswith("memory_"):
        active.observation.transport = "memory"
    return returned
