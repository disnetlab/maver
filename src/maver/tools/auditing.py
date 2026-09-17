"""One measured tool attempt, including refusals and interruptions."""

import asyncio
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from maver.audit.context import CallContext, ToolObservation, bind_call, current_call
from maver.audit.journal import Journal
from maver.contracts.events import ToolCallEvent, ToolCallStartedEvent
from maver.tools.errors import call_failed


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
            status = (
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
                incomplete=obs.incomplete,
                unavailable=obs.unavailable,
                errors=obs.errors,
                error=error or (content if failed else None),
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
