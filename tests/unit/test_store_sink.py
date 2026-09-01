"""The incident log, written where it survives the process.

The store is a database on another host and the port into it is synchronous, so a write
made from the event loop holds the loop for as long as the database takes and nothing
above can cut it short. Measured with ``agent_run_events`` under an ``ACCESS EXCLUSIVE``
lock: one ``Journal.record`` held the loop for the full thirty-second statement timeout,
ran none of the other incidents in the process, and then lost the event anyway. So the
write is queued, and what these tests pin is both halves of that — the queue never makes
the caller wait, and closing the sink still writes everything queued.

Every store here answers on a bound of its own. A regression must fail on this clock
rather than hanging the suite, which is what a store that blocks forever would do.
"""

import pathlib
import threading
import time
from datetime import UTC, datetime
from typing import Any

from kira.audit.store_sink import StoreSink
from kira.contracts.events import AuditEvent, StageEvent, ToolCallEvent
from kira.memory.factory import build_storage
from kira.memory.ports import AgentRunEventDraft, AgentRunEventRecord, AgentRunRecord
from kira.settings.groups import ServerSettings
from kira.settings.root import Settings

STALLED = 2.0
"""How long a store stays unresponsive. Long enough that waiting on it is unmistakable.

An absolute deadline rather than a per-call one, so that reverting the fix costs the
suite this once and not this per event.
"""


def storage(tmp_path: pathlib.Path) -> Any:
    return build_storage(Settings(server=ServerSettings(workdir=tmp_path)))


def event(payload: Any, *, sequence: int = 1, run_id: str | None = "r1") -> AuditEvent:
    return AuditEvent(
        session_id="s1",
        sequence=sequence,
        stage="execution",
        run_id=run_id,
        occurred_at=datetime.now(UTC),
        payload=payload,
    )


def a_run(stack: Any) -> None:
    stack.runs.start_run(
        AgentRunRecord(
            agent_run_id="r1",
            pipeline_id="s1",
            agent_name="execution",
            attempt=1,
            status="running",
            started_at="2026-09-01T00:00:00Z",
        )
    )


class _SlowStore:
    """A store that makes every write wait, the way a lock on the table does.

    Bounded rather than held open, so a sink that waits for it fails on this clock
    instead of stopping the suite.
    """

    def __init__(self) -> None:
        self.released = threading.Event()
        self.written: list[AgentRunEventDraft] = []
        self._until = time.monotonic() + STALLED

    def append_event(self, draft: AgentRunEventDraft) -> AgentRunEventRecord:
        self.released.wait(timeout=max(0.0, self._until - time.monotonic()))
        self.written.append(draft)
        return AgentRunEventRecord(
            event_id=draft.event_id,
            agent_run_id=draft.agent_run_id,
            pipeline_id=draft.pipeline_id,
            seq=len(self.written),
            event_type=draft.event_type,
            payload=draft.payload,
            created_at=draft.created_at,
        )


def test_recording_an_event_does_not_wait_for_the_store(tmp_path: pathlib.Path) -> None:
    """The defect this sink exists to fix, at its own scale.

    One event took the whole statement timeout against a real database with the table
    locked, and the loop ran nothing for the length of it. Hundreds go through here per
    incident, so what matters is not how fast a write is but that the caller is never
    the one waiting for it.
    """
    store = _SlowStore()
    sink = StoreSink(store)

    started = time.perf_counter()
    for sequence in range(5):
        sink.emit(event(StageEvent(stage="execution", event="entered"), sequence=sequence + 1))
    recording = time.perf_counter() - started

    store.released.set()
    sink.close()
    assert recording < STALLED / 2, f"recording five events waited {recording:.1f}s on the store"
    assert len(store.written) == 5


def test_a_backlog_that_cannot_be_written_is_dropped_rather_than_waited_on(
    tmp_path: pathlib.Path,
) -> None:
    """The same claim once the queue is full, which is the only way it could still block.

    A store that has stopped answering must cost the log its events and not the run its
    clock — the trade this module already makes, now made where the queue runs out
    rather than where the write fails.
    """
    store = _SlowStore()
    sink = StoreSink(store, capacity=1)

    started = time.perf_counter()
    for sequence in range(50):
        sink.emit(event(StageEvent(stage="execution", event="entered"), sequence=sequence + 1))
    recording = time.perf_counter() - started

    store.released.set()
    sink.close()
    assert recording < STALLED / 2, f"a full backlog waited {recording:.1f}s on the store"
    assert len(store.written) < 50, "a capacity of one should not have held all fifty"


def test_events_reach_the_store_in_the_order_the_journal_gave_them(
    tmp_path: pathlib.Path,
) -> None:
    """Ordering is by sequence and not by timestamp: timestamps have collided at the
    resolution these stores use, and every interesting question is about what followed
    what. One drain thread, so queueing does not reorder what the journal stamped."""
    stack = storage(tmp_path)
    a_run(stack)
    sink = StoreSink(stack.runs)

    sink.emit(event(StageEvent(stage="execution", event="entered"), sequence=1))
    sink.emit(
        event(
            ToolCallEvent(
                tool="exec_kubectl_cmd_safely",
                arguments={"cmd": "kubectl get pods"},
                succeeded=True,
                result_chars=42,
            ),
            sequence=2,
        )
    )
    # An event is durable once the drain has written it, not once it was recorded. This
    # is the cost of not waiting, and the shutdown path pays it exactly here.
    sink.close()

    stored = stack.runs.events_for("r1")
    assert [e.event_type for e in stored] == ["stage", "tool_call"]
    assert [e.payload["sequence"] for e in stored] == [1, 2]
    assert stored[1].tool_name == "exec_kubectl_cmd_safely"
    stack.close()


def test_an_oversized_payload_is_truncated_visibly(tmp_path: pathlib.Path) -> None:
    """A tool result can be megabytes, and the log is meant to be readable rather than a
    second copy of every output. What is dropped is named, so a reader sees a truncation
    instead of a short answer."""
    stack = storage(tmp_path)
    a_run(stack)

    sink = StoreSink(stack.runs)
    sink.emit(
        event(
            ToolCallEvent(
                tool="get_logs",
                arguments={"query": "x" * 400_000},
                succeeded=True,
                result_chars=400_000,
            )
        )
    )
    sink.close()

    payload = stack.runs.events_for("r1")[0].payload
    assert payload["truncated"] is True
    assert payload["chars"] > 200_000
    stack.close()


def test_a_failed_write_does_not_end_the_incident(tmp_path: pathlib.Path) -> None:
    """An incident being handled correctly must not be ended by a write to its own log.
    A lost event shows as a gap in the sequence, which is why the journal allocates it
    rather than the store."""

    class Broken:
        def append_event(self, draft: Any) -> Any:
            raise RuntimeError("the store is gone")

    sink = StoreSink(Broken())
    sink.emit(event(StageEvent(stage="execution", event="entered")))
    # Closing a sink whose every write raises still returns, and the drain is still
    # alive to take the next event.
    sink.close()
