"""The incident log, written where it survives the process."""

import pathlib
from datetime import UTC, datetime
from typing import Any

from kira.audit.store_sink import StoreSink
from kira.contracts.events import AuditEvent, StageEvent, ToolCallEvent
from kira.memory.factory import build_storage
from kira.memory.ports import AgentRunRecord
from kira.settings.groups import ServerSettings
from kira.settings.root import Settings


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


def test_events_reach_the_store_in_the_order_the_journal_gave_them(
    tmp_path: pathlib.Path,
) -> None:
    """Ordering is by sequence and not by timestamp: timestamps have collided at the
    resolution these stores use, and every interesting question is about what followed
    what."""
    stack = storage(tmp_path)
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

    StoreSink(stack.runs).emit(
        event(
            ToolCallEvent(
                tool="get_logs",
                arguments={"query": "x" * 400_000},
                succeeded=True,
                result_chars=400_000,
            )
        )
    )

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

    StoreSink(Broken()).emit(event(StageEvent(stage="execution", event="entered")))
