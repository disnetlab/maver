"""Canonical audit files survive an incomplete legacy database projection."""

from pathlib import Path
from typing import Any

from test_store_sink import _SlowStore, a_run, event, storage

from maver.audit.durable_sink import DurableAuditSink
from maver.audit.files import AuditFiles
from maver.audit.journal import NullSink
from maver.audit.store_sink import StoreSink
from maver.contracts.events import StageEvent, ToolCallEvent


def test_a_dropped_database_backlog_does_not_drop_canonical_events(tmp_path: Path) -> None:
    database = _SlowStore()
    sink = DurableAuditSink(StoreSink(database, capacity=1), audit_dir=tmp_path / "audit")
    try:
        for sequence in range(1, 21):
            sink.emit(event(StageEvent(stage="execution", event="entered"), sequence=sequence))
        assert [item["sequence"] for item in sink.files.events("s1")] == list(range(1, 21))
    finally:
        database.released.set()
        sink.close()
    assert len(database.written) < 20
    assert sink.status()["canonical_failures"] == 0
    assert sink.status()["database_coverage"] == "unknown"


def test_full_canonical_event_survives_the_legacy_projection_payload_limit(tmp_path: Path) -> None:
    stack = storage(tmp_path / "db")
    a_run(stack)
    sink = DurableAuditSink(StoreSink(stack.runs), audit_dir=tmp_path / "audit")
    payload = "evidence\x00" * 50_000
    original = event(
        ToolCallEvent(
            tool="get_logs", arguments={"query": payload}, succeeded=True, result_chars=len(payload)
        )
    )
    sink.emit(original)
    sink.close()
    assert sink.files.events("s1") == [original.model_dump(mode="json")]
    assert stack.runs.events_for("r1")[0].payload["truncated"] is True
    reference = sink.capture("raw_telemetry", {"content": payload})
    assert AuditFiles(tmp_path / "audit").resolve(reference) == {"content": payload}
    assert "inline" not in reference
    stack.close()


def test_a_failing_projection_still_leaves_the_full_event_and_persistent_order(
    tmp_path: Path,
) -> None:
    class Broken:
        def emit(self, item: Any) -> None:
            raise RuntimeError("database unavailable")

    sink = DurableAuditSink(Broken(), audit_dir=tmp_path / "audit")
    original = event(StageEvent(stage="execution", event="entered"))
    sink.emit(original)
    assert sink.files.events("s1") == [original.model_dump(mode="json")]
    assert sink.status()["projection_failures"] == 1
    assert sink.sequence_reserver("s1", 40) == 41
    restarted = DurableAuditSink(NullSink(), audit_dir=tmp_path / "audit")
    assert restarted.sequence_reserver("s1", 0) == 42
