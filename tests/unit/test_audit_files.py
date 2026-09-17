"""Durable file writes report failures and flush published file identities."""

import os
import stat
from pathlib import Path
from typing import Any

import pytest

from maver.audit.durable_sink import DurableAuditSink
from maver.audit.files import AuditFiles
from maver.audit.journal import Journal, ListSink, NullSink, SequenceRegistry
from maver.contracts.events import StageEvent, ToolCallEvent


def test_blob_publication_flushes_its_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = AuditFiles(tmp_path / "audit")
    operations = []
    replace, fsync = os.replace, os.fsync

    def record_replace(source: Any, target: Any) -> None:
        replace(source, target)
        operations.append("rename")

    def record_sync(descriptor: int) -> None:
        fsync(descriptor)
        operations.append("directory" if stat.S_ISDIR(os.fstat(descriptor).st_mode) else "file")

    monkeypatch.setattr(os, "replace", record_replace)
    monkeypatch.setattr(os, "fsync", record_sync)
    reference = files.capture("telemetry", {"body": "full response"})
    assert operations == ["file", "rename", "directory"]
    assert files.resolve(reference) == {"body": "full response"}


def test_capture_failure_keeps_an_immutable_complete_inline_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    sink = DurableAuditSink(NullSink(), audit_dir=tmp_path / "audit")
    journal = Journal(sink, session_id="incident", stage="rca")

    def fail_write(relative: str, value: Any) -> None:
        raise OSError("audit volume full")

    original = {"body": "evidence\x00" * 30_000, "nested": {"count": 2}}
    with monkeypatch.context() as mutation:
        mutation.setattr(sink.files, "_write", fail_write)
        reference = journal.capture("telemetry", original)
    original["nested"]["count"] = 99
    assert reference["inline"]["nested"] == {"count": 2}
    assert reference["inline"]["body"] == "evidence\x00" * 30_000
    assert reference["capture_error"] == "audit volume full"
    assert sink.status()["capture_failures"] == 1
    assert "audit evidence capture failed" in caplog.text
    journal.record(
        ToolCallEvent(
            tool="get_logs",
            arguments={},
            succeeded=True,
            result=reference,
            result_chars=len(reference["inline"]["body"]),
        )
    )
    persisted = sink.files.events("incident")[0]["payload"]["result"]
    assert sink.files.resolve(persisted) == reference["inline"]


def test_corrupt_sequence_reservation_is_observable_and_does_not_invent_an_order(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    downstream = ListSink()
    sink = DurableAuditSink(downstream, audit_dir=tmp_path / "audit")
    marker = sink.files.root / "sequences" / sink.files.session_key("incident")
    marker.write_text("damaged")
    allocator = SequenceRegistry(reserve=sink.sequence_reserver).for_session("incident", start=3)
    journal = Journal(sink, session_id="incident", stage="rca", sequence=allocator)
    with pytest.raises(ValueError, match="invalid literal"):
        journal.record(StageEvent(stage="rca", event="entered"))
    assert marker.read_text() == "damaged"
    assert allocator.last == 3
    assert downstream.events == []
    assert sink.files.events("incident") == []
    assert sink.status()["sequence_failures"] == 1
    assert "audit sequence reservation failed" in caplog.text
