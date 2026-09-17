"""An offline bundle carries complete evidence and makes missing evidence explicit."""

import json
import stat
from hashlib import sha256
from pathlib import Path

from typer.testing import CliRunner

from maver.audit.files import AuditFiles, encoded, reference
from maver.cli import app


def append(files: AuditFiles, sequence: int, payload: dict, session: str = "incident-1") -> None:
    files.append_event(
        {
            "session_id": session,
            "sequence": sequence,
            "stage": "rca",
            "run_id": "run-1",
            "occurred_at": "2026-09-18T00:00:00Z",
            "payload": payload,
        },
    )
    legacy_pending = files.root / "pending"
    legacy_pending.mkdir(exist_ok=True)
    (legacy_pending / f"{session}-{sequence}.json").write_bytes(
        encoded({"event_id": f"{session}-{sequence}", "pipeline_id": session})
    )


def export(root: Path, target: Path, *extra: str):
    return CliRunner().invoke(
        app,
        [
            "audit",
            "export",
            "incident-1",
            "--audit-dir",
            str(root),
            "--output",
            str(target),
            *extra,
        ],
    )


def test_export_keeps_nested_evidence_future_fields_and_matching_pending_rows(
    tmp_path: Path,
) -> None:
    source = AuditFiles(tmp_path / "audit")
    messages = source.capture("messages", [{"role": "tool", "content": "complete " * 20000}])
    manifest = source.capture("request", {"messages": messages, "future_field": [1, 2, 3]})
    append(source, 1, {"kind": "model_call_started", "call_id": "model-1", "request": manifest})
    append(
        source,
        2,
        {
            "kind": "model_attempt",
            "call_id": "model-1",
            "status": "succeeded",
            "response": messages,
        },
    )
    append(
        source,
        3,
        {"kind": "model_call", "call_id": "model-1", "response": messages, "attempts_observed": 1},
    )
    append(source, 4, {"kind": "future_kind", "future": {"value": "preserved"}})
    append(source, 1, {"kind": "other"}, "another-incident")
    target = tmp_path / "bundle"

    result = export(source.root, target)

    assert result.exit_code == 0, result.output
    summary = json.loads((target / "manifest.json").read_text())
    assert summary["event_count"] == 4
    assert summary["capture_coverage"] == "complete_for_observed_calls"
    assert summary["evidence_integrity"] == "complete"
    assert summary["started_without_terminal"] == {"model": [], "tool": []}
    assert summary["database_pending_event_ids"] == [
        "incident-1-1",
        "incident-1-2",
        "incident-1-3",
        "incident-1-4",
    ]
    assert len(summary["artifacts"]) == 2
    for ref in (messages, manifest):
        assert (target / ref["path"]).read_bytes() == (source.root / ref["path"]).read_bytes()
    rows = [json.loads(line) for line in (target / "events.jsonl").read_text().splitlines()]
    assert rows[-1]["payload"]["future"] == {"value": "preserved"}
    assert stat.S_IMODE(target.stat().st_mode) == 0o700
    assert stat.S_IMODE((target / "manifest.json").stat().st_mode) == 0o600
    assert not (target / "pending/another-incident-1.json").exists()
    assert export(source.root, target).exit_code == 1


def test_export_reports_missing_corrupt_and_escaping_evidence_without_reading_it(
    tmp_path: Path,
) -> None:
    source = AuditFiles(tmp_path / "audit")
    corrupt = source.capture("request", {"data": "original"})
    (source.root / corrupt["path"]).write_text("{}")
    missing = source.capture("request", {"data": "missing"})
    (source.root / missing["path"]).unlink()
    escaped = {**source.capture("request", {"secret": "private"}), "path": "../secret.json"}
    (tmp_path / "secret.json").write_bytes(encoded({"secret": "private"}))
    append(source, 1, {"kind": "tool_call_started", "invocation_id": "attempt-1"})
    append(source, 3, {"kind": "future", "evidence": [corrupt, missing, escaped]})
    target = tmp_path / "bundle"

    result = export(source.root, target)

    assert result.exit_code == 0, result.output
    summary = json.loads((target / "manifest.json").read_text())
    assert summary["evidence_integrity"] == "incomplete"
    assert summary["missing_sequence_ranges"] == [[2, 2]]
    assert summary["started_without_terminal"]["tool"] == ["attempt-1"]
    assert {issue["reason"] for issue in summary["evidence_issues"]} == {
        "checksum mismatch",
        "missing evidence",
        "evidence path escapes audit root",
    }
    assert not (target / "secret.json").exists()


def test_export_joins_host_artifacts_and_marks_legacy_rows_as_limited(tmp_path: Path) -> None:
    source = AuditFiles(tmp_path / "audit")
    append(source, 1, {"kind": "tool_call_started", "invocation_id": "attempt-1"})
    append(source, 2, {"kind": "tool_call", "invocation_id": "attempt-1", "status": "completed"})
    bridge_path = tmp_path / "bridge.log"
    response_data = ("complete host result" * 1000).encode()
    response_path = tmp_path / "bridge.log.artifacts/safe-id.response.txt"
    response_path.parent.mkdir()
    response_path.write_bytes(response_data)
    response = {
        "path": str(response_path.relative_to(tmp_path)),
        "bytes": len(response_data),
        "sha256": sha256(response_data).hexdigest(),
    }
    rows = []
    for event in ("received", "finished"):
        rows.append(
            {
                "event": event,
                "session": "incident-1",
                "receipt_id": "receipt-1",
                "request_id": "request-1",
                "invocation_id": "attempt-1",
                "result": response,
            }
        )
    rows.append({"session": "incident-1", "op": "kubectl", "decision": "served"})
    rows.append({"session": "another-incident", "op": "must stay out"})
    bridge_path.write_bytes(b"".join(encoded(row) + b"\n" for row in rows))
    target = tmp_path / "bundle"

    result = export(source.root, target, "--bridge-log", str(bridge_path))

    assert result.exit_code == 0, result.output
    summary = json.loads((target / "manifest.json").read_text())
    assert summary["bridge"]["event_count"] == 3
    assert summary["bridge"]["legacy_rows"] == 1
    assert summary["evidence_integrity"] == "complete"
    assert summary["capture_coverage"] == "limited"
    assert summary["bridge"]["unmatched_invocation_ids"] == []
    assert summary["bridge"]["evidence_integrity"] == "complete"
    assert summary["bridge"]["capture_coverage"] == "limited"
    assert (target / "bridge" / response["path"]).read_text() == "complete host result" * 1000
    assert "must stay out" not in (target / "bridge/bridge.log").read_text()


def test_valid_checksums_do_not_claim_capture_or_database_completeness(tmp_path: Path) -> None:
    source = AuditFiles(tmp_path / "audit")
    value = {"capture_layer": "neutral_request", "messages": []}
    request = {**reference("request", value), "inline": value, "capture_error": "disk full"}
    append(source, 1, {"kind": "model_call_started", "call_id": "custom-1", "request": request})
    append(source, 2, {"kind": "model_call", "call_id": "custom-1", "attempts_observed": 0})
    append(source, 3, {"kind": "model_call_started", "call_id": "unfinished", "request": request})
    for marker in (source.root / "pending").iterdir():
        marker.unlink()
    target = tmp_path / "bundle"

    result = export(source.root, target)

    assert result.exit_code == 0, result.output
    summary = json.loads((target / "manifest.json").read_text())
    assert summary["evidence_integrity"] == "complete"
    assert summary["capture_coverage"] == "limited"
    assert summary["coverage_gaps"] == {
        "successful_model_calls_without_response_evidence": 1,
        "model_calls_without_provider_attempt_observations": 1,
        "unfinished_model_calls": 1,
    }
    assert summary["database_pending_event_ids"] == []
    assert summary["persistence_degradations"] == [
        {
            "path": "inline",
            "sha256": request["sha256"],
            "capture_error": "disk full",
            "retained_inline": True,
        }
    ]
    assert summary["database_projection"]["coverage"] == "unknown"
    assert summary["database_projection"]["delivery"] == "best_effort"
    assert summary["database_projection"]["pending_markers_are_exhaustive"] is False
    assert json.loads(result.stdout)["capture_coverage"] == "limited"
