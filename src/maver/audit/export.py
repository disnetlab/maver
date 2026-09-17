"""Offline, private bundles of an incident's audit stream and referenced evidence."""

import json
import os
from collections import Counter
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

from maver.audit.files import AuditFiles, encoded


def _read_lines(path: Path) -> tuple[bytes, list[dict[str, Any]], list[dict[str, Any]]]:
    raw = path.read_bytes()
    rows: list[dict[str, Any]] = []
    issues = []
    for number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("event is not an object")
            rows.append(value)
        except (ValueError, UnicodeError) as exc:
            issues.append({"line": number, "reason": type(exc).__name__})
    return raw, rows, issues


class _Evidence:
    """Follow verified references without opening paths outside their source root."""

    def __init__(self, root: Path, *, raw: bool = False, prefix: str = "blobs") -> None:
        self.root = root.resolve()
        self.raw = raw
        self.prefix = prefix
        self.files: dict[str, bytes] = {}
        self.artifacts: list[dict[str, Any]] = []
        self.issues: list[dict[str, Any]] = []
        self.persistence_degradations: list[dict[str, Any]] = []
        self._seen: set[tuple[str, str]] = set()

    def visit(self, value: Any) -> None:
        if isinstance(value, list):
            for item in value:
                self.visit(item)
        elif isinstance(value, dict):
            if "sha256" in value and "bytes" in value and ("path" in value or "inline" in value):
                self._reference(value)
            else:
                for item in value.values():
                    self.visit(item)

    def _reference(self, ref: dict[str, Any]) -> None:
        name = str(ref.get("path", "inline"))
        if ref.get("capture_error"):
            degradation = {
                "path": name,
                "sha256": ref["sha256"],
                "capture_error": str(ref["capture_error"]),
                "retained_inline": "inline" in ref,
            }
            if degradation not in self.persistence_degradations:
                self.persistence_degradations.append(degradation)
        identity = (name, str(ref["sha256"]))
        if identity in self._seen:
            return
        self._seen.add(identity)
        value: Any = None
        if "inline" in ref:
            value = ref["inline"]
            data = encoded(value)
        else:
            relative = Path(name)
            path = (self.root / relative).resolve()
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or not path.is_relative_to(self.root)
            ):
                self.issues.append({"path": name, "reason": "evidence path escapes audit root"})
                return
            if not relative.parts or relative.parts[0] != self.prefix:
                self.issues.append({"path": name, "reason": "unexpected audit evidence path"})
                return
            try:
                data = path.read_bytes()
            except FileNotFoundError:
                self.issues.append({"path": name, "reason": "missing evidence"})
                return
            self.files[name] = data
        actual = sha256(data).hexdigest()
        self.artifacts.append({"path": name, "bytes": len(data), "sha256": actual})
        if actual != ref["sha256"] or len(data) != ref["bytes"]:
            self.issues.append({"path": name, "reason": "checksum mismatch"})
        if self.raw:
            return
        if value is None:
            try:
                value = json.loads(data)
            except (ValueError, UnicodeError):
                self.issues.append({"path": name, "reason": "invalid JSON evidence"})
                return
        self.visit(value)


def _pairing(events: list[dict[str, Any]]) -> dict[str, Any]:
    started: dict[str, set[str]] = {"model": set(), "tool": set()}
    ended: dict[str, set[str]] = {"model": set(), "tool": set()}
    kinds: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    coverage: Counter[str] = Counter()
    attempts: Counter[str] = Counter()
    expected_attempts: dict[str, int] = {}
    for event in events:
        payload = event.get("payload") or {}
        if not isinstance(payload, dict):
            payload = {}
        kind = str(payload.get("kind", "unknown"))
        kinds[kind] += 1
        if kind == "model_attempt":
            if payload.get("call_id"):
                attempts[str(payload["call_id"])] += 1
            else:
                coverage["model_attempts_without_identity"] += 1
            if payload.get("status") == "succeeded" and not payload.get("response"):
                coverage["successful_model_attempts_without_response_evidence"] += 1
            continue
        if kind in {"tool_call", "tool_call_started"}:
            family, identity = "tool", payload.get("invocation_id")
            if kind == "tool_call":
                statuses[str(payload.get("status") or "unknown")] += 1
                if not payload.get("result"):
                    coverage["tool_results_without_evidence"] += 1
                if payload.get("incomplete") or payload.get("upstream_truncated"):
                    coverage["tool_results_reported_incomplete"] += 1
        elif kind in {"model_call", "model_call_started"}:
            family, identity = "model", payload.get("call_id")
            if kind == "model_call_started" and not payload.get("request"):
                coverage["model_requests_without_evidence"] += 1
            if kind == "model_call":
                if not payload.get("error") and not payload.get("response"):
                    coverage["successful_model_calls_without_response_evidence"] += 1
                observed = payload.get("attempts_observed")
                if observed == 0:
                    coverage["model_calls_without_provider_attempt_observations"] += 1
                elif type(observed) is not int or observed < 0:
                    coverage["model_calls_with_unknown_provider_attempt_coverage"] += 1
                elif identity:
                    expected_attempts[str(identity)] = observed
        else:
            continue
        if not identity:
            coverage[f"{family}_events_without_identity"] += 1
            continue
        target = started if kind.endswith("_started") else ended
        target[family].add(str(identity))
    unfinished = {key: sorted(started[key] - ended[key]) for key in started}
    orphaned = {key: sorted(ended[key] - started[key]) for key in ended}
    for family in started:
        if unfinished[family]:
            coverage[f"unfinished_{family}_calls"] = len(unfinished[family])
        if orphaned[family]:
            coverage[f"{family}_terminals_without_start"] = len(orphaned[family])
    mismatched = [
        identity for identity, count in expected_attempts.items() if attempts[identity] != count
    ]
    if mismatched:
        coverage["model_calls_with_attempt_count_mismatch"] = len(mismatched)
    return {
        "event_kinds": dict(kinds),
        "tool_statuses": dict(statuses),
        "coverage_gaps": dict(coverage),
        "started_without_terminal": unfinished,
        "terminal_without_started": orphaned,
        "model_attempt_count_mismatches": sorted(mismatched),
    }


def _sequences(events: list[dict[str, Any]]) -> dict[str, Any]:
    positions = [
        e["sequence"] for e in events if type(e.get("sequence")) is int and e["sequence"] > 0
    ]
    counts = Counter(positions)
    missing = []
    previous = 0
    for position in sorted(counts):
        if position > previous + 1:
            missing.append([previous + 1, position - 1])
        previous = position
    return {
        "first_sequence": min(positions, default=None),
        "last_sequence": max(positions, default=None),
        "missing_sequence_ranges": missing,
        "duplicate_sequences": sorted(key for key, count in counts.items() if count > 1),
        "invalid_sequence_count": len(events) - len(positions),
    }


def _private_write(root: Path, relative: str, data: bytes) -> None:
    target = root / relative
    if target.parent != root:
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        for parent in (target.parent, *target.parent.parents):
            if parent == root:
                break
            parent.chmod(0o700)
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as output:
        output.write(data)


def export_incident(
    audit_dir: Path,
    session_id: str,
    destination: Path,
    *,
    bridge_log: Path | None = None,
) -> dict[str, Any]:
    """Export a verifiable snapshot without opening a database or contacting services.

    Missing or damaged evidence is reported in the manifest; it is never replaced
    by an invented empty result. Existing destination contents are never overwritten.
    """
    source = audit_dir.resolve()
    target = destination.resolve()
    if target == source or target.is_relative_to(source) or source.is_relative_to(target):
        raise ValueError("export destination must be outside the source audit directory")
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise ValueError("export destination must be absent or empty")
    path = source / "events" / f"{AuditFiles.session_key(session_id)}.jsonl"
    if not path.resolve().is_relative_to(source):
        raise ValueError("audit event path escapes its root")
    raw, events, read_issues = _read_lines(path)
    if any(event.get("session_id") != session_id for event in events):
        raise ValueError("audit stream contains another incident")
    evidence = _Evidence(source)
    evidence.visit(events)
    pending: dict[str, bytes] = {}
    pending_ids = []
    pending_issues = []
    for candidate in sorted((source / "pending").glob("*.json")):
        if not candidate.resolve().is_relative_to(source):
            pending_issues.append(candidate.name)
            continue
        try:
            data = candidate.read_bytes()
            draft = json.loads(data)
            if not isinstance(draft, dict):
                raise ValueError("pending projection must be an object")
        except FileNotFoundError:
            continue  # the database worker acknowledged it during this snapshot
        except (ValueError, UnicodeError):
            pending_issues.append(candidate.name)
            continue
        if draft.get("pipeline_id") == session_id:
            pending[f"pending/{candidate.name}"] = data
            pending_ids.append(str(draft.get("event_id", candidate.stem)))
            evidence.visit(draft)

    summary = {
        "export_format_version": 1,
        "session_id": session_id,
        "exported_at": datetime.now(UTC).isoformat(),
        "source_audit_dir": str(source),
        "source_events_sha256": sha256(raw).hexdigest(),
        "event_count": len(events),
        "stream_read_issues": read_issues,
        "artifacts": evidence.artifacts,
        "evidence_issues": evidence.issues,
        "persistence_degradations": evidence.persistence_degradations,
        "database_pending_event_ids": sorted(pending_ids),
        "unreadable_pending_files": pending_issues,
        "database_projection": {
            "coverage": "unknown",
            "delivery": "best_effort",
            "verification": "not_queried",
            "legacy_pending_markers_observed": len(pending_ids),
            "pending_markers_are_exhaustive": False,
            "note": "Pending markers are optional legacy records; their absence does not prove database completeness.",
        },
        **_pairing(events),
        **_sequences(events),
    }
    damaged = bool(
        evidence.issues
        or read_issues
        or summary["missing_sequence_ranges"]
        or summary["duplicate_sequences"]
        or summary["invalid_sequence_count"]
    )
    # Integrity checks the evidence we have; coverage asks what was captured.
    # Neither proves that every database projection was delivered.
    summary["evidence_integrity"] = "incomplete" if damaged else "complete"
    summary["capture_coverage"] = (
        "limited"
        if damaged or summary["coverage_gaps"]
        else "complete_for_observed_calls"
        if events
        else "unknown"
    )
    files = {"events.jsonl": raw, **evidence.files, **pending}

    if bridge_log is not None:
        _, rows, issues = _read_lines(bridge_log)
        rows = [row for row in rows if row.get("session") == session_id]
        host = _Evidence(bridge_log.parent, raw=True, prefix=bridge_log.name + ".artifacts")
        for row in rows:
            host.visit(row.get("request"))
            host.visit(row.get("result"))
        identities = {
            str(e.get("payload", {}).get("invocation_id"))
            for e in events
            if isinstance(e.get("payload"), dict) and e["payload"].get("invocation_id")
        }
        received = {
            str(row["receipt_id"])
            for row in rows
            if row.get("event") == "received" and row.get("receipt_id")
        }
        finished = {
            str(row["receipt_id"])
            for row in rows
            if row.get("event") == "finished" and row.get("receipt_id")
        }
        legacy = sum(not row.get("event") or not row.get("invocation_id") for row in rows)
        unmatched = sorted(
            {
                str(row["invocation_id"])
                for row in rows
                if row.get("invocation_id") and row["invocation_id"] not in identities
            }
        )
        host_coverage = {}
        if legacy:
            host_coverage["legacy_rows_without_full_identity"] = legacy
        if received - finished:
            host_coverage["unfinished_requests"] = len(received - finished)
        if finished - received:
            host_coverage["finished_requests_without_receipt"] = len(finished - received)
        if unmatched:
            host_coverage["unmatched_invocation_ids"] = len(unmatched)
        missing_payloads = sum(
            not row.get("request") or (row.get("event") == "finished" and not row.get("result"))
            for row in rows
            if row.get("event") in {"received", "finished"}
        )
        if missing_payloads:
            host_coverage["rows_without_full_payload_evidence"] = missing_payloads
        audit_failures = sum(bool(row.get("audit_errors")) for row in rows)
        if audit_failures:
            host_coverage["rows_with_audit_recording_errors"] = audit_failures
        summary["bridge"] = {
            "event_count": len(rows),
            "legacy_rows": legacy,
            "started_without_terminal": sorted(received - finished),
            "terminal_without_started": sorted(finished - received),
            "unmatched_invocation_ids": unmatched,
            "coverage_gaps": host_coverage,
            "evidence_integrity": "incomplete" if host.issues or issues else "complete",
            "capture_coverage": "limited"
            if host.issues or issues or host_coverage
            else "complete_for_observed_calls"
            if rows
            else "unknown",
            "read_issues": issues,
            "evidence_issues": host.issues,
            "artifacts": host.artifacts,
        }
        if summary["bridge"]["evidence_integrity"] == "incomplete":
            summary["evidence_integrity"] = "incomplete"
        if summary["bridge"]["capture_coverage"] != "complete_for_observed_calls":
            summary["capture_coverage"] = "limited"
        files["bridge/bridge.log"] = b"".join(encoded(row) + b"\n" for row in rows)
        for relative, data in host.files.items():
            files[f"bridge/{relative}"] = data

    target.mkdir(parents=True, exist_ok=True, mode=0o700)
    target.chmod(0o700)
    for relative, data in files.items():
        _private_write(target, relative, data)
    _private_write(target, "manifest.json", encoded(summary) + b"\n")
    return summary
