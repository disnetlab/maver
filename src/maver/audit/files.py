"""Private, complete audit evidence independent of the database projection."""

import fcntl
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def encoded(value: Any) -> bytes:
    """Encode an audit value deterministically without truncation."""
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")


def reference(kind: str, value: Any) -> dict[str, Any]:
    """Describe the exact bytes captured, for persistent and in-memory sinks."""
    data = encoded(value)
    return {"kind": kind, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class AuditFiles:
    """Append-only events and content-addressed blobs outside agent workspaces."""

    def __init__(self, root: Path) -> None:
        """Create private directories for the canonical stream and its evidence."""
        self.root = root
        for path in (root, root / "blobs", root / "events", root / "sequences"):
            path.mkdir(parents=True, exist_ok=True, mode=0o700)
            path.chmod(0o700)

    def _write(self, relative: str, value: Any) -> None:
        target = self.root / relative
        descriptor, temporary = tempfile.mkstemp(prefix=".audit-", dir=target.parent)
        try:
            with os.fdopen(descriptor, "wb") as output:
                output.write(encoded(value))
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, target)
            _sync_directory(target.parent)
        finally:
            Path(temporary).unlink(missing_ok=True)

    def capture(self, kind: str, value: Any) -> dict[str, Any]:
        """Persist a complete JSON value once and return its verifiable reference."""
        ref = reference(kind, value)
        relative = f"blobs/{ref['sha256']}.json"
        target = self.root / relative
        if not target.exists():
            self._write(relative, value)
        return {**ref, "path": relative}

    def resolve(self, ref: dict[str, Any]) -> Any:
        """Read and verify a captured value; reject paths outside the audit root."""
        if "inline" in ref:
            value = ref["inline"]
        else:
            path = (self.root / str(ref["path"])).resolve()
            if not path.is_relative_to(self.root.resolve()):
                raise ValueError("audit reference escapes its root")
            value = json.loads(path.read_text(encoding="utf-8"))
        measured = reference(str(ref.get("kind", "evidence")), value)
        if measured["sha256"] != ref["sha256"] or measured["bytes"] != ref["bytes"]:
            raise ValueError("audit evidence checksum mismatch")
        return value

    @staticmethod
    def session_key(session_id: str) -> str:
        """Map an arbitrary incident identifier to a safe, stable filename."""
        return hashlib.sha256(session_id.encode("utf-8")).hexdigest()

    def reserve_sequence(self, session_id: str, minimum: int = 0) -> int:
        """Reserve an ordered position across processes, restarts and registry eviction."""
        path = self.root / "sequences" / self.session_key(session_id)
        descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        with os.fdopen(descriptor, "r+", encoding="utf-8") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            previous = handle.read().strip()
            position = max(int(previous or 0), minimum) + 1
            handle.seek(0)
            handle.write(str(position))
            handle.truncate()
            handle.flush()
            os.fsync(handle.fileno())
            _sync_directory(path.parent)
            return position

    def append_event(self, event: dict[str, Any]) -> None:
        """Append one complete canonical event without changing any database state.

        The immutable blob also preserves the whole record if the process stops during
        the JSONL append. Existing export detects incomplete JSONL lines explicitly.
        """
        self.capture("audit_event", event)
        path = self.root / "events" / f"{self.session_key(str(event['session_id']))}.jsonl"
        descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        with os.fdopen(descriptor, "ab") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            handle.write(encoded(event) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        _sync_directory(path.parent)

    def events(self, session_id: str) -> list[dict[str, Any]]:
        """Read one complete event stream, detecting incomplete writes explicitly."""
        path = self.root / "events" / f"{self.session_key(session_id)}.jsonl"
        if not path.exists():
            return []
        events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        return sorted(events, key=lambda item: item["sequence"])
