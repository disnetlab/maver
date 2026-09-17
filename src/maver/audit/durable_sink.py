"""A complete private file mirror around the existing best-effort database sink.

The mirror is canonical. Database queue overflow or payload limits affect only the
legacy projection; this wrapper does not retry or modify database writes.
"""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from maver.audit.files import AuditFiles
from maver.audit.journal import EventSink
from maver.contracts.events import AuditEvent

_log = logging.getLogger(__name__)


class DurableAuditSink:
    """Persist every full event before delegating to an optional database projection."""

    def __init__(self, downstream: EventSink, *, audit_dir: Path) -> None:
        """Bind the existing sink and the private canonical audit directory."""
        self.files = AuditFiles(audit_dir)
        self._downstream = downstream
        self._written = 0
        self._canonical_failures = 0
        self._capture_failures = 0
        self._sequence_failures = 0
        self._projection_failures = 0
        self._closed = False

    @property
    def sequence_reserver(self) -> Callable[[str, int], int]:
        """Return persistent ordering shared across processes and restarts."""
        return self.reserve_sequence

    def reserve_sequence(self, session_id: str, minimum: int = 0) -> int:
        """Reserve the next canonical event position for an incident."""
        try:
            return self.files.reserve_sequence(session_id, minimum)
        except Exception:
            self._sequence_failures += 1
            _log.exception("audit sequence reservation failed")
            raise

    def capture(self, kind: str, value: Any) -> dict[str, Any]:
        """Persist complete evidence separately from model-visible workspaces."""
        try:
            return self.files.capture(kind, value)
        except Exception:
            self._capture_failures += 1
            raise

    def emit(self, event: AuditEvent) -> None:
        """Mirror the complete event before offering it to the database queue.

        A file failure remains visible to the journal. The legacy sink still gets a
        chance to retain the event, but its projection is never claimed to be complete.
        """
        failure: Exception | None = None
        try:
            self.files.append_event(event.model_dump(mode="json"))
            self._written += 1
        except Exception as exc:
            self._canonical_failures += 1
            failure = exc
        try:
            self._downstream.emit(event)
        except Exception:
            self._projection_failures += 1
            _log.exception("database audit projection rejected an event")
        if failure is not None:
            raise failure

    def status(self) -> dict[str, Any]:
        """Report file failures without claiming the best-effort DB projection is complete."""
        return {
            "durable": True,
            "canonical_events": self._written,
            "canonical_failures": self._canonical_failures,
            "capture_failures": self._capture_failures,
            "sequence_failures": self._sequence_failures,
            "projection_failures": self._projection_failures,
            "database_projection": "best_effort",
            "database_coverage": "unknown",
            "closed": self._closed,
        }

    def close(self, *, timeout: float = 10.0) -> None:
        """Close the downstream queue; canonical events were already flushed on emit."""
        if self._closed:
            return
        self._closed = True
        closer = getattr(self._downstream, "close", None)
        if closer is not None:
            closer(timeout=timeout)
