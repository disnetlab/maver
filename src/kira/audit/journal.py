"""The ordered per-incident event log.

Reconstructing a run afterwards is a stated property of the system, not a debugging
convenience, so the log is written on the path that does the work rather than derived
later from application logs. Every model call, tool call, submission outcome, policy
decision and harness intervention lands here in sequence.

The sink is a protocol with a no-op default. A run that has not been given somewhere to
write still executes; it simply is not reconstructible, which is a deployment choice
rather than something the loop should have to branch on.
"""

import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable

from kira.contracts.events import AuditEvent, EventPayload

__all__ = ["EventSink", "Journal", "ListSink", "NullSink"]

_log = logging.getLogger(__name__)


@runtime_checkable
class EventSink(Protocol):
    """Somewhere audit events are written."""

    def emit(self, event: AuditEvent) -> None:
        """Record one event.

        Args:
            event: The event to record.
        """
        ...


class NullSink:
    """Discards events. The default, so an unconfigured run still executes."""

    def emit(self, event: AuditEvent) -> None:
        """Discard the event.

        Args:
            event: Ignored.
        """


class ListSink:
    """Keeps events in memory, in order.

    Used by tests and by anything that wants a run's events without a database.
    """

    def __init__(self) -> None:
        """Start with no events."""
        self.events: list[AuditEvent] = []

    def emit(self, event: AuditEvent) -> None:
        """Append the event.

        Args:
            event: The event to keep.
        """
        self.events.append(event)


class Journal:
    """Stamps and writes a run's events.

    Owns the sequence number so ordering is a property of the log rather than of the
    clock: two events written in the same millisecond still have a defined order, which
    matters because the interesting questions are about what followed what.
    """

    def __init__(
        self,
        sink: EventSink | None = None,
        *,
        session_id: str,
        stage: str,
        run_id: str | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        """Build a journal for one stage of one session.

        Args:
            sink: Where events go. Defaults to discarding them.
            session_id: The incident session.
            stage: The role or mechanism these events occur under.
            run_id: The attempt within the session, when there is one.
            now: Clock override, for tests.
        """
        self._sink = sink or NullSink()
        self._session_id = session_id
        self._stage = stage
        self._run_id = run_id
        self._now = now or (lambda: datetime.now(UTC))
        self._sequence = 0

    @property
    def sequence(self) -> int:
        """How many events have been written.

        Returns:
            The current sequence number.
        """
        return self._sequence

    def record(self, payload: EventPayload) -> AuditEvent:
        """Write one event.

        A sink that raises is logged and swallowed: losing the audit trail is bad, and
        killing a live incident response because the audit database is unreachable is
        worse.

        Args:
            payload: What happened.

        Returns:
            The stamped event, whether or not the sink accepted it.
        """
        self._sequence += 1
        event = AuditEvent(
            session_id=self._session_id,
            sequence=self._sequence,
            stage=self._stage,
            run_id=self._run_id,
            occurred_at=self._now(),
            payload=payload,
        )
        try:
            self._sink.emit(event)
        except Exception as exc:  # the log must not end the run
            _log.warning("audit sink rejected event %d: %s", event.sequence, exc, exc_info=exc)
        return event
