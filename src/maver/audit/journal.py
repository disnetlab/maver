"""The ordered per-incident event log.

Reconstructing a run afterwards is a stated property of the system, not a debugging
convenience, so the log is written on the path that does the work rather than derived
later from application logs. Every model call, tool call, submission outcome, policy
decision and harness intervention lands here in sequence.

The sink is a protocol with a no-op default. A run that has not been given somewhere to
write still executes; it simply is not reconstructible, which is a deployment choice
rather than something the loop should have to branch on.
"""

import json
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable

from maver.audit.files import encoded, reference
from maver.contracts.events import AuditEvent, EventPayload

__all__ = [
    "EventSink",
    "Journal",
    "ListSink",
    "NullSink",
    "SequenceAllocator",
    "SequenceRegistry",
]

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


class SequenceAllocator:
    """Hands out one session's event positions, in order.

    Session-scoped and not journal-scoped. A session is written by a journal per stage,
    so a position each of them restarts cannot order the incident it describes: sorting
    one incident by a per-journal count puts the recovery verdict before the write it
    verifies.
    """

    def __init__(self, start: int = 0, reserve: Callable[[int], int] | None = None) -> None:
        """Continue an existing order, or begin one.

        Args:
            start: The last position already handed out. The next is one past it.
            reserve: Optional durable allocator receiving the current sequence floor.
        """
        self._last = start
        self._reserve = reserve

    @property
    def last(self) -> int:
        """The position handed out most recently.

        Returns:
            The position, or where the order was started when nothing has been.
        """
        return self._last

    def allocate(self) -> int:
        """Take the next position.

        Returns:
            The position, one past the last handed out.
        """
        self._last = self._reserve(self._last) if self._reserve is not None else self._last + 1
        return self._last


class SequenceRegistry:
    """The allocator each session's journals share, for the length of this process.

    Held by the deployment rather than in a module global, so two deployments in one
    process are genuinely independent — the same reason the write lock lives there.

    Production reserves positions through the durable audit store, so resumed,
    asynchronous and restarted writers share one incident order. In-memory sinks keep
    the supplied checkpoint floor and process-local ordering.
    """

    def __init__(
        self, capacity: int = 256, reserve: Callable[[str, int], int] | None = None
    ) -> None:
        """Start empty.

        Args:
            capacity: How many sessions keep their place. Far above any real
                concurrency, so what is evicted is a session that stopped emitting
                thousands of incidents ago; an unbounded map in a served process is the
                alternative.
            reserve: Optional durable allocator receiving a session and sequence floor.
        """
        self._capacity = capacity
        self._reserve = reserve
        self._allocators: dict[str, SequenceAllocator] = {}

    def for_session(self, session_id: str, *, start: int = 0) -> SequenceAllocator:
        """The allocator this session's journals write through.

        Args:
            session_id: The incident.
            start: Where to continue from, honoured only when this process has no
                allocator for the session yet. A process that already holds one is
                mid-incident, and its own position is ahead of any stored number.

        Returns:
            The session's allocator, created on first use.
        """
        allocator = self._allocators.get(session_id)
        if allocator is None:
            reserve = self._reserve
            allocator = SequenceAllocator(
                start,
                (lambda minimum: reserve(session_id, minimum)) if reserve is not None else None,
            )
            self._allocators[session_id] = allocator
            while len(self._allocators) > self._capacity:
                self._allocators.pop(next(iter(self._allocators)))
        return allocator


class Journal:
    """Stamps and writes a run's events.

    Ordering is a property of the log rather than of the clock: two events written in
    the same millisecond still have a defined order, which matters because the
    interesting questions are about what followed what. The order is the session's, so
    the allocator is shared by every stage's journal rather than owned by one of them.
    """

    def __init__(
        self,
        sink: EventSink | None = None,
        *,
        session_id: str,
        stage: str,
        run_id: str | None = None,
        now: Callable[[], datetime] | None = None,
        sequence: SequenceAllocator | None = None,
    ) -> None:
        """Build a journal for one stage of one session.

        Args:
            sink: Where events go. Defaults to discarding them.
            session_id: The incident session.
            stage: The role or mechanism these events occur under.
            run_id: The attempt within the session, when there is one.
            now: Clock override, for tests.
            sequence: The session's position allocator. A journal given none allocates
                alone, which only orders its own events.
        """
        self._sink = sink or NullSink()
        self._session_id = session_id
        self._stage = stage
        self._run_id = run_id
        self._now = now or (lambda: datetime.now(UTC))
        self._sequence = sequence or SequenceAllocator()
        self._last = 0

    @property
    def sequence(self) -> int:
        """The position of the last event written through this journal.

        Returns:
            The position, or zero before anything is written.
        """
        return self._last

    @property
    def run_id(self) -> str | None:
        """Return the owning role run, when the event has one."""
        return self._run_id

    @property
    def session_id(self) -> str:
        """Return the incident this journal records."""
        return self._session_id

    @property
    def stage(self) -> str:
        """Return the calling role or mechanism."""
        return self._stage

    def capture(self, kind: str, value: Any) -> dict[str, Any]:
        """Preserve full evidence, with an inline fallback if durable capture fails."""
        value = json.loads(encoded(value))
        capture = getattr(self._sink, "capture", None)
        if capture is not None:
            try:
                return dict(capture(kind, value))
            except Exception as exc:
                _log.error("audit evidence capture failed: %s", exc, exc_info=exc)
                return {**reference(kind, value), "inline": value, "capture_error": str(exc)}
        return {**reference(kind, value), "inline": value}

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
        self._last = self._sequence.allocate()
        event = AuditEvent(
            session_id=self._session_id,
            sequence=self._last,
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
