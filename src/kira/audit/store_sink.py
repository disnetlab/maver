"""Writing the incident log where it survives the process.

Everything a run does — every model call, tool call, submission validation, harness
intervention, authority decision and stage transition — reaches this. It is the record
the system's auditability claim rests on, and until now it reached a sink that discarded
it: the events were built, ordered and stamped, and then dropped.

The order is the journal's, not the store's. A sequence number is allocated per session
before the write — by one allocator every stage's journal shares — because timestamps
have collided at the resolution these stores use and the interesting questions are all
about what followed what. The store's own ``seq`` remains per run, so the session's order
is the one in the payload.

A failure to record is logged and swallowed. An incident that is being handled correctly
must not be ended by an audit write, and a lost event is visible as a gap in the sequence
— which is a better outcome than a lost run and is why the sequence is the journal's to
allocate.

**The write is handed to a thread rather than waited for.** Not for speed: a healthy
insert costs under half a millisecond and would be invisible either way. Measured against
a live PostgreSQL with ``agent_run_events`` under an ``ACCESS EXCLUSIVE`` lock, one
``Journal.record`` held the event loop for the whole statement timeout — thirty seconds,
in which the loop ran nothing at all, not the other incidents this process is handling,
not the harness polling them, not the readiness probe — and then the statement was
cancelled and the event was lost anyway. Hundreds of events go through here per incident.
Inline, that is hours of a frozen process bought in exchange for a log that was not being
written either.

``record`` cannot become a coroutine to fix it. Seventeen of its call sites are inside
synchronous functions — the write gate, the second-pass audit, the loop's own nested
helpers — so the hop belongs on the one method they all funnel into.

What it costs is the tail, and that cost is the reason to state it plainly: an event is
durable when the drain thread has written it, not when ``record`` returned, so a process
killed outright loses whatever had not drained. At this write cost and this event rate
that is normally one event; an orderly shutdown loses none, because :meth:`close` drains
before the process goes. The trade is the same one the paragraph above already makes in
the other direction — a lost event is a gap a reader can see, and a lost run is not.
"""

import json
import logging
import queue
import threading
import time
import uuid
from contextlib import suppress

from kira.contracts.events import AuditEvent
from kira.memory.ports import AgentRunEventDraft, RunStore

__all__ = ["StoreSink"]

_log = logging.getLogger(__name__)

_MAX_PAYLOAD_CHARS = 200_000
"""Ceiling on one event's serialised payload.

A tool result can be megabytes, and the log is meant to be readable and queryable rather
than to be a second copy of every output. What is dropped is named in place, so a reader
sees a truncation rather than a short answer.
"""

QUEUE_CAPACITY = 1000
"""How many events may be waiting on the store.

Deep enough to ride out a stall — at the rate a run emits, on the order of a quarter of
an hour of a database that is not answering — and shallow enough that one which never
answers costs bounded memory rather than the process. Past it events are dropped, which
is what the synchronous sink did under exactly these conditions, without the freeze.
"""

DRAIN_TIMEOUT_S = 10.0
"""How long :meth:`close` waits for the backlog before naming what it is leaving."""

_STOP = None
"""Queued to end the drain. Nothing else put on the queue may be None."""


class StoreSink:
    """Writes audit events to the run store, from a thread of its own."""

    def __init__(self, runs: RunStore, *, capacity: int = QUEUE_CAPACITY) -> None:
        """Bind to the store that holds the runs these events belong to.

        Args:
            runs: Where events are appended.
            capacity: How many events may be waiting to be written.
        """
        self._runs = runs
        self._pending: queue.Queue[AgentRunEventDraft | None] = queue.Queue(maxsize=capacity)
        self._dropped = 0
        self._overflowing = False
        self._closed = False
        # Daemon, so a process that never closes the sink still exits. Started with the
        # sink rather than on the first event, because the sink is built once by the
        # composition root and a thread that appears mid-run is one nobody expected.
        self._worker = threading.Thread(target=self._drain, name="kira-audit-sink", daemon=True)
        self._worker.start()

    def emit(self, event: AuditEvent) -> None:
        """Queue one event for the store, and return.

        Rendered here, on the caller's thread, so that what waits in the queue is a
        finished record rather than a live object a later turn could still change — and
        so the payload ceiling stays where the caller can be charged for it.

        Args:
            event: The event to record.
        """
        if self._closed:
            # A shutdown drain is bounded, so an incident can outlive the flush. Its
            # events are lost either way, since the stores go next; what is left worth
            # doing is saying so, rather than queueing them where nobody will read.
            self._dropped += 1
            _log.warning(
                "an event for %s was raised after the audit sink closed, and is not recorded",
                event.session_id,
            )
            return
        draft = AgentRunEventDraft(
            event_id=str(uuid.uuid4()),
            # An event raised before any role opened a run belongs to the incident
            # rather than to a run, and the store's own column says so.
            agent_run_id=event.run_id or "",
            pipeline_id=event.session_id,
            event_type=event.payload.kind,
            payload=_payload(event),
            created_at=event.occurred_at.isoformat(),
            role=event.stage,
            tool_call_id=getattr(event.payload, "call_id", None),
            tool_name=getattr(event.payload, "tool", None),
        )
        try:
            self._pending.put_nowait(draft)
        except queue.Full:
            self._dropped += 1
            if not self._overflowing:
                self._overflowing = True
                _log.warning(
                    "the audit backlog is full at %d events; dropping events for %s until "
                    "the store keeps up",
                    self._pending.maxsize,
                    event.session_id,
                )
            return
        if self._overflowing:
            self._overflowing = False
            _log.warning("the audit backlog cleared, %d event(s) lost in the gap", self._dropped)

    def close(self, *, timeout: float = DRAIN_TIMEOUT_S) -> None:
        """Write what is still queued, then stop. Idempotent.

        Bounded, because a store that has stopped answering must not hold a shutdown
        open. What is abandoned is counted into the log, since the alternative is an
        incident log that is truncated and does not say so.

        Args:
            timeout: How long to wait for the backlog, in total.
        """
        if self._closed:
            return
        self._closed = True
        deadline = time.monotonic() + timeout
        # A full queue here means the drain is stuck on a write that is not returning,
        # so there is no room for the sentinel and no point in waiting for one.
        with suppress(queue.Full):
            self._pending.put(_STOP, timeout=max(0.0, deadline - time.monotonic()))
        self._worker.join(max(0.0, deadline - time.monotonic()))
        abandoned = self._pending.qsize()
        if abandoned or self._dropped:
            _log.warning(
                "the incident log is incomplete: %d event(s) dropped while running, "
                "%d still queued at shutdown",
                self._dropped,
                abandoned,
            )

    def _drain(self) -> None:
        """Write queued events in the order they were queued, one at a time.

        One consumer, so the order the journal stamped is the order the store is given —
        which is the order a single event loop was already giving it.
        """
        while True:
            draft = self._pending.get()
            if draft is _STOP:
                return
            try:
                self._runs.append_event(draft)
            except Exception:
                _log.exception(
                    "could not record a %s event for %s", draft.event_type, draft.pipeline_id
                )


def _payload(event: AuditEvent) -> dict[str, object]:
    """Render one event's payload, bounded."""
    body = event.payload.model_dump(mode="json")
    body["sequence"] = event.sequence
    encoded = json.dumps(body, default=str)
    if len(encoded) <= _MAX_PAYLOAD_CHARS:
        return body
    return {
        "kind": event.payload.kind,
        "sequence": event.sequence,
        "truncated": True,
        "chars": len(encoded),
        "excerpt": encoded[:_MAX_PAYLOAD_CHARS],
    }
