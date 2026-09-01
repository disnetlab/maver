"""Writing the incident log where it survives the process.

Everything a run does — every model call, tool call, submission validation, harness
intervention, authority decision and stage transition — reaches this. It is the record
the system's auditability claim rests on, and until now it reached a sink that discarded
it: the events were built, ordered and stamped, and then dropped.

The order is the journal's, not the store's. A sequence number is allocated per session
before the write, because timestamps have collided at the resolution these stores use and
the interesting questions are all about what followed what.

A failure to record is logged and swallowed. An incident that is being handled correctly
must not be ended by an audit write, and a lost event is visible as a gap in the sequence
— which is a better outcome than a lost run and is why the sequence is the journal's to
allocate.
"""

import json
import logging
import uuid

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


class StoreSink:
    """Writes audit events to the run store."""

    def __init__(self, runs: RunStore) -> None:
        """Bind to the store that holds the runs these events belong to.

        Args:
            runs: Where events are appended.
        """
        self._runs = runs

    def emit(self, event: AuditEvent) -> None:
        """Record one event, or lose it loudly.

        Args:
            event: The event to record.
        """
        try:
            self._runs.append_event(
                AgentRunEventDraft(
                    event_id=str(uuid.uuid4()),
                    # An event raised before any role opened a run belongs to the
                    # incident rather than to a run, and the store's own column says so.
                    agent_run_id=event.run_id or "",
                    pipeline_id=event.session_id,
                    event_type=event.payload.kind,
                    payload=_payload(event),
                    created_at=event.occurred_at.isoformat(),
                    role=event.stage,
                    tool_call_id=getattr(event.payload, "call_id", None),
                    tool_name=getattr(event.payload, "tool", None),
                )
            )
        except Exception:
            # Never fatal. An incident being handled correctly must not be ended by a
            # write to its own log.
            _log.exception(
                "could not record a %s event for %s", event.payload.kind, event.session_id
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
