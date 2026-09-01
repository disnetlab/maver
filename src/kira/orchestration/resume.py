"""Parking an incident on a human, and continuing it once they answer.

The pause is a database write, not a held coroutine. Everything needed to continue is
serialised to the session — the paused role, the call it was making, its transcript and
the incident's state — so an approval that arrives twenty minutes later, or after a
restart, resumes the same incident rather than starting another one.

The continuation lives here rather than in the role that raised, which is the layering
that makes it testable. The prototype's remediating role imported the graph, the outcome
resolver and the post-incident hooks in order to drive the pipeline itself after a
resume, and that inversion is why the approval path was the least exercised part of the
system. Here a role raises, the orchestrator catches, and the restored state re-enters
the same router every other run goes through — which also fixes the prototype's
imprecision that a resumed re-plan landed on diagnosis.

The retry counters are carried across the pause. The prototype reset them, so an
operator's approval silently bought a fresh remediation budget and made a resumed run
incomparable with one that never paused.
"""

import logging
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict

from kira.contracts.artifacts import (
    ExecutionResult,
    Plan,
    RootCause,
    SingleDiagnosis,
    SingleResult,
    VerifyResult,
)
from kira.contracts.events import PolicyDecisionEvent, ToolCallEvent
from kira.contracts.lifecycle import RunStatus
from kira.contracts.state import IncidentState
from kira.memory.ports import ApprovalRecord, ApprovalStore, SessionStore
from kira.middleware.authority import ApprovalPending

__all__ = [
    "ApprovalCheckpoint",
    "denial_result",
    "load_checkpoint",
    "restored_state",
    "resume_events",
    "save_checkpoint",
]

_log = logging.getLogger(__name__)

_RESUMABLE = ("approved", "denied", "timed_out")


class ApprovalCheckpoint(BaseModel):
    """Everything needed to restart a paused role, and nothing role-specific.

    The prototype's checkpoint was shaped for the remediating role — it named that
    stage and that role's message list — so no other stage could pause. This one names
    the stage instead of assuming it.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    session_id: str
    stage: str
    request_id: str
    tool_call_id: str
    tool_name: str
    tool_arguments: dict[str, Any]

    messages: list[dict[str, Any]]
    """The paused role's transcript. The other roles' live inside ``state``."""
    state: dict[str, Any]

    run_id: str
    run_ids: list[str]

    remediation_retries: int
    reassessment_count: int
    verify_retries: int
    """Carried, not reset. An approval must not buy a fresh budget."""


def save_checkpoint(
    store: SessionStore, state: IncidentState, pending: ApprovalPending
) -> ApprovalCheckpoint:
    """Park the incident on the request that was just raised.

    The status, the stage, the checkpoint and the request being waited on move together
    in one write, which is what makes the pause atomic from a poller's point of view: a
    reader never sees a session awaiting a human without the checkpoint that belongs to
    it.

    Args:
        store: Where the session lives.
        state: The incident state as the paused role was entered with it.
        pending: The raised pause, with the role's transcript attached.

    Returns:
        The checkpoint that was written.
    """
    checkpoint = ApprovalCheckpoint(
        session_id=pending.session_id,
        stage=pending.stage,
        request_id=pending.request_id,
        tool_call_id=pending.tool_call_id,
        tool_name=pending.tool_name,
        tool_arguments=dict(pending.tool_arguments),
        messages=list(pending.messages),
        state=_serialisable(state),
        run_id=str(state.get("run_ids", [""])[-1]) if state.get("run_ids") else "",
        run_ids=list(state.get("run_ids", [])),
        remediation_retries=state.get("remediation_retries", 0),
        reassessment_count=state.get("reassessment_count", 0),
        verify_retries=state.get("verify_retries", 0),
    )
    store.set_run_status(
        pending.session_id,
        RunStatus.AWAITING_HUMAN,
        current_stage=pending.stage,
        checkpoint=checkpoint.model_dump(mode="json"),
        awaiting_request_id=pending.request_id,
    )
    _log.info(
        "incident %s is parked on approval request %s at stage %s",
        pending.session_id,
        pending.request_id,
        pending.stage,
    )
    return checkpoint


def load_checkpoint(
    sessions: SessionStore, approvals: ApprovalStore, request_id: str
) -> tuple[ApprovalCheckpoint, ApprovalRecord] | str:
    """Read back what a decided request needs in order to continue.

    Args:
        sessions: Where the session lives.
        approvals: Where the request lives.
        request_id: The request that was decided.

    Returns:
        The checkpoint and its request, or a short status naming why it cannot continue.
        The statuses are returned rather than raised because the caller is a scheduler
        job, and a job's log is the only place they are read.
    """
    record = approvals.get_request(request_id)
    if record is None:
        return "missing_request"
    if record.status not in _RESUMABLE:
        return f"not_ready:{record.status}"

    session = sessions.get_session(record.session_id)
    if session is None:
        return "missing_session"
    raw = session.checkpoint
    if not raw:
        return "missing_checkpoint"

    try:
        checkpoint = ApprovalCheckpoint.model_validate(raw)
    except Exception:
        _log.exception("session %s has an unreadable checkpoint", record.session_id)
        return "missing_checkpoint"
    if checkpoint.request_id != request_id:
        # The session moved on: it is parked on a different request now, and replaying
        # this one would apply a write against a state nobody approved it for.
        return "missing_checkpoint"
    return checkpoint, record


def denial_result(record: ApprovalRecord) -> str:
    """Render what the paused tool call returns when the answer was no.

    Args:
        record: The decided request.

    Returns:
        The tool result the role sees in place of the call it made.
    """
    reason = record.reason or (
        "timed out with no operator decision" if record.status == "timed_out" else "denied"
    )
    return f"denied by HITL gate: {reason}"


def restored_state(checkpoint: ApprovalCheckpoint) -> IncidentState:
    """Rebuild the incident state the paused role was entered with.

    Args:
        checkpoint: What was written at the pause.

    Returns:
        The state, with the counters as they stood rather than reset.
    """
    state = dict(checkpoint.state)
    for slot, models in _SLOTS.items():
        state[slot] = _rehydrate(state.get(slot), models)
    state["remediation_retries"] = checkpoint.remediation_retries
    state["reassessment_count"] = checkpoint.reassessment_count
    state["verify_retries"] = checkpoint.verify_retries
    state["run_ids"] = list(checkpoint.run_ids)
    return cast(IncidentState, state)


def resume_events(
    checkpoint: ApprovalCheckpoint, record: ApprovalRecord, *, result_chars: int
) -> list[Any]:
    """Build the audit entries a continuation owes its log.

    Both, always. The decision says what a person answered, and the call event says what
    the run then did with it — a log carrying only the first cannot distinguish an
    approval that was replayed from one that was never acted on.

    Args:
        checkpoint: What was written at the pause.
        record: The decided request.
        result_chars: How much the replayed call returned.

    Returns:
        The events, in the order they should be written.
    """
    return [
        PolicyDecisionEvent(
            tool=checkpoint.tool_name,
            policy="prompt",
            decision=cast(Any, record.status),
            reason=record.reason or "",
            request_id=record.request_id,
            decided_by=record.decided_by,
        ),
        ToolCallEvent(
            tool=checkpoint.tool_name,
            arguments=dict(checkpoint.tool_arguments),
            rationale="replayed after an operator decision",
            succeeded=record.status == "approved",
            result_chars=result_chars,
        ),
    ]


_SLOTS: dict[str, tuple[type[BaseModel], ...]] = {
    "root_cause": (RootCause, SingleDiagnosis),
    "plan": (Plan,),
    "execution_result": (ExecutionResult, SingleResult),
    "verify_result": (VerifyResult,),
}
"""Which artifact each state slot can hold. Two entries where the single-agent
configuration fills the same slot with its own unstructured artifact."""


def _rehydrate(value: Any, models: tuple[type[BaseModel], ...]) -> Any:
    """Turn a stored artifact back into the object a role expects.

    A checkpoint is JSON, and a role reads its predecessor's artifact by attribute. The
    first model that accepts the payload wins; the pipeline's shape is tried before the
    single-agent one because the two are not ambiguous in practice — one has a fault
    type, the other a sentence.
    """
    if not isinstance(value, dict):
        return value
    for model in models:
        try:
            return model.model_validate(value)
        except Exception:
            continue
    _log.warning("could not restore a stored artifact; leaving it as it was stored")
    return value


def _serialisable(state: IncidentState) -> dict[str, Any]:
    """Render the state as JSON, keeping the artifacts as their stored dumps."""
    out: dict[str, Any] = {}
    for key, value in state.items():
        dump = getattr(value, "model_dump", None)
        out[key] = dump(mode="json") if dump is not None else value
    return out
