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
from datetime import datetime
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict

from maver.contracts.artifacts import (
    ExecutionResult,
    Plan,
    RootCause,
    SingleDiagnosis,
    SingleResult,
    VerifyResult,
)
from maver.contracts.events import PolicyDecisionEvent, ToolCallEvent
from maver.contracts.lifecycle import RunStatus
from maver.contracts.state import IncidentState, TargetStage
from maver.memory.ports import ApprovalRecord, ApprovalStore, SessionRecord, SessionStore
from maver.middleware.authority import ApprovalPending
from maver.orchestration.inflight import held_episode
from maver.settings.authority import AuthoritySettings

__all__ = [
    "ApprovalCheckpoint",
    "approves_on_timeout",
    "denial_result",
    "load_checkpoint",
    "restored_state",
    "resume_events",
    "save_checkpoint",
    "stored_state",
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

    sequence: int = 0
    """The last event position this incident handed out, so the continuation carries on
    from it. An order that restarted at 1 could no longer sort the incident's own log."""

    remediation_retries: int
    reassessment_count: int
    verify_retries: int
    """Carried, not reset. An approval must not buy a fresh budget."""

    mode: str = "multi"
    """The pipeline shape this incident was running, so the continuation runs it too.

    Resuming always drove the graph from the interrupted stage. A single-agent run
    resumed as a graph run, and a diagnosis-only task resumed at diagnosis and then
    carried on into planning and remediation — mutating an environment whose read-only
    answer was the only thing being graded."""
    target_stage: str = "full"
    fire_post_hooks: bool = True
    finalize_run_status: bool = True
    """Held from the call that parked. An evaluation driver owns the run's ending: it
    submits, collects the grade and releases the environment. A continuation that
    finalised the run itself would close the episode from the wrong side."""
    adapter: str = ""
    """Which environment was serving this incident, when one was.

    Not enough to continue under, and it is not meant to be: an adapter instance holds
    the transport its episode's tools speak and the correlation id its harness keys on,
    and one built fresh from this name has acquired none of it. It is enough to *refuse*
    to continue, which is the point. A continuation that found no live episode for an
    incident this names would otherwise replay the approved write over the process's
    default transport, outside the session being graded."""


def save_checkpoint(
    store: SessionStore,
    state: IncidentState,
    pending: ApprovalPending,
    *,
    sequence: int = 0,
    mode: str = "multi",
    target_stage: str = "full",
    fire_post_hooks: bool = True,
    finalize_run_status: bool = True,
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
        sequence: The last event position handed out, for the continuation to resume the
            order from.
        mode: The pipeline shape the incident was running.
        target_stage: How far the caller asked the pipeline to go.
        fire_post_hooks: Whether this caller wants the post-incident hooks fired.
        finalize_run_status: Whether this caller wants the run marked terminal.

    Returns:
        The checkpoint that was written.
    """
    # Read here rather than passed in, for two reasons. The caller that parks is the
    # incident runner, which by design cannot name an environment; and the stamp has to
    # land in the same write as the pause, or a decision answered in the same breath
    # would read back a checkpoint that did not yet say what was serving the incident.
    held = held_episode(pending.session_id)
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
        sequence=sequence,
        mode=mode,
        target_stage=target_stage,
        fire_post_hooks=fire_post_hooks,
        finalize_run_status=finalize_run_status,
        adapter=held.adapter.name if held is not None else "",
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


def approves_on_timeout(
    record: ApprovalRecord, authority: AuthoritySettings, *, now: str
) -> str | None:
    """Name who approves an expired request, when a rule said one class of call may.

    The rule the gate matched is recorded on the request, so the answer comes from what
    governed the call rather than from re-matching a policy that may have been edited
    while the operator was away. Only a rule can approve on expiry, never the global
    mode: an operator naming the narrow class of actions that are safe unattended is the
    whole of the safety property, and a global switch would approve everything.

    Args:
        record: The request that may have run out of time.
        authority: The gate's configuration.
        now: The current timestamp.

    Returns:
        The decider to attribute the approval to, or None when the request denies.
    """
    if record.status != "pending" or not _has_expired(record.expires_at, now):
        return None
    name = record.payload.get("rule")
    for rule in authority.rules:
        if rule.name == name and rule.on_timeout == "approve":
            return f"rule:{rule.name}"
    return None


def _has_expired(expires_at: str | None, now: str) -> bool:
    """Say whether a deadline has passed, on the store's own terms.

    An unparseable expiry is not evidence that anything has expired, which is what the
    store's sweep does with one; here it is what keeps a job that fired early from
    approving a write nobody has yet failed to answer.
    """
    if not expires_at:
        return False
    try:
        return datetime.fromisoformat(expires_at) <= datetime.fromisoformat(now)
    except ValueError:
        return False


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


def stored_state(record: SessionRecord, *, target_stage: TargetStage = "full") -> IncidentState:
    """Rebuild what an incident produced, from what was persisted for it.

    A continuation reports a status, not a state, so an episode that still has answers to
    hand over after one reads them back from the store. Only the alert and the artefacts
    survive that round trip, which is exactly what a submission and an outcome are
    resolved from; the transcripts and the retry counters do not, and nothing after the
    pipeline wants them.

    Args:
        record: The stored session.
        target_stage: How far the run was asked to go, which the store does not keep and
            which decides the outcome of a diagnosis-only task.

    Returns:
        The state, as far as it was written down.
    """
    state: dict[str, Any] = {
        "session_id": record.session_id,
        "alert": dict(record.alert),
        "target_stage": target_stage,
        "error": None,
    }
    for slot, models in _SLOTS.items():
        state[slot] = _rehydrate(getattr(record, slot), models)
    return cast(IncidentState, state)


def resume_events(
    checkpoint: ApprovalCheckpoint, record: ApprovalRecord, *, result_chars: int
) -> list[Any]:
    """Build the audit entries a continuation owes its log.

    Both, always. The decision says what a person answered, and the call event says what
    the run then did with it — a log carrying only the first cannot distinguish an
    approval that was replayed from one that was never acted on. Both name the call, so
    they join to the position in the stored transcript that the operator was shown.

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
            call_id=checkpoint.tool_call_id,
            policy="prompt",
            decision=cast(Any, record.status),
            reason=record.reason or "",
            request_id=record.request_id,
            decided_by=record.decided_by,
        ),
        ToolCallEvent(
            tool=checkpoint.tool_name,
            call_id=checkpoint.tool_call_id,
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
