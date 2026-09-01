"""How an incident ended: resolved once, then written down.

The prototype decided this twice, in two vocabularies over the same nine conditions.
One function returned a telemetry string for the post-incident hooks to match on, and
another returned a lifecycle-and-run-status pair for persistence to record. Two copies
of one decision drift, and these two already disagreed on their last rule by
construction. One record carries both, so they cannot.

Resolving the outcome is kept apart from persisting it because the two do not always
happen at the same moment. A benchmark adapter resolves as soon as the pipeline stops
but withholds the terminal run status until its own submission has landed, so that a
driver polling the run does not read "finished" before the answer has been handed over.
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pydantic import BaseModel

from kira.contracts.lifecycle import LifecycleState, RunStatus
from kira.contracts.state import IncidentState
from kira.memory.ports import SessionStore

__all__ = ["Outcome", "OutcomeRecord", "persist_outcome", "resolve_outcome"]


class Outcome(StrEnum):
    """What the incident amounted to, in the vocabulary the hooks match on."""

    SUCCESS = "success"
    FAILURE = "failure"
    HALTED = "halted"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class OutcomeRecord:
    """One incident's ending, in every vocabulary that needs it."""

    outcome: Outcome
    lifecycle: LifecycleState | None
    """None means: write no lifecycle transition. See rule 9 in :func:`resolve_outcome`."""
    run_status: RunStatus | None
    notes: str | None


def resolve_outcome(state: IncidentState) -> OutcomeRecord:
    """Map a final incident state to its outcome, lifecycle, run status and notes.

    Single source of truth shared by persistence, which records it, and by an adapter
    that defers the terminal run status until after its own submission step.

    Args:
        state: The incident state as the pipeline left it.

    Returns:
        The resolved outcome.
    """
    error = state.get("error")
    root_cause = state.get("root_cause")
    plan = state.get("plan")
    execution = state.get("execution_result")
    verify = state.get("verify_result")

    if error:
        return OutcomeRecord(Outcome.ERROR, LifecycleState.INVALIDATED, RunStatus.FAILED, error)

    # getattr rather than attribute access: the diagnosis slot may hold the single-agent
    # configuration's free-text diagnosis, which has no halt field.
    rca_halted = root_cause is not None and bool(getattr(root_cause, "halt", False))
    plan_halted = plan is not None and plan.halt
    if rca_halted or plan_halted:
        # A halt is COMPLETED, not FAILED. A role that stopped deliberately, with a
        # stated reason, did its job; recording that as a failure would make the halt
        # path indistinguishable from a crash in every count taken downstream.
        stage = "planning" if plan_halted else "rca"
        halted = plan if plan_halted else root_cause
        reason = getattr(halted, "halt_reason", None) or "no reason given"
        return OutcomeRecord(
            Outcome.HALTED,
            LifecycleState.HALTED,
            RunStatus.COMPLETED,
            f"{stage} halted: {reason}",
        )

    if state.get("target_stage") == "diagnose":
        if root_cause is not None:
            return OutcomeRecord(
                Outcome.SUCCESS,
                LifecycleState.TENTATIVE_SUCCESS,
                RunStatus.COMPLETED,
                "RCA-only task completed; no remediation was requested",
            )
        return OutcomeRecord(
            Outcome.FAILURE,
            LifecycleState.INVALIDATED,
            RunStatus.FAILED,
            "RCA-only task produced no result",
        )

    # When the Verify step ran, its end-to-end verdict — not Execution's "applied"
    # claim — decides the outcome. recovered/inconclusive are tentative successes
    # (the latter applied but recovery could not be confirmed); not_recovered is a
    # disproven fix.
    if verify is not None:
        if verify.recovered or verify.verdict == "inconclusive":
            return OutcomeRecord(
                Outcome.SUCCESS,
                LifecycleState.TENTATIVE_SUCCESS,
                RunStatus.COMPLETED,
                f"verify: {verify.verdict}",
            )
        return OutcomeRecord(
            Outcome.FAILURE, LifecycleState.INVALIDATED, RunStatus.FAILED, "verify: not_recovered"
        )

    if execution is not None:
        if execution.succeeded:
            return OutcomeRecord(
                Outcome.SUCCESS, LifecycleState.TENTATIVE_SUCCESS, RunStatus.COMPLETED, None
            )
        return OutcomeRecord(
            Outcome.FAILURE,
            LifecycleState.INVALIDATED,
            RunStatus.FAILED,
            "execution reported failure",
        )

    # Nothing decisive was produced. The prototype signalled that with an all-None
    # triple, on which its persister returned early and wrote nothing at all; that
    # sentinel survives here as lifecycle=None, which the persister reads the same way.
    # The outcome itself is still populated, so a hook or a telemetry counter always has
    # a value to match on — which is exactly where the prototype's two functions
    # disagreed.
    return OutcomeRecord(Outcome.FAILURE, None, None, None)


def persist_outcome(
    state: IncidentState,
    record: OutcomeRecord,
    store: SessionStore,
    *,
    finalize_run_status: bool = True,
) -> None:
    """Write an incident's artefacts and, when it reached one, its ending.

    The lifecycle state is always recorded as soon as it is known, so the stored record
    reflects the pipeline's result even when the run status is deferred.

    Args:
        state: The incident state as the pipeline left it.
        record: The outcome resolved from that state.
        store: Where the session lives.
        finalize_run_status: Whether to write the terminal run status now, which is the
            signal a poller treats as "done". Every ordinary incident finalises
            immediately. A benchmark adapter passes False so the run stays ``running``
            through the offline learning pass and its own submission step, then
            finalises the status itself once that has landed.
    """
    session_id = state.get("session_id")
    if not session_id:
        return

    store.update_artefacts(
        session_id,
        root_cause=_serialised(state.get("root_cause")),
        plan=_serialised(state.get("plan")),
        execution_result=_serialised(state.get("execution_result")),
        verify_result=_serialised(state.get("verify_result")),
        notes=record.notes,
    )

    # Rule 9: no decisive result, so there is no transition to write and the session is
    # left exactly as the pipeline left it. The two fields are None together, which is
    # why one check covers both.
    if record.lifecycle is None or record.run_status is None:
        return

    store.set_state(session_id, record.lifecycle)
    # Unlike update_artefacts, a None here clears rather than leaves alone: the stage,
    # the checkpoint and the request being waited on all go. That is what releases a
    # session parked on an approval.
    store.set_run_status(
        session_id,
        record.run_status if finalize_run_status else RunStatus.RUNNING,
        current_stage=None,
        checkpoint=None,
        awaiting_request_id=None,
    )


def _serialised(artefact: BaseModel | None) -> dict[str, Any] | None:
    return artefact.model_dump(mode="json") if artefact is not None else None
