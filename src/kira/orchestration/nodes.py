"""What happens around every role, so no role has to do it.

Applying a hand-back, opening the run row, writing the stage events, summarising the
trace, merging the artifact into state, deciding whether a failure halts the incident:
uniform for every role, and repeated at roughly sixty lines in each of four role files in
the prototype. Repeated code drifts — the four copies there were already not identical —
and each divergence is invisible until it changes a result.

A role therefore returns an outcome and nothing else. Everything below turns that into a
state delta.
"""

import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, cast

from kira.audit.journal import Journal
from kira.contracts.events import StageEvent, SubmissionEvent
from kira.contracts.routing import HandBack
from kira.contracts.state import IncidentState
from kira.middleware.authority import ApprovalPending
from kira.orchestration.budget import RetryBudget
from kira.orchestration.routing import RoutingPolicy, apply_handback, pending_handback
from kira.roles.base import RoleContext, RoleOutcome, RoleSpec, StateDelta
from kira.roles.trace_summary import build_trace_summary

__all__ = ["IncidentDeps", "NodeFn", "make_node"]

_log = logging.getLogger(__name__)

NodeFn = Callable[[IncidentState], Awaitable[StateDelta]]


class ContextBuilder(Protocol):
    """Builds one role's context. Owned by the runner, which holds the run's resources."""

    def __call__(
        self,
        spec: RoleSpec,
        state: IncidentState,
        handback: HandBack | None,
        run_id: str,
    ) -> RoleContext:
        """Assemble the context for one role run.

        Args:
            spec: The role about to run.
            state: The state it is entered with, hand-back already applied.
            handback: Why this is a re-entry, or None.
            run_id: The persisted run this belongs to.

        Returns:
            The role's context.
        """
        ...


class RunRecorder(Protocol):
    """Opens and closes the persisted row for one role run."""

    def start(self, *, session_id: str, role: str, after: str | None) -> str:
        """Open a run row.

        Args:
            session_id: The incident.
            role: The role starting.
            after: The previous run in this pipeline, for the transition edge.

        Returns:
            The new run's identifier.
        """
        ...

    def finish(
        self,
        run_id: str,
        *,
        status: str,
        output: Mapping[str, Any] | None,
        trace_summary: Mapping[str, Any] | None,
        error: str | None,
        metadata: Mapping[str, Any],
    ) -> None:
        """Close a run row.

        Args:
            run_id: Which run.
            status: How it ended.
            output: The artifact it produced.
            trace_summary: Its bounded account of itself.
            error: Why it failed, when it did.
            metadata: Anything the role recorded about its own run.
        """
        ...


@dataclass(frozen=True, slots=True)
class IncidentDeps:
    """Everything the nodes need that is the same for the whole incident."""

    policy: RoutingPolicy
    runs: RunRecorder
    journal: Journal
    build_context: ContextBuilder


def make_node(spec: RoleSpec, deps: IncidentDeps) -> NodeFn:
    """Build the graph node for one role.

    Args:
        spec: The role this node runs.
        deps: The incident's shared resources.

    Returns:
        An async function the graph can call with the state.
    """

    async def node(state: IncidentState) -> StateDelta:
        delta: dict[str, Any] = {}

        handback = pending_handback(state)
        if handback is not None and handback.target.value == spec.name:
            # Charged here, immediately before the role runs, so the check and the spend
            # cannot disagree and a role that then fails still consumes its retry.
            delta.update(apply_handback(state, handback, deps.policy))
            state = _merged(state, delta)
        else:
            handback = None

        if spec.name == "verify" and _is_recheck(state):
            delta.update(RetryBudget.of(state, deps.policy).spend_recheck().to_state())
            state = _merged(state, delta)

        run_ids = state.get("run_ids") or []
        previous = run_ids[-1] if run_ids else None
        run_id = deps.runs.start(
            session_id=state.get("session_id", ""), role=spec.name, after=previous
        )
        deps.journal.record(StageEvent(stage=spec.name, event="entered"))

        context = deps.build_context(spec, state, handback, run_id)
        try:
            outcome = await spec.run(context)
        except ApprovalPending:
            # Not a failure. The pipeline catches it, checkpoints and exits cleanly, so
            # the run row stays open for the resume to close.
            raise
        except Exception as exc:
            deps.runs.finish(
                run_id,
                status="failed",
                output=None,
                trace_summary=None,
                error=repr(exc),
                metadata={},
            )
            deps.journal.record(StageEvent(stage=spec.name, event="failed", detail=repr(exc)))
            raise

        return {**delta, **_merge(spec, state, outcome, run_id, deps)}

    return node


def _merged(state: IncidentState, delta: Mapping[str, Any]) -> IncidentState:
    """Apply a delta to the state the role will be entered with.

    A partial update is a plain mapping while the state is a typed one, so the cast is
    made once here rather than at each of the call sites that need it.
    """
    return cast(IncidentState, {**state, **delta})


def _is_recheck(state: IncidentState) -> bool:
    """Say whether verification is being entered to look a second time.

    The only way the graph routes verification back to itself is an inconclusive verdict,
    so the presence of one in the incoming state is what distinguishes a re-check from a
    first pass. The prototype incremented on every pass instead, which spent the
    allowance before any re-check had happened.
    """
    result = state.get("verify_result")
    return result is not None and result.verdict == "inconclusive"


def _merge(
    spec: RoleSpec,
    state: IncidentState,
    outcome: RoleOutcome,
    run_id: str,
    deps: IncidentDeps,
) -> StateDelta:
    """Turn a role's outcome into a state delta, uniformly."""
    artifact = outcome.artifact
    halted = bool(getattr(artifact, "halt", False))
    summary = build_trace_summary(
        spec.name, transcript=outcome.transcript, output=artifact, error=outcome.error
    )

    delta: dict[str, Any] = {
        "messages": {**state.get("messages", {}), spec.name: outcome.transcript},
        "trace_summaries": [*state.get("trace_summaries", []), summary.model_dump(mode="json")],
        "run_ids": [*state.get("run_ids", []), run_id],
    }
    if spec.state_slot is not None:
        delta[spec.state_slot] = artifact

    if artifact is None and spec.failure_sets_error:
        # Only the roles that declare it. Verification deliberately does not: a check
        # that could not run must not invalidate a remediation that applied cleanly.
        delta["error"] = f"{spec.name} failed: {outcome.error or 'no submission'}"

    status = "succeeded" if artifact is not None else "failed"
    if halted:
        status = "halted"
    deps.runs.finish(
        run_id,
        status=status,
        output=artifact.model_dump(mode="json") if artifact is not None else None,
        trace_summary=summary.model_dump(mode="json"),
        error=outcome.error,
        metadata=dict(outcome.metadata),
    )

    if artifact is not None or outcome.error:
        deps.journal.record(
            SubmissionEvent(
                artifact=spec.submit_tool,
                accepted=artifact is not None,
                attempt=1,
                validation_error=outcome.error,
            )
        )
    deps.journal.record(
        StageEvent(
            stage=spec.name,
            event="halted" if halted else "exited",
            detail=outcome.error or "",
        )
    )

    # The role's own keys last, so a role can add state the wrapper does not know about.
    delta.update(outcome.delta)
    return delta
