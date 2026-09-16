"""Where the pipeline goes after each role, and what a back-edge costs.

Four pure predicates over the artifacts. They read no configuration of their own — the
policy is resolved once per incident and passed in — so the same state and the same policy
always produce the same route, whether the run is in memory or rebuilt from a database.

The routing decisions here are the ones a reader will otherwise assume are arbitrary, so
the reasoning stays attached to them.
"""

from dataclasses import dataclass

from maver.contracts.routing import HandBack, Route
from maver.contracts.state import IncidentState
from maver.orchestration.budget import RetryBudget
from maver.roles.base import StateDelta
from maver.settings.root import Settings

__all__ = [
    "RoutingPolicy",
    "after_execution",
    "after_planning",
    "after_rca",
    "after_verify",
    "apply_handback",
    "pending_handback",
]


@dataclass(frozen=True, slots=True)
class RoutingPolicy:
    """Everything routing needs from configuration, resolved once per incident."""

    verify_enabled: bool
    remediation_retries: int
    inconclusive_retries: int
    verify_handback_retries: int | None = None

    @property
    def verify_handback_limit(self) -> int:
        """Resolve the Verify subset cap without widening the shared budget."""
        requested = self.verify_handback_retries
        return (
            self.remediation_retries
            if requested is None
            else min(self.remediation_retries, requested)
        )

    @classmethod
    def from_settings(cls, settings: Settings) -> "RoutingPolicy":
        """Resolve the policy from configuration.

        Args:
            settings: The run's settings.

        Returns:
            The routing policy.
        """
        return cls(
            verify_enabled=settings.verify.enabled,
            remediation_retries=settings.pipeline.remediation_retries,
            inconclusive_retries=settings.verify.inconclusive_retries,
            verify_handback_retries=settings.verify.handback_retries,
        )


def after_rca(state: IncidentState, policy: RoutingPolicy) -> Route:
    """A diagnosis that is missing or halted ends the incident; anything else plans.

    Args:
        state: The incident state.
        policy: The routing policy.

    Returns:
        The next route.
    """
    # getattr rather than attribute access: this slot may hold the single-agent
    # configuration's diagnosis, which has no halt field and is treated as a normal one.
    root_cause = state.get("root_cause")
    if not root_cause or getattr(root_cause, "halt", False):
        return Route.END
    return Route.PLANNING


def after_planning(state: IncidentState, policy: RoutingPolicy) -> Route:
    """A plan that is missing or halted ends the incident; anything else executes.

    Args:
        state: The incident state.
        policy: The routing policy.

    Returns:
        The next route.
    """
    plan = state.get("plan")
    if not plan or getattr(plan, "halt", False):
        return Route.END
    return Route.EXECUTION


def after_execution(state: IncidentState, policy: RoutingPolicy) -> Route:
    """Route a clean apply onward, and a failed apply back to planning.

    The edge back to planning exists in every configuration, including those without
    verification. It fires when the apply itself could not be carried out, which is a
    different signal from "the fix did not work": the diagnosis may still be right while
    the plan could not be executed as written.

    Args:
        state: The incident state.
        policy: The routing policy.

    Returns:
        The next route.
    """
    result = state.get("execution_result")
    if result is None:
        return Route.END
    if result.succeeded:
        return Route.VERIFY if policy.verify_enabled else Route.END
    if RetryBudget.of(state, policy).remediation_left and getattr(result, "needs_replan", False):
        return Route.PLANNING
    return Route.END


def after_verify(state: IncidentState, policy: RoutingPolicy) -> Route:
    """Route on the recovery verdict.

    Args:
        state: The incident state.
        policy: The routing policy.

    Returns:
        The next route.
    """
    result = state.get("verify_result")
    if result is None or result.recovered:
        return Route.END

    # "Could not tell" is not "done". Verification is read-only, so one more look costs a
    # few telemetry reads and cannot damage the cluster — cheaper than closing an incident
    # whose symptom may still be live. Bounded separately, and kept off the remediation
    # budget, which exists to cap mutations.
    budget = RetryBudget.of(state, policy)
    if result.verdict == "inconclusive":
        return Route.VERIFY if budget.recheck_left else Route.END
    if not budget.remediation_left or not budget.verify_handback_left:
        return Route.END
    if result.needs_reassessment:
        return Route.RCA
    if result.needs_replan:
        return Route.PLANNING
    # Unreachable through the schema: a not-recovered verdict is rejected at validation
    # unless it carries a hand-back. Kept as a defensive default and pinned by a test.
    return Route.END


def pending_handback(state: IncidentState) -> HandBack | None:
    """Say whether this entry is a back-edge, and why.

    Derived from the artifacts alone rather than from a flag written when the edge was
    taken, so the router and the node receiving it cannot disagree, and so a resumed run
    reaches the same conclusion as one that never stopped.

    Args:
        state: The incident state.

    Returns:
        The hand-back being acted on, or None on a first entry.
    """
    verify = state.get("verify_result")
    if verify is not None and verify.needs_reassessment and state.get("root_cause") is not None:
        return HandBack(
            Route.RCA, verify.reassessment_reason or verify.summary or "", "reassessment"
        )
    if verify is not None and verify.needs_replan:
        return HandBack(
            Route.PLANNING, verify.replan_reason or verify.summary or "", "replan_from_verify"
        )
    execution = state.get("execution_result")
    if (
        execution is not None
        and not execution.succeeded
        and getattr(execution, "needs_replan", False)
    ):
        return HandBack(
            Route.PLANNING,
            getattr(execution, "replan_reason", "") or _execution_summary(execution),
            "replan_from_execution",
        )
    return None


def apply_handback(state: IncidentState, handback: HandBack, policy: RoutingPolicy) -> StateDelta:
    """Charge a back-edge and record what the destination role is retrying.

    Both halves happen here, in one place, immediately before the destination runs. The
    prototype checked the budget in the router and spent it inside two role modules, so
    a role that failed before its own increment leaked a free retry, and answering "what
    does a retry cost" meant reading three files.

    Args:
        state: The state the destination is being entered with.
        handback: The back-edge being taken.
        policy: The routing policy.

    Returns:
        The counter and history updates to merge before the role runs.
    """
    budget = RetryBudget.of(state, policy)
    delta: dict[str, object] = {}
    if handback.kind in {"reassessment", "replan_from_verify"}:
        budget = budget.spend_verify_handback()

    if handback.kind == "reassessment":
        budget = budget.spend_reassessment().reset_rechecks()
        root_cause = state.get("root_cause")
        if root_cause is not None:
            delta["prior_attempts"] = [
                *state.get("prior_attempts", []),
                _attempt_record(state, handback),
            ]
        # A new diagnosis supersedes the plan built on the old one and everything that
        # plan produced. They are summarised into prior_attempts first, so nothing is
        # lost; what would be wrong is leaving them live.
        delta["plan"] = None
        delta["execution_result"] = None
    else:
        budget = budget.spend_replan().reset_rechecks()
        plan = state.get("plan")
        if plan is not None:
            delta["prior_plans"] = [
                *state.get("prior_plans", []),
                {"plan": plan.model_dump(mode="json"), "reason": handback.reason},
            ]

    # The verdict has been acted on, so it stops being a pending instruction.
    #
    # This is the whole of the defect. A hand-back is derived from the artifacts rather
    # than from a flag, which is what lets a resumed run reach the same conclusion as one
    # that never stopped — but a verdict left in place is re-derived at every subsequent
    # node entry. A reassessment kept naming rca, so planning read a hand-back addressed
    # to someone else, treated it as none, and charged nothing: the budget never fell and
    # planning and execution cycled until the step limit killed the incident, each cycle
    # a fresh mutation of the cluster. Consuming it and leaving it is the same class of
    # mistake as reading a flag nobody clears.
    delta["verify_result"] = None

    delta.update(budget.to_state())
    return delta


def _execution_summary(result: object) -> str:
    """Read an execution outcome's own account of where it left the cluster.

    The two result shapes name it differently: the pipeline's calls it a final state
    summary, the single-agent configuration's calls it a summary.
    """
    return str(getattr(result, "final_state_summary", "") or getattr(result, "summary", "") or "")


def _attempt_record(state: IncidentState, handback: HandBack) -> dict[str, object]:
    """Summarise the diagnosis being handed back, for the re-diagnosis prompt."""
    root_cause = state.get("root_cause")
    execution = state.get("execution_result")
    return {
        "root_cause": root_cause.model_dump(mode="json") if root_cause is not None else None,
        "execution_summary": _execution_summary(execution) if execution is not None else "",
        "reason": handback.reason,
    }
