from kira.contracts.artifacts import ExecutionResult, Plan, PlanStep, RootCause, VerifyResult
from kira.contracts.routing import Route
from kira.contracts.state import IncidentState
from kira.orchestration.budget import RetryBudget
from kira.orchestration.routing import (
    RoutingPolicy,
    after_execution,
    after_planning,
    after_rca,
    after_verify,
    apply_handback,
    pending_handback,
)

POLICY = RoutingPolicy(verify_enabled=True, remediation_retries=2, inconclusive_retries=1)
NO_VERIFY = RoutingPolicy(verify_enabled=False, remediation_retries=2, inconclusive_retries=1)


def root_cause(**kw) -> RootCause:
    return RootCause(
        fault_type="misconfig",
        affected_resource="deploy/web",
        confidence=0.8,
        evidence=["probe fails"],
        summary="readiness probe points at the wrong port",
        **kw,
    )


def plan(**kw) -> Plan:
    return Plan(
        steps=[
            PlanStep(
                action="kubectl patch deploy web",
                target="deploy/web",
                justification="point the readiness probe at the serving port",
            )
        ],
        expected_outcome="pods become ready",
        confidence=0.8,
        **kw,
    )


def execution(succeeded: bool, **kw) -> ExecutionResult:
    kw.setdefault("confidence", 0.8)
    if kw.get("needs_replan"):
        kw.setdefault("replan_reason", "the patch target did not exist")
    return ExecutionResult(succeeded=succeeded, steps=[], final_state_summary="applied", **kw)


def verify(**kw) -> VerifyResult:
    kw.setdefault(
        "checks",
        ["curl http://checkout.shop/health returned 200; error rate 0.00 over 5m"],
    )
    kw.setdefault("confidence", 0.7)
    if kw.get("needs_replan"):
        kw.setdefault("replan_reason", "the plan did not address the failing binding")
    if kw.get("recovered"):
        # Recovery must be claimed from an observed entry point, not from the causal
        # chain the diagnosis proposed.
        kw.setdefault("entry_point_status", "the checkout page returned 200")
    if kw.get("needs_reassessment"):
        kw.setdefault("reassessment_reason", "the named component was not the origin")
    return VerifyResult(**kw)


def state(**kw) -> IncidentState:
    base: IncidentState = {"session_id": "s1", "alert": {}}
    base.update(kw)  # type: ignore[typeddict-item]
    return base


# --- the edge table -----------------------------------------------------------------


def test_a_halted_diagnosis_ends_the_incident() -> None:
    """Halting is a role's deliberate stop with a stated reason, not a failure to answer."""
    assert after_rca(state(root_cause=root_cause()), POLICY) is Route.PLANNING
    assert (
        after_rca(state(root_cause=root_cause(halt=True, halt_reason="no access")), POLICY)
        is Route.END
    )
    assert after_rca(state(), POLICY) is Route.END


def test_a_missing_or_halted_plan_ends_the_incident() -> None:
    assert after_planning(state(plan=plan()), POLICY) is Route.EXECUTION
    assert after_planning(state(), POLICY) is Route.END


def test_a_clean_apply_verifies_only_where_verification_is_enabled() -> None:
    applied = state(execution_result=execution(True))
    assert after_execution(applied, POLICY) is Route.VERIFY
    assert after_execution(applied, NO_VERIFY) is Route.END


def test_a_failed_apply_returns_to_planning_in_every_configuration() -> None:
    """A failed apply is a different signal from a fix that did not work.

    The diagnosis may still be right while the plan could not be carried out as written,
    so this edge exists even where verification is off.
    """
    failed = state(execution_result=execution(False, needs_replan=True))
    assert after_execution(failed, NO_VERIFY) is Route.PLANNING

    spent = state(execution_result=execution(False, needs_replan=True), remediation_retries=2)
    assert after_execution(spent, NO_VERIFY) is Route.END


def test_recovery_ends_and_a_wrong_cause_re_diagnoses() -> None:
    assert (
        after_verify(
            state(verify_result=verify(recovered=True, verdict="recovered", summary="ok")), POLICY
        )
        is Route.END
    )

    wrong = state(
        root_cause=root_cause(),
        verify_result=verify(
            recovered=False,
            verdict="not_recovered",
            summary="still failing",
            needs_reassessment=True,
            reassessment_reason="the probe was never the cause",
        ),
    )
    assert after_verify(wrong, POLICY) is Route.RCA


def test_inconclusive_looks_once_more_then_stops() -> None:
    """Could-not-tell is not done, and looking again cannot damage anything.

    Verification is read-only, so a second look costs a few telemetry reads — cheaper
    than closing an incident whose symptom may still be live.
    """
    unsure = verify(recovered=False, verdict="inconclusive", summary="no signal", needs_replan=True)
    assert after_verify(state(verify_result=unsure), POLICY) is Route.VERIFY
    assert after_verify(state(verify_result=unsure, verify_retries=1), POLICY) is Route.END


# --- the budget ---------------------------------------------------------------------


def test_all_three_back_edges_draw_on_one_budget() -> None:
    """Separate budgets would let an incident cycle far longer than the stated bound."""
    budget = RetryBudget.of(state(), POLICY)
    assert budget.remediation_left

    after_replan = budget.spend_replan()
    after_reassess = after_replan.spend_reassessment()
    assert not after_reassess.remediation_left
    # The re-diagnosis count is telemetry over the same budget, not a second allowance.
    assert after_reassess.reassessments == 1
    assert after_reassess.remediation_spent == 2


def test_a_recheck_does_not_spend_the_remediation_budget() -> None:
    """That budget exists to bound mutations, and looking again changes nothing."""
    spent = RetryBudget.of(state(), POLICY).spend_recheck()
    assert spent.remediation_left
    assert not spent.recheck_left


def test_a_new_cycle_may_look_twice_again() -> None:
    """The prototype counted verification passes rather than re-checks, so after a
    replan the counter was already spent and the next cycle's verification could not
    look twice even though it had never looked twice."""
    carried = RetryBudget.of(state(verify_retries=1), POLICY)
    assert not carried.recheck_left
    assert carried.reset_rechecks().recheck_left


def test_a_hand_back_is_charged_where_it_is_applied() -> None:
    """Charged immediately before the destination runs, in one place.

    The prototype checked the budget in the router and spent it inside two role modules,
    so a role that failed before its own increment leaked a free retry.
    """
    handed = state(
        root_cause=root_cause(),
        verify_result=verify(
            recovered=False,
            verdict="not_recovered",
            summary="still failing",
            needs_reassessment=True,
            reassessment_reason="wrong component",
        ),
        verify_retries=1,
    )
    hb = pending_handback(handed)
    assert hb is not None and hb.target is Route.RCA and hb.kind == "reassessment"

    delta = apply_handback(handed, hb, POLICY)
    assert delta["remediation_retries"] == 1
    assert delta["reassessment_count"] == 1
    assert delta["verify_retries"] == 0
    assert len(delta["prior_attempts"]) == 1


def test_a_first_entry_carries_no_hand_back() -> None:
    assert pending_handback(state(root_cause=root_cause())) is None
