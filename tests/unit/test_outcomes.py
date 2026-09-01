"""The nine-rule outcome table, and what persisting it writes.

The table is the reason this module exists: it is the one decision the prototype made
twice, and every count reported about a run is downstream of it.
"""

from collections.abc import Iterator

import pytest

from kira.contracts.artifacts import ExecutionResult, RootCause, VerifyResult
from kira.contracts.lifecycle import LifecycleState, RunStatus
from kira.contracts.state import IncidentState
from kira.memory.backends.sqlite import SqliteStore
from kira.orchestration.outcomes import Outcome, persist_outcome, resolve_outcome


def root_cause(**kw) -> RootCause:
    return RootCause(
        fault_type="misconfig",
        affected_resource="deploy/web",
        confidence=0.8,
        evidence=["probe fails"],
        summary="readiness probe points at the wrong port",
        **kw,
    )


def execution(succeeded: bool) -> ExecutionResult:
    return ExecutionResult(
        succeeded=succeeded, steps=[], final_state_summary="applied", confidence=0.8
    )


def verify(**kw) -> VerifyResult:
    kw.setdefault("checks", ["curl http://checkout.shop/health returned 200"])
    kw.setdefault("summary", "the checkout path serves again")
    kw.setdefault("confidence", 0.7)
    kw.setdefault("recovered", False)
    if kw.get("recovered"):
        kw.setdefault("entry_point_status", "the checkout page returned 200")
    if kw.get("needs_replan"):
        kw.setdefault("replan_reason", "the plan left the failing binding in place")
    return VerifyResult(**kw)


def state(**kw) -> IncidentState:
    base: IncidentState = {"session_id": "s1", "alert": {}}
    base.update(kw)  # type: ignore[typeddict-item]
    return base


@pytest.fixture
def store() -> Iterator[SqliteStore]:
    opened = SqliteStore(":memory:")
    yield opened
    opened.close()


# --- the decision table -------------------------------------------------------------


@pytest.mark.parametrize(
    ("rule", "incident", "expected"),
    [
        (
            "1: an error invalidates the run",
            state(error="the model endpoint refused every call"),
            (Outcome.ERROR, LifecycleState.INVALIDATED, RunStatus.FAILED),
        ),
        (
            "2: a halt is a deliberate stop, so the run completed",
            state(root_cause=root_cause(halt=True, halt_reason="telemetry was unreachable")),
            (Outcome.HALTED, LifecycleState.HALTED, RunStatus.COMPLETED),
        ),
        (
            "3: a diagnosis-only task ends on the diagnosis",
            state(target_stage="diagnose", root_cause=root_cause()),
            (Outcome.SUCCESS, LifecycleState.TENTATIVE_SUCCESS, RunStatus.COMPLETED),
        ),
        (
            "4: a diagnosis-only task with no diagnosis failed",
            state(target_stage="diagnose"),
            (Outcome.FAILURE, LifecycleState.INVALIDATED, RunStatus.FAILED),
        ),
        (
            "5: a confirmed recovery",
            state(verify_result=verify(recovered=True, verdict="recovered")),
            (Outcome.SUCCESS, LifecycleState.TENTATIVE_SUCCESS, RunStatus.COMPLETED),
        ),
        (
            "6: a disproven fix",
            state(verify_result=verify(verdict="not_recovered", needs_replan=True)),
            (Outcome.FAILURE, LifecycleState.INVALIDATED, RunStatus.FAILED),
        ),
        (
            "7: execution applied its plan and nothing checked it",
            state(execution_result=execution(succeeded=True)),
            (Outcome.SUCCESS, LifecycleState.TENTATIVE_SUCCESS, RunStatus.COMPLETED),
        ),
        (
            "8: execution reported its own failure",
            state(execution_result=execution(succeeded=False)),
            (Outcome.FAILURE, LifecycleState.INVALIDATED, RunStatus.FAILED),
        ),
        (
            "9: nothing decisive, so no lifecycle to write",
            state(root_cause=root_cause()),
            (Outcome.FAILURE, None, None),
        ),
    ],
)
def test_each_rule_of_the_outcome_table(
    rule: str,
    incident: IncidentState,
    expected: tuple[Outcome, LifecycleState | None, RunStatus | None],
) -> None:
    resolved = resolve_outcome(incident)
    assert (resolved.outcome, resolved.lifecycle, resolved.run_status) == expected, rule


def test_an_unconfirmed_recovery_is_still_a_tentative_success() -> None:
    """Inconclusive means the change applied but recovery could not be established."""
    resolved = resolve_outcome(state(verify_result=verify(verdict="inconclusive")))
    assert resolved.outcome is Outcome.SUCCESS
    assert resolved.notes == "verify: inconclusive"


# --- where the order of the rules is load-bearing ------------------------------------


def test_the_recovery_verdict_outranks_the_claim_that_the_fix_was_applied() -> None:
    """Rule 6 before rule 7: Execution says it applied a change, Verify says it did not work."""
    resolved = resolve_outcome(
        state(
            execution_result=execution(succeeded=True),
            verify_result=verify(verdict="not_recovered", needs_replan=True),
        )
    )
    assert resolved.outcome is Outcome.FAILURE
    assert resolved.notes == "verify: not_recovered"


def test_an_error_outranks_every_artefact_the_run_produced() -> None:
    resolved = resolve_outcome(
        state(
            error="boom",
            root_cause=root_cause(halt=True, halt_reason="telemetry was unreachable"),
            execution_result=execution(succeeded=True),
            verify_result=verify(recovered=True, verdict="recovered"),
        )
    )
    assert resolved.outcome is Outcome.ERROR
    assert resolved.notes == "boom"


# --- persistence ---------------------------------------------------------------------


def test_an_undecided_run_is_left_holding_its_artefacts_and_nothing_else(
    store: SqliteStore,
) -> None:
    """Rule 9 keeps the diagnosis but writes no transition over it."""
    session = store.create_session({})
    incident = state(session_id=session.session_id, root_cause=root_cause())

    persist_outcome(incident, resolve_outcome(incident), store)

    read = store.get_session(session.session_id)
    assert read is not None
    assert read.root_cause is not None
    assert read.state == LifecycleState.PENDING.value
    assert read.run_status == RunStatus.PENDING.value


def test_a_deferred_run_status_records_the_lifecycle_but_keeps_the_run_open(
    store: SqliteStore,
) -> None:
    """An adapter withholds "done" until its own submission has landed."""
    session = store.create_session({})
    incident = state(
        session_id=session.session_id,
        execution_result=execution(succeeded=True),
        verify_result=verify(recovered=True, verdict="recovered"),
    )

    persist_outcome(incident, resolve_outcome(incident), store, finalize_run_status=False)

    read = store.get_session(session.session_id)
    assert read is not None
    assert read.verify_result is not None
    assert read.state == LifecycleState.TENTATIVE_SUCCESS.value
    assert read.run_status == RunStatus.RUNNING.value
