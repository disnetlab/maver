"""Artifact schemas and the invariants their validators enforce.

Every validator here was added after a measured failure. The tests name the failure
so that a future reader can tell a load-bearing rule from an incidental one.
"""

from typing import Any

import pytest
from pydantic import ValidationError

from kira.contracts import ExecutionResult, Plan, RootCause, VerifyResult


def _root_cause(**overrides: Any) -> dict[str, Any]:
    return {
        "fault_type": "misconfiguration",
        "affected_resource": "test-ns/Service/frontend",
        "confidence": 0.85,
        "evidence": ["endpoints empty for 10m", "selector app=frontnd"],
        "summary": "The Service selector no longer matches any pod.",
        **overrides,
    }


def _verify(**overrides: Any) -> dict[str, Any]:
    return {
        "recovered": True,
        "verdict": "recovered",
        "checks": ["curl http://frontend:8080/ returned 200 for 20/20 requests"],
        "summary": "The entry path serves traffic again.",
        "entry_point_status": "Probed the LoadBalancer Service; 200 OK.",
        "confidence": 0.9,
        **overrides,
    }


class TestRootCause:
    def test_accepts_a_complete_diagnosis(self) -> None:
        rc = RootCause(**_root_cause())
        assert rc.halt is False
        assert rc.halt_reason is None

    def test_literal_null_halt_reason_does_not_read_as_a_halt_signal(self) -> None:
        """A literal "null" string here once short-circuited the pipeline."""
        rc = RootCause(**_root_cause(halt_reason="null"))
        assert rc.halt_reason is None

    def test_evidence_submitted_as_a_string_is_decoded(self) -> None:
        rc = RootCause(**_root_cause(evidence='["a", "b"]'))
        assert rc.evidence == ["a", "b"]

    def test_confidence_is_bounded(self) -> None:
        with pytest.raises(ValidationError):
            RootCause(**_root_cause(confidence=1.5))


class TestPlan:
    def test_steps_submitted_as_a_string_are_decoded(self) -> None:
        """Cost four graded mitigations in the era-2 baseline before it was handled."""
        plan = Plan(
            steps='[{"action": "patch", "target": "svc/frontend", '
            '"params": {"selector": {"app": "frontend"}}, "justification": "fix selector"}]',
            expected_outcome="Endpoints repopulate.",
            confidence=0.8,
        )
        assert len(plan.steps) == 1
        assert plan.steps[0].action == "patch"
        assert plan.steps[0].params == {"selector": {"app": "frontend"}}

    def test_params_default_to_empty(self) -> None:
        plan = Plan(
            steps=[{"action": "restart", "target": "deploy/api", "justification": "j"}],
            expected_outcome="Pods restart cleanly.",
            confidence=0.5,
        )
        assert plan.steps[0].params == {}


class TestExecutionResult:
    def test_replan_handback_requires_a_reason(self) -> None:
        """A contentless hand-back routes back to Planning with no context."""
        with pytest.raises(ValidationError, match="replan_reason is required"):
            ExecutionResult(
                succeeded=False,
                steps=[{"step_index": 0, "succeeded": False, "observation": "errored"}],
                final_state_summary="Nothing applied.",
                needs_replan=True,
                confidence=0.3,
            )

    def test_replan_handback_with_a_reason_is_accepted(self) -> None:
        result = ExecutionResult(
            succeeded=False,
            steps=[{"step_index": 0, "succeeded": False, "observation": "errored"}],
            final_state_summary="Nothing applied.",
            needs_replan=True,
            replan_reason="The Deployment has no container named 'web'.",
            confidence=0.3,
        )
        assert result.needs_replan

    def test_a_clean_apply_needs_no_reason(self) -> None:
        result = ExecutionResult(
            succeeded=True,
            steps=[{"step_index": 0, "succeeded": True, "observation": "patched"}],
            final_state_summary="Selector corrected; rollout complete.",
            confidence=0.9,
        )
        assert result.replan_reason is None


class TestVerifyResult:
    def test_accepts_a_grounded_recovery(self) -> None:
        assert VerifyResult(**_verify()).recovered

    def test_recovery_requires_an_independent_entry_point_observation(self) -> None:
        """Without it, a wrong diagnosis certifies its own recovery."""
        with pytest.raises(ValidationError, match="entry_point_status"):
            VerifyResult(**_verify(entry_point_status=""))

    def test_a_handback_does_not_require_the_entry_point_probe(self) -> None:
        """Namespaces where a probe is impossible are the ones worth handing back.

        Demanding the anchor here cost a correct not_recovered verdict on
        rbac_misconfiguration: the run was marked failed and the cell graded as a
        mitigation failure.
        """
        result = VerifyResult(
            **_verify(
                recovered=False,
                verdict="not_recovered",
                entry_point_status="",
                needs_replan=True,
                replan_reason="The RoleBinding still lacks the get verb.",
            )
        )
        assert not result.recovered

    def test_recovery_cannot_rest_on_pod_status_alone(self) -> None:
        """Workloads go healthy precisely when the fix hit the wrong object."""
        with pytest.raises(ValidationError, match="request path"):
            VerifyResult(**_verify(checks=["kubectl get pods shows 3/3 Running"]))

    def test_verdict_must_agree_with_the_boolean(self) -> None:
        with pytest.raises(ValidationError):
            VerifyResult(**_verify(verdict="inconclusive"))
        with pytest.raises(ValidationError):
            VerifyResult(**_verify(recovered=False, verdict="recovered"))

    def test_a_recovered_run_does_not_hand_back(self) -> None:
        with pytest.raises(ValidationError):
            VerifyResult(**_verify(needs_replan=True, replan_reason="r"))

    def test_hand_backs_are_mutually_exclusive(self) -> None:
        with pytest.raises(ValidationError, match="mutually exclusive"):
            VerifyResult(
                **_verify(
                    recovered=False,
                    verdict="not_recovered",
                    needs_replan=True,
                    replan_reason="r",
                    needs_reassessment=True,
                    reassessment_reason="r",
                )
            )

    def test_not_recovered_must_name_a_way_back(self) -> None:
        """Six of 21 not_recovered verdicts once ended the incident silently."""
        with pytest.raises(ValidationError, match="requires a hand-back"):
            VerifyResult(**_verify(recovered=False, verdict="not_recovered"))

    def test_inconclusive_needs_no_hand_back(self) -> None:
        result = VerifyResult(
            **_verify(recovered=False, verdict="inconclusive", entry_point_status="")
        )
        assert result.verdict == "inconclusive"

    def test_reassessment_requires_its_reason(self) -> None:
        with pytest.raises(ValidationError, match="reassessment_reason is required"):
            VerifyResult(
                **_verify(recovered=False, verdict="not_recovered", needs_reassessment=True)
            )
