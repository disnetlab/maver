"""Later operator verdicts keep their own history and do not rewrite observations."""

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
from pydantic import ValidationError

from maver.api.feedback import FeedbackRequest
from maver.contracts.lifecycle import LifecycleState, RunStatus
from maver.memory.backends.sqlite import SqliteStore
from maver.memory.lifecycle import IllegalTransition
from maver.memory.ports import SessionMemory
from maver.orchestration.outcome_feedback import latest_feedback_packet, submit_feedback


def test_operator_review_updates_lifecycle_and_preserves_original_artifacts() -> None:
    """A correction is private metadata; evidence and the RCA artifact remain original."""
    with closing(SqliteStore(":memory:")) as store:
        session = store.create_session({"alertname": "bad-port"})
        store.update_artefacts(session.session_id, root_cause={"cause": "wrong image"})
        store.set_run_status(session.session_id, RunStatus.COMPLETED)
        store.set_state(session.session_id, LifecycleState.TENTATIVE_SUCCESS)
        store.records.upsert_summary(
            SessionMemory(
                session_id=session.session_id,
                evidence_text="pods failed readiness",
                lesson_text="image was wrong",
                incident_summary="failed readiness",
                lifecycle_state=LifecycleState.TENTATIVE_SUCCESS.value,
                fault_category="configuration",
                alert_name="bad-port",
                namespace="default",
                affected_resource="service/web",
            )
        )

        feedback = submit_feedback(
            store,
            store.records,
            session_id=session.session_id,
            verdict="corrected",
            decided_by="on-call",
            reason="inspected endpoints",
            evidence="targetPort was 9999",
            corrected_cause="wrong targetPort",
        )
        current = store.get_session(session.session_id)
        memory = store.records.open_record(session.session_id)
        assert current is not None and memory is not None
        assert current.state == LifecycleState.INVALIDATED.value
        assert current.root_cause == {"cause": "wrong image"}
        assert memory.evidence_text == "pods failed readiness"
        assert memory.corrected_cause == "wrong targetPort"
        assert memory.lifecycle_state == LifecycleState.INVALIDATED.value
        assert store.feedback_for(session.session_id) == [feedback]
        # A delayed Learning write assembled before review cannot restore the old
        # lifecycle or erase the private correction.
        store.records.upsert_summary(
            SessionMemory(
                session_id=session.session_id,
                evidence_text="pods failed readiness",
                lesson_text="image was wrong",
                incident_summary="failed readiness",
                lifecycle_state=LifecycleState.TENTATIVE_SUCCESS.value,
                fault_category="configuration",
                alert_name="bad-port",
                namespace="default",
                affected_resource="service/web",
            )
        )
        after_late_write = store.records.open_record(session.session_id)
        assert after_late_write is not None
        assert after_late_write.lifecycle_state == LifecycleState.INVALIDATED.value
        assert after_late_write.corrected_cause == "wrong targetPort"
        assert latest_feedback_packet(store, session.session_id) == {
            "source": "operator",
            "operator": "on-call",
            "reason": "inspected endpoints",
            "evidence": "targetPort was 9999",
            "mitigation": {"success": False},
            "diagnosis": {"success": False},
            "ground_truth_root_cause": "wrong targetPort",
        }

        submit_feedback(
            store,
            store.records,
            session_id=session.session_id,
            verdict="confirmed",
            decided_by="reviewer",
            reason="fix verified",
            evidence="error rate recovered",
        )
        assert store.get_session(session.session_id).state == LifecycleState.HUMAN_VERIFIED.value
        assert len(store.feedback_for(session.session_id)) == 2


def test_feedback_refuses_unfinished_runs_and_invalid_transition() -> None:
    """A human outcome is distinct from a write approval during the run."""
    with closing(SqliteStore(":memory:")) as store:
        session = store.create_session({})
        fields = {
            "session_id": session.session_id,
            "verdict": "confirmed",
            "decided_by": "ops",
            "reason": "checked",
            "evidence": "metrics recovered",
        }
        with pytest.raises(ValueError, match="not complete"):
            submit_feedback(store, store.records, **fields)
        store.set_run_status(session.session_id, RunStatus.COMPLETED)
        with pytest.raises(IllegalTransition):
            submit_feedback(store, store.records, **fields)
        assert store.feedback_for(session.session_id) == []


def test_corrected_verdict_requires_cause() -> None:
    with pytest.raises(ValidationError, match="corrected_cause"):
        FeedbackRequest(verdict="corrected", decided_by="ops", reason="reviewed", evidence="logs")


def test_existing_v4_store_gains_feedback_history(tmp_path: Path) -> None:
    """The layout migration reaches a stamped database with existing incidents."""
    path = tmp_path / "memory.db"
    with closing(SqliteStore(path)) as store:
        session = store.create_session({"alertname": "existing"})
        store.set_run_status(session.session_id, RunStatus.COMPLETED)
        store.set_state(session.session_id, LifecycleState.TENTATIVE_SUCCESS)
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("DROP TABLE outcome_feedback")
        conn.execute("UPDATE schema_version SET version = 4")
        conn.commit()
    with closing(SqliteStore(path)) as reopened:
        submit_feedback(
            reopened,
            reopened.records,
            session_id=session.session_id,
            verdict="confirmed",
            decided_by="ops",
            reason="observed recovery",
            evidence="request rate normal",
        )
        assert reopened.get_session(session.session_id).state == LifecycleState.HUMAN_VERIFIED.value
