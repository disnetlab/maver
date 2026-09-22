"""A late operator review must not allow a stale lesson into the corpus."""

import pathlib

import pytest
from test_learning_job import closed_incident, deployment

from maver.contracts.lifecycle import RunStatus
from maver.orchestration.learning import run_learning
from maver.orchestration.outcome_feedback import submit_feedback


async def test_review_committing_after_learning_check_prevents_stale_write(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    deploy, model = deployment(tmp_path)
    sessions = deploy.storage.sessions
    session_id = closed_incident(deploy)
    sessions.set_run_status(session_id, RunStatus.COMPLETED)
    index = deploy.storage.index
    upsert = index.upsert_summary

    def review_then_write(memory, **kwargs):
        submit_feedback(
            sessions,
            session_id=session_id,
            verdict="corrected",
            decided_by="on-call",
            reason="Fresh evidence arrived",
            evidence="The service selector was wrong",
            corrected_cause="Incorrect service selector",
        )
        upsert(memory, **kwargs)

    monkeypatch.setattr(index, "upsert_summary", review_then_write)
    try:
        assert await run_learning(session_id, deploy) == "feedback_changed"
        assert index.open_record(session_id) is None
        assert not sessions.get_session(session_id).distilled
        monkeypatch.setattr(index, "upsert_summary", upsert)
        assert await run_learning(session_id, deploy) == "distilled"
        assert model.calls == 2
        assert index.open_record(session_id).corrected_cause == "Incorrect service selector"
    finally:
        deploy.storage.close()
