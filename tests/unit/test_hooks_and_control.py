"""Scheduled work between incidents, and tracking the runs in flight."""

import dataclasses
import pathlib
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from maver.contracts.lifecycle import RunStatus
from maver.memory.factory import build_storage
from maver.memory.ports import ApprovalRecord
from maver.orchestration import hooks, jobs
from maver.orchestration.control import RunRegistry
from maver.orchestration.hooks import PostIncidentHook, fire_hooks, register_hook
from maver.orchestration.outcomes import Outcome
from maver.orchestration.runner import Deployment
from maver.settings.groups import MemorySettings, ServerSettings
from maver.settings.profiles import load_profile
from maver.settings.root import Settings

PROFILE = load_profile("qwen3.6-35b-a3b", pathlib.Path("configs/models"))


async def distil(session_id: str, **kwargs: Any) -> None:
    """Stands in for the offline pass. Module-level, as every job must be."""


class FakeScheduler:
    def __init__(self) -> None:
        self.jobs: list[dict[str, Any]] = []
        self.fail = False

    def add_job(self, fn: Any, trigger: str, **kwargs: Any) -> None:
        if self.fail:
            raise RuntimeError("job store is unreachable")
        self.jobs.append({"fn": fn, "trigger": trigger, **kwargs})


@pytest.fixture
def scheduler(monkeypatch: pytest.MonkeyPatch) -> FakeScheduler:
    hooks.clear_hooks()
    fake = FakeScheduler()
    monkeypatch.setattr(hooks, "scheduler_for", lambda workdir: fake)
    yield fake
    hooks.clear_hooks()


async def test_a_hook_runs_only_for_the_outcomes_it_names(
    scheduler: FakeScheduler, tmp_path: pathlib.Path
) -> None:
    register_hook(
        PostIncidentHook(
            name="learning",
            job_fn=distil,
            on_outcomes=frozenset({Outcome.SUCCESS, Outcome.FAILURE}),
            delay_seconds=300,
        )
    )

    await fire_hooks(Outcome.ERROR, "s1", workdir=tmp_path)
    assert scheduler.jobs == []

    await fire_hooks(Outcome.SUCCESS, "s1", workdir=tmp_path)
    assert len(scheduler.jobs) == 1
    assert scheduler.jobs[0]["kwargs"]["session_id"] == "s1"


async def test_extra_arguments_reach_the_job_through_the_registry(
    scheduler: FakeScheduler, tmp_path: pathlib.Path
) -> None:
    """So an adapter carrying an external grade goes through the registry rather than
    calling the job directly and bypassing it."""
    register_hook(PostIncidentHook(name="learning", job_fn=distil, on_outcomes=frozenset(Outcome)))

    await fire_hooks(Outcome.SUCCESS, "s1", workdir=tmp_path, external_feedback={"score": 0})

    assert scheduler.jobs[0]["kwargs"]["external_feedback"] == {"score": 0}


async def test_a_scheduling_failure_does_not_fail_the_incident(
    scheduler: FakeScheduler, tmp_path: pathlib.Path
) -> None:
    """The incident is over and its result is already stored. A queue problem must not
    become a lost run."""
    register_hook(PostIncidentHook(name="learning", job_fn=distil, on_outcomes=frozenset(Outcome)))
    scheduler.fail = True

    await fire_hooks(Outcome.SUCCESS, "s1", workdir=tmp_path)


async def test_re_running_an_incident_replaces_its_job_rather_than_queueing_a_second(
    scheduler: FakeScheduler, tmp_path: pathlib.Path
) -> None:
    register_hook(PostIncidentHook(name="learning", job_fn=distil, on_outcomes=frozenset(Outcome)))

    await fire_hooks(Outcome.SUCCESS, "s1", workdir=tmp_path)

    assert scheduler.jobs[0]["id"] == "learning:s1"
    assert scheduler.jobs[0]["replace_existing"] is True


def test_the_registry_forgets_a_run_that_has_finished() -> None:
    """State here is only ever about runs in flight; a finished run has nothing left
    to stop."""
    registry = RunRegistry()
    registry.register("s1")
    assert registry.live() == ["s1"]

    registry.unregister("s1")
    assert registry.live() == []


class _Counting:
    """Wraps a store and records what the sweep asked it for."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.listed: list[str | None] = []
        self.session_reads = 0

    def list_requests(self, *, status: str | None = None) -> Any:
        self.listed.append(status)
        return self.inner.list_requests(status=status)

    def get_session(self, session_id: str) -> Any:
        self.session_reads += 1
        return self.inner.get_session(session_id)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


def _request(request_id: str, session_id: str, *, expires_at: str) -> ApprovalRecord:
    return ApprovalRecord(
        request_id=request_id,
        session_id=session_id,
        kind="tool_call",
        stage="execution",
        status="pending",
        summary="kubectl delete pod checkout-0 -n shop",
        payload={},
        expires_at=expires_at,
    )


async def test_the_approval_sweep_does_not_reread_the_whole_approval_history(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """It ran inline on the event loop, on a repeating schedule, over every request ever
    answered plus a session read for each one.

    Nothing prunes decided approvals, so at a couple of thousand of them over a database
    on another host the loop that serves every incident's model calls, every tool result
    and the readiness probe was held for seconds at a time, once every poll interval, and
    it never came back down. The sessions parked right now are the bounded side of the
    same question, and each one names the single request it is waiting on.
    """
    settings = Settings(
        server=ServerSettings(workdir=tmp_path),
        memory=MemorySettings(enabled=False, learning_enabled=False),
    )
    storage = build_storage(settings)
    approvals, sessions = storage.approvals, storage.sessions
    later = (datetime.now(UTC) + timedelta(hours=1)).isoformat()

    for i in range(20):  # history: answered long ago, on incidents that have finished
        old = sessions.create_session({"alertname": f"old-{i}"})
        approvals.create_request(_request(f"old-{i}", old.session_id, expires_at=later))
        approvals.decide(f"old-{i}", status="approved", decided_by="an operator")
        sessions.set_run_status(old.session_id, RunStatus.COMPLETED)

    decided = sessions.create_session({"alertname": "DecidedElsewhere"})
    approvals.create_request(_request("decided", decided.session_id, expires_at=later))
    sessions.set_run_status(
        decided.session_id, RunStatus.AWAITING_HUMAN, awaiting_request_id="decided"
    )
    approvals.decide("decided", status="approved", decided_by="the CLI, straight to the store")

    overdue = sessions.create_session({"alertname": "NobodyAnswered"})
    approvals.create_request(
        _request(
            "overdue",
            overdue.session_id,
            expires_at=(datetime.now(UTC) - timedelta(hours=2)).isoformat(),
        )
    )
    sessions.set_run_status(
        overdue.session_id, RunStatus.AWAITING_HUMAN, awaiting_request_id="overdue"
    )

    counting = _Counting(approvals)
    counted_sessions = _Counting(sessions)
    deployment = Deployment(
        settings=settings,
        storage=dataclasses.replace(storage, approvals=counting, sessions=counted_sessions),
        completion=None,  # type: ignore[arg-type] - the sweep makes no model call
        profile=PROFILE,
        transport_for=lambda session_id: None,  # type: ignore[arg-type, return-value]
    )
    scheduled: list[str] = []
    monkeypatch.setattr(jobs, "schedule_approval_resume", scheduled.append)
    jobs.bind_deployment(deployment)
    try:
        assert await jobs.sweep_decided_approvals() == "resumed 2"
    finally:
        jobs.unbind_deployment()

    assert sorted(scheduled) == ["decided", "overdue"]
    assert approvals.get_request("overdue").status == "timed_out"
    assert counting.listed == ["pending"], "the answered requests are never listed again"
    assert counted_sessions.session_reads == 0, "nor read one at a time behind them"
    storage.close()
