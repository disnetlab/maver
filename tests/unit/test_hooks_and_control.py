"""Scheduling what happens after an incident, and tracking the runs in flight."""

import pathlib
from typing import Any

import pytest

from kira.orchestration import hooks
from kira.orchestration.control import RunRegistry
from kira.orchestration.hooks import PostIncidentHook, fire_hooks, register_hook
from kira.orchestration.outcomes import Outcome


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
