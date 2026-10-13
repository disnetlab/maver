"""Persisted scheduler references resolve to MAVER's runtime entry points."""

import pytest
from apscheduler.util import ref_to_obj

from maver.orchestration import jobs, learning


@pytest.mark.parametrize(
    "module,name",
    [
        ("jobs", "expire_approval"),
        ("jobs", "resume_approval"),
        ("jobs", "sweep_decided_approvals"),
        ("learning", "distil_incident"),
    ],
)
def test_persisted_scheduler_references_resolve_to_the_maver_runtime(
    module: str, name: str
) -> None:
    current = ref_to_obj(f"maver.orchestration.{module}:{name}")
    expected = getattr(jobs if module == "jobs" else learning, name)
    assert current is expected
