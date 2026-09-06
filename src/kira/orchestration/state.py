"""Seeding the incident state.

One function, and one rule it exists to enforce: every declared key is written before
the first role runs. The prototype seeded eleven of nineteen and let the rest come into
being through ``state.get(key, default)`` at each reader, which is how two keys came to
be read in three places with two different defaults before anything had ever written
them. A missing key and a key holding a default are indistinguishable to a reader, so
the defaults belong here, once, where they can be seen together.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from kira.contracts.state import IncidentState, TargetStage

__all__ = ["initial_state"]


def initial_state(
    *,
    session_id: str,
    alert: dict[str, Any],
    target_stage: TargetStage,
    incident_dir: Path,
    shared_dir: Path,
    withheld_record_ids: Sequence[str] = (),
) -> IncidentState:
    """Seed every key of the incident state.

    Args:
        session_id: The incident this state belongs to.
        alert: The payload that raised it.
        target_stage: How far the pipeline should run.
        incident_dir: The writable workspace for this run.
        shared_dir: The workspace shared across runs.
        withheld_record_ids: Corpus records this incident may not retrieve, over and
            above the process-wide floor.

    Returns:
        The state the first role is entered with.
    """
    return IncidentState(
        session_id=session_id,
        alert=alert,
        target_stage=target_stage,
        withheld_record_ids=sorted(set(withheld_record_ids)),
        incident_dir=str(incident_dir),
        shared_dir=str(shared_dir),
        messages={},
        trace_summaries=[],
        run_ids=[],
        root_cause=None,
        plan=None,
        execution_result=None,
        verify_result=None,
        remediation_retries=0,
        reassessment_count=0,
        verify_retries=0,
        prior_attempts=[],
        prior_plans=[],
        error=None,
    )
