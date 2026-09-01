"""Waiting for a stage, handing one answer over, and reducing the grades that come back.

The stage machine is the awkward part of this integration and the reason each of these
is its own function. The conductor advances a stage only after grading the previous one,
grading a diagnosis is an asynchronous judge that takes tens of seconds, and a
submission for any stage other than the live one is rejected. Firing both answers back
to back therefore leaves the second silently ungraded, which looks in the results
exactly like an agent that never answered.
"""

import asyncio
import logging
from typing import Any

import httpx

from kira.adapters.base import ExternalFeedback, SubmissionRecord
from kira.adapters.sregym.conductor import Conductor

__all__ = ["ORACLE_SOURCE", "reduce_feedback", "submit_stage", "wait_for_stage"]

_log = logging.getLogger(__name__)

ORACLE_SOURCE = "sregym_oracle"
"""What the distilling pass is told the grades came from."""

_TERMINAL_STAGES = frozenset({"tearing_down", "done"})

_ERROR_BODY_CHARS = 200


async def wait_for_stage(
    conductor: Conductor, target: str, *, timeout_s: float = 600.0, poll_s: float = 5.0
) -> bool:
    """Block until the grader will accept a submission for one stage.

    Args:
        conductor: The episode's conductor.
        target: The stage being waited for.
        poll_s: How often to ask.
        timeout_s: How long to keep asking.

    Returns:
        True when the stage came live. False when the episode has already moved past
        grading, or when the wait ran out.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    stage: str | None = None
    while loop.time() < deadline:
        stage = await conductor.stage()
        if stage == target:
            return True
        if stage in _TERMINAL_STAGES:
            _log.warning("conductor is at terminal stage %r while waiting for %r", stage, target)
            return False
        await asyncio.sleep(poll_s)
    _log.warning(
        "timed out after %.0fs waiting for stage %r, last seen %r", timeout_s, target, stage
    )
    return False


async def submit_stage(conductor: Conductor, solution: str, stage: str) -> SubmissionRecord:
    """Hand one answer over, recording a failure rather than raising it.

    A submission that could not land must not end the episode: the environment is still
    held and the later stages, including the release, still have to run.

    Args:
        conductor: The episode's conductor.
        solution: The text to grade, empty where the grade comes from the cluster.
        stage: Which stage this answers.

    Returns:
        What happened.
    """
    try:
        response = await conductor.submit(solution, stage)
    except httpx.HTTPStatusError as exc:
        body = exc.response.text[:_ERROR_BODY_CHARS]
        return SubmissionRecord(
            kind=stage,
            payload=solution,
            error=f"HTTP {exc.response.status_code}: {body}",
        )
    except Exception as exc:
        return SubmissionRecord(kind=stage, payload=solution, error=f"{type(exc).__name__}: {exc}")
    return SubmissionRecord(kind=stage, payload=solution, response=response)


def reduce_feedback(results: Any, *, details: bool) -> ExternalFeedback | None:
    """Reduce the oracle's verdicts to the compact record the distilling pass reads.

    The judge's per-dimension scores and its checklist are dropped, because what the
    lesson needs is whether the answer was right and how right, and the rest only
    lengthens the prompt.

    Args:
        results: The conductor's results body.
        details: Whether this arm also sees the judge's reasoning and the reference
            root cause.

    Returns:
        The grades, or None when nothing gradable came back.
    """
    if not isinstance(results, dict):
        return None
    diagnosis = results.get("Diagnosis") or {}
    mitigation = results.get("Mitigation") or {}
    graded: dict[str, Any] = {
        "success": diagnosis.get("success"),
        "score": diagnosis.get("accuracy"),
    }
    if details:
        graded["reasoning"] = diagnosis.get("reasoning")
    return ExternalFeedback(
        source=ORACLE_SOURCE,
        diagnosis=graded,
        mitigation={"success": mitigation.get("success")},
        ground_truth_root_cause=results.get("ground_truth_root_cause") if details else None,
    )
