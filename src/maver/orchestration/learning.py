"""Distilling a closed incident, minutes after it closed.

This is the offline half of the memory loop, and it is offline for two reasons that are
easy to conflate. It runs outside the incident path because that is what the published
description of the system says, and it runs *late* because a remediation's outcome
arrives seconds to minutes after the action: a distillation taken the instant the
pipeline stopped would record a verdict that had not settled.

It reads what was written down rather than what a run held in memory. The job that
starts it has nothing but a session identifier, which is the right constraint — a lesson
that could only be distilled from live state would be lost to any restart.
"""

import logging
from dataclasses import replace
from typing import Any

from maver.contracts.artifacts import LearningPacket, LearningResult, RoleTraceSummary
from maver.contracts.runtime import RunState
from maver.memory.distil import distil
from maver.orchestration.jobs import bound_deployment
from maver.orchestration.runner import (
    Deployment,
    _context_builder,
    incident_context,
)
from maver.orchestration.runtime import load_runtime, save_runtime
from maver.pipelines import PipelineChangedError, restore_selection
from maver.roles.prompting import visible_alert
from maver.roles.registry import get_role

__all__ = ["build_packet", "distil_incident", "run_learning"]

_log = logging.getLogger(__name__)


def build_packet(
    deployment: Deployment, session_id: str, *, external_feedback: dict[str, Any] | None = None
) -> LearningPacket | None:
    """Assemble what the offline pass is given about one closed incident.

    Args:
        deployment: This process's resources.
        session_id: The incident to distil.
        external_feedback: An outside evaluation of the session, when one exists.

    Returns:
        The packet, or None when the session is gone or has already been distilled.
    """
    record = deployment.storage.sessions.get_session(session_id)
    if record is None:
        _log.warning("no session %s to distil", session_id)
        return None
    if record.distilled:
        # Re-distilling would replace the record rather than duplicate it, so this is a
        # cost guard rather than a correctness one: a retried job should not spend a
        # model call re-deriving a lesson that is already stored.
        return None

    summaries: list[RoleTraceSummary] = []
    for run in deployment.storage.runs.runs_for(session_id):
        if not run.trace_summary:
            continue
        try:
            summaries.append(RoleTraceSummary.model_validate(run.trace_summary))
        except Exception:
            _log.warning("ignoring an unreadable trace summary on run %s", run.agent_run_id)

    return LearningPacket(
        session_id=session_id,
        lifecycle_state=record.state,
        # Filtered here rather than at each reader. The packet is rendered whole into
        # this pass's prompt, so the underscore-prefixed keys the other roles' prompt
        # builders drop — for an evaluation harness, the ground-truth fault label —
        # reached the one role whose output is the corpus record later incidents
        # retrieve. Doing it at assembly covers every reader of the packet, including
        # the corpus write, which renders the alert into a searchable projection.
        alert=visible_alert(record.alert),
        root_cause=record.root_cause,
        plan=record.plan,
        execution_result=record.execution_result,
        verify_result=record.verify_result,
        trace_summaries=summaries,
        operator_notes=record.notes,
        external_feedback=external_feedback,
    )


async def run_learning(
    session_id: str, deployment: Deployment, *, external_feedback: dict[str, Any] | None = None
) -> str:
    """Distil one closed incident into the corpus.

    Args:
        session_id: The incident to distil.
        deployment: This process's resources.
        external_feedback: An outside evaluation of the session, when one exists.

    Returns:
        A short status for the job log.
    """
    if not deployment.settings.memory.learning_enabled:
        return "disabled"
    packet = build_packet(deployment, session_id, external_feedback=external_feedback)
    if packet is None:
        return "nothing_to_distil"

    runtime = load_runtime(deployment.storage.sessions, session_id) or RunState()
    if runtime.deployment_digest and runtime.deployment_digest != deployment.runtime_digest:
        raise PipelineChangedError(
            "deployment behavior changed since admission; restore its original configuration before learning"
        )
    definition = None
    if runtime.selection:
        assert deployment.pipelines is not None
        selected = restore_selection(deployment.pipelines, runtime.selection)
        definition = selected.pipeline.learning_agent
        if definition is None:
            return "disabled"
    # Sessions predating configured pipelines retain their original role selection.
    spec = definition.spec if definition is not None else get_role("learning")
    attempt = 1 + sum(agent.role == "learning" for agent in runtime.agents.values())
    run_id = f"{session_id}:learning:{attempt}"
    while run_id in runtime.agents:
        attempt += 1
        run_id = f"{session_id}:learning:{attempt}"
    previous_stage, previous_agent = runtime.active_stage, runtime.active_agent
    runtime.active_stage = None
    try:
        async with incident_context(session_id, dict(packet.alert), deployment) as resources:
            context = _context_builder(
                deployment, resources, retrieval=True, definition_override=definition
            )(
                spec,
                {"session_id": session_id, "alert": dict(packet.alert), "runtime": runtime},
                None,
                run_id,
            )
            outcome = await spec.run(replace(context, packet=packet))
            if context.agent_state is not None:
                context.agent_state.messages = list(outcome.transcript)
    finally:
        # Learning is an asynchronous invocation, not a foreground workflow step.
        # Its private state persists without advancing or replacing that workflow.
        runtime.active_stage, runtime.active_agent = previous_stage, previous_agent
        save_runtime(deployment.storage.sessions, session_id, runtime)

    if not isinstance(outcome.artifact, LearningResult):
        _log.warning("learning produced nothing for %s: %s", session_id, outcome.error)
        return "no_submission"

    distil(
        packet,
        outcome.artifact,
        index=deployment.storage.index,
        sessions=deployment.storage.sessions,
        settings=deployment.settings.memory,
    )
    return "distilled"


async def distil_incident(session_id: str, **kwargs: Any) -> str:
    """The scheduled form of :func:`run_learning`.

    Separate because a scheduler restores a job by dotted import path and hands it only
    its keyword arguments: there is nowhere for a deployment to come from except the one
    the process bound at startup.

    Args:
        session_id: The incident to distil.
        **kwargs: Passed through. ``external_feedback`` is the one that means something;
            anything else is tolerated so a job stored by an older build still runs.

    Returns:
        A short status for the job log.
    """
    feedback = kwargs.get("external_feedback")
    return await run_learning(
        session_id,
        bound_deployment(),
        external_feedback=feedback if isinstance(feedback, dict) else None,
    )
