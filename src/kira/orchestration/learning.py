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
from typing import Any

from kira.audit.journal import Journal
from kira.contracts.artifacts import LearningPacket, LearningResult, RoleTraceSummary
from kira.memory.distil import distil
from kira.middleware.base import MiddlewareStack
from kira.middleware.write_gate import WriteGate
from kira.observability.usage import RecordingCompletion
from kira.orchestration.jobs import bound_deployment
from kira.orchestration.runner import Deployment, incident_context, tool_context_for
from kira.roles.base import RoleContext
from kira.roles.registry import get_role
from kira.tools.toolsets import build_toolset

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
        alert=record.alert,
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

    spec = get_role("learning")
    async with incident_context(session_id, dict(packet.alert), deployment) as resources:
        journal = Journal(deployment.sink, session_id=session_id, stage=spec.name)
        tools = tool_context_for(deployment, resources, spec.name, session_id, journal)
        outcome = await spec.run(
            RoleContext(
                spec=spec,
                state={"session_id": session_id, "alert": dict(packet.alert)},
                handback=None,
                settings=deployment.settings,
                # Wrapped, like every other stage. Handed the raw client, the distilling
                # pass ran off the books entirely: no model-call event, no tokens, no
                # cost, in the arm whose whole claim is that its memory is worth what it
                # costs to build. The rule is that a model call anywhere is counted.
                completion=RecordingCompletion(
                    deployment.completion, journal, deployment.profile, deployment.sampling
                ),
                profile=deployment.profile,
                middleware=MiddlewareStack(
                    [
                        WriteGate(
                            authority=deployment.authority,
                            settings=deployment.settings.authority,
                            journal=journal,
                        ),
                        *deployment.hooks,
                    ]
                ),
                write_lock=deployment.write_lock,
                toolset=build_toolset(spec.name, tools, extras=deployment.tool_extras),
                tool_context=tools,
                journal=journal,
                run_id=f"{session_id}:learning:1",
                session_id=session_id,
                memory=tools.memory,
                packet=packet,
            )
        )

    if not isinstance(outcome.artifact, LearningResult):
        _log.warning("learning produced nothing for %s: %s", session_id, outcome.error)
        return "no_submission"

    distil(
        packet,
        outcome.artifact,
        index=deployment.storage.index,
        sessions=deployment.storage.sessions,
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
