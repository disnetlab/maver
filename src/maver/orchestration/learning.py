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

import asyncio
import logging
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from maver.audit.context import bind_journal
from maver.audit.journal import Journal
from maver.contracts.artifacts import LearningPacket, LearningResult, RoleTraceSummary
from maver.contracts.events import StageEvent
from maver.contracts.runtime import RunState
from maver.memory.distil import distil
from maver.memory.ports import AgentRunRecord, LearningIntent, SessionStore
from maver.orchestration.jobs import bound_deployment
from maver.orchestration.nodes import capture_run, context_metadata
from maver.orchestration.outcome_feedback import latest_feedback_packet
from maver.orchestration.runner import (
    Deployment,
    _context_builder,
    incident_context,
)
from maver.orchestration.runtime import load_runtime, save_runtime
from maver.orchestration.scheduler import scheduler_for
from maver.pipelines import PipelineChangedError, restore_selection
from maver.roles.base import RoleContext, RoleOutcome
from maver.roles.prompting import visible_alert
from maver.roles.registry import get_role
from maver.roles.trace_summary import build_trace_summary

__all__ = [
    "active_learning_tasks",
    "build_packet",
    "distil_incident",
    "restore_learning_intents",
    "run_learning",
    "run_learning_intent",
    "schedule_learning_intent",
]

_log = logging.getLogger(__name__)
_active_tasks: set[asyncio.Task[Any]] = set()
_learning_capacity = asyncio.Semaphore(1)


def active_learning_tasks() -> set[asyncio.Task[Any]]:
    """Snapshot jobs that may still use storage during process shutdown."""
    return set(_active_tasks)


def schedule_learning_intent(workdir: Path, intent: LearningIntent | None) -> None:
    """Queue one persisted intent; a later sweep repairs a scheduler write failure."""
    if intent is None:
        return
    due_at = datetime.fromisoformat(intent.due_at)
    scheduler_for(workdir).add_job(
        run_learning_intent,
        "date",
        run_date=max(due_at, datetime.now(UTC)),
        kwargs={"session_id": intent.session_id},
        id=f"learning:{intent.session_id}",
        replace_existing=True,
    )


def restore_learning_intents(sessions: SessionStore, workdir: Path) -> None:
    """Restore explicit eligibility, then sweep periodically for lost jobs and retries."""
    global _learning_capacity
    _learning_capacity = asyncio.Semaphore(1)
    scheduler = scheduler_for(workdir)
    scheduler.add_job(
        sweep_learning_intents,
        "interval",
        seconds=60,
        id="learning:intent_sweep",
        replace_existing=True,
    )
    # Legacy one-shot jobs carry their own proof of eligibility and benchmark grade.
    # Adopt them before replacing their callable with the retryable wrapper.
    for job in scheduler.get_jobs():
        if not job.id.startswith("learning:") or job.id == "learning:intent_sweep":
            continue
        if job.func is run_learning_intent:
            continue
        if job.func is not distil_incident:
            continue
        session_id = job.kwargs.get("session_id")
        if not isinstance(session_id, str):
            continue
        feedback = job.kwargs.get("external_feedback")
        sessions.put_learning_intent(
            LearningIntent(
                session_id=session_id,
                due_at=(job.next_run_time or datetime.now(UTC)).isoformat(),
                external_feedback=feedback if isinstance(feedback, dict) else None,
            )
        )
    for intent in sessions.pending_learning_intents():
        try:
            schedule_learning_intent(workdir, intent)
        except Exception:
            _log.exception("could not restore Learning intent for %s", intent.session_id)


async def sweep_learning_intents() -> None:
    """Repair jobs lost after durable intent creation without scanning terminal sessions."""
    task = asyncio.current_task()
    assert task is not None
    _active_tasks.add(task)
    try:
        deployment = bound_deployment()
        scheduler = scheduler_for(deployment.settings.server.workdir)
        for intent in deployment.storage.sessions.pending_learning_intents():
            if scheduler.get_job(f"learning:{intent.session_id}") is None:
                try:
                    schedule_learning_intent(deployment.settings.server.workdir, intent)
                except Exception:
                    _log.exception("could not restore Learning intent for %s", intent.session_id)
    finally:
        _active_tasks.discard(task)


async def run_learning_intent(session_id: str) -> str:
    """Run a due intent and retain it until a successful corpus write."""
    task = asyncio.current_task()
    assert task is not None
    _active_tasks.add(task)
    try:
        async with _learning_capacity:
            deployment = bound_deployment()
            sessions = deployment.storage.sessions
            intent = sessions.get_learning_intent(session_id)
            if intent is None:
                return "no_intent"
            due_at = datetime.fromisoformat(intent.due_at)
            if due_at > datetime.now(UTC):
                schedule_learning_intent(deployment.settings.server.workdir, intent)
                return "not_due"
            record = sessions.get_session(session_id)
            if record is not None and record.distilled:
                sessions.complete_learning_intent(session_id)
                return "already_distilled"
            try:
                status = await run_learning(
                    session_id, deployment, external_feedback=intent.external_feedback
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                _log.exception("Learning failed for %s; retaining intent", session_id)
                status = "exception"
            if status in {"distilled", "disabled", "nothing_to_distil"}:
                sessions.complete_learning_intent(session_id)
            else:
                # One minute after the first failure, at most fifteen minutes thereafter.
                delay = min(900, 60 * 2 ** min(intent.attempts, 4))
                next_due = datetime.now(UTC) + timedelta(seconds=delay)
                sessions.advance_learning_intent(session_id, next_due.isoformat())
                try:
                    schedule_learning_intent(
                        deployment.settings.server.workdir,
                        sessions.get_learning_intent(session_id),
                    )
                except Exception:
                    _log.exception("could not schedule Learning retry for %s", session_id)
            return status
    finally:
        _active_tasks.discard(task)


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

    # A human verdict may have arrived after the delayed job was scheduled. It is
    # authoritative for the deployable incident; benchmark feedback remains the source
    # when there is no operator review.
    external_feedback = (
        latest_feedback_packet(deployment.storage.sessions, session_id) or external_feedback
    )

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
    runs = deployment.storage.runs
    attempt = 1 + sum(agent.role == "learning" for agent in runtime.agents.values())
    run_id = f"{session_id}:learning:{attempt}"
    while run_id in runtime.agents or runs.get_run(run_id) is not None:
        attempt += 1
        run_id = f"{session_id}:learning:{attempt}"
    runs.start_run(
        AgentRunRecord(
            agent_run_id=run_id,
            pipeline_id=session_id,
            agent_name="learning",
            attempt=attempt,
            status="running",
            started_at=datetime.now(UTC).isoformat(),
            metadata={"asynchronous": True},
        )
    )
    journal = Journal(
        deployment.sink,
        session_id=session_id,
        stage="learning",
        run_id=run_id,
        sequence=deployment.sequences.for_session(session_id),
    )
    journal.record(StageEvent(source="learning", stage="learning", event="entered"))
    previous_stage, previous_agent = runtime.active_stage, runtime.active_agent
    runtime.active_stage = None
    context: RoleContext | None = None
    outcome: RoleOutcome | None = None
    status = "no_submission"
    try:
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
                with bind_journal(context.journal):
                    outcome = await spec.run(replace(context, packet=packet))
                if context.agent_state is not None:
                    context.agent_state.messages = list(outcome.transcript)
            if isinstance(outcome.artifact, LearningResult):
                # An operator may have reviewed the run while the model was distilling.
                # A result based on an older verdict must not enter the corpus.
                current_feedback = latest_feedback_packet(deployment.storage.sessions, session_id)
                if current_feedback is not None and current_feedback != packet.external_feedback:
                    _log.warning("operator feedback changed during learning for %s", session_id)
                    status = "feedback_changed"
                    continue_learning = False
                else:
                    continue_learning = True
                if continue_learning:
                    latest_state = deployment.storage.sessions.get_states([session_id]).get(
                        session_id
                    )
                    if latest_state is not None:
                        packet = packet.model_copy(update={"lifecycle_state": latest_state})
                    distil(
                        packet,
                        outcome.artifact,
                        index=deployment.storage.index,
                        sessions=deployment.storage.sessions,
                        settings=deployment.settings.memory,
                    )
                    status = "distilled"
            else:
                _log.warning("learning produced nothing for %s: %s", session_id, outcome.error)
        finally:
            # Learning has its own conversation; it never advances foreground steps.
            runtime.active_stage, runtime.active_agent = previous_stage, previous_agent
            save_runtime(deployment.storage.sessions, session_id, runtime)
    except BaseException as exc:
        cancelled = isinstance(exc, asyncio.CancelledError)
        error = "cancelled" if cancelled else repr(exc)
        metadata = context_metadata(context, journal)
        output = (
            outcome.artifact.model_dump(mode="json")
            if outcome is not None and outcome.artifact is not None
            else None
        )
        journal.record(
            StageEvent(
                source="learning",
                stage="learning",
                event="cancelled" if cancelled else "failed",
                detail=error,
                run=capture_run(
                    journal,
                    run_id,
                    role="learning",
                    status="cancelled" if cancelled else "failed",
                    metadata=metadata,
                    output=output,
                    error=error,
                ),
            )
        )
        runs.finish_run(
            run_id,
            status="cancelled" if cancelled else "failed",
            output=output,
            error=error,
            metadata=metadata,
        )
        raise

    assert outcome is not None
    output = outcome.artifact.model_dump(mode="json") if outcome.artifact is not None else None
    summary = build_trace_summary(
        "learning", transcript=outcome.transcript, output=outcome.artifact, error=outcome.error
    ).model_dump(mode="json")
    metadata = {
        **outcome.metadata,
        **context_metadata(context, journal, transcript=outcome.transcript),
        "distilled": status == "distilled",
    }
    evidence = capture_run(
        journal,
        run_id,
        role="learning",
        status="succeeded" if status == "distilled" else "failed",
        metadata=metadata,
        output=output,
        trace_summary=summary,
        error=outcome.error,
    )
    journal.record(
        StageEvent(
            source="learning",
            stage="learning",
            event="exited" if status == "distilled" else "failed",
            detail=status,
            run=evidence,
        )
    )
    try:
        runs.finish_run(
            run_id,
            status="succeeded" if status == "distilled" else "failed",
            output=output,
            trace_summary=summary,
            error=outcome.error,
            metadata=metadata,
        )
    except Exception as exc:
        journal.record(
            StageEvent(
                source="learning",
                stage="persistence",
                event="failed",
                detail=f"{type(exc).__name__}: {exc}",
                run=evidence,
            )
        )
        raise

    return status


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
    deployment = bound_deployment()
    feedback = kwargs.get("external_feedback")
    external_feedback = feedback if isinstance(feedback, dict) else None
    sessions = deployment.storage.sessions
    try:
        if sessions.get_learning_intent(session_id) is None:
            sessions.put_learning_intent(
                LearningIntent(
                    session_id=session_id,
                    due_at=datetime.now(UTC).isoformat(),
                    external_feedback=external_feedback,
                )
            )
    except Exception:
        # A legacy job is the only remaining proof of eligibility if the operational
        # store is temporarily unavailable. Attempt it once as the old release did.
        _log.exception("could not adopt legacy Learning job for %s", session_id)
        task = asyncio.current_task()
        assert task is not None
        _active_tasks.add(task)
        try:
            async with _learning_capacity:
                return await run_learning(
                    session_id, deployment, external_feedback=external_feedback
                )
        finally:
            _active_tasks.discard(task)
    return await run_learning_intent(session_id)
