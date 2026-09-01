"""Incident intake, and the background task that carries one run.

The request and response shapes here are held byte-compatible with the prototype's,
because an evaluation harness outside this repository posts to them and polls them
unchanged. Where the prototype's vocabulary and this system's disagree — it called the
diagnosis-only run ``rca`` — the wire keeps the old word and the pipeline gets the new
one, rather than either side being asked to move.

Accepting an incident returns immediately. The run itself is minutes to hours of model
calls, so the caller is given a session id and polls it.
"""

import asyncio
import logging
from collections.abc import Mapping
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from kira.api.deps import Service, ServiceDep
from kira.contracts.lifecycle import LifecycleState, RunStatus
from kira.contracts.state import TargetStage
from kira.middleware.authority import ApprovalPending
from kira.orchestration.runner import Deployment, run_incident

__all__ = ["IncidentRequest", "router", "run_incident_task", "start_incident"]

_log = logging.getLogger(__name__)

router = APIRouter()

_TARGET_STAGES: Mapping[str, TargetStage] = {
    "full": "full",
    "diagnose": "diagnose",
    # The held wire value. The prototype named the diagnosis-only run after the role
    # that produced it; the pipeline now names it after what it delivers.
    "rca": "diagnose",
}

_UNFINISHED = ("running", "awaiting_human")


class IncidentRequest(BaseModel):
    """An alert, and how far to take it."""

    alert: dict[str, Any]
    target_stage: Literal["full", "rca", "diagnose"] = "full"


@router.post("/incidents", status_code=202)
async def create_incident(body: IncidentRequest, service: ServiceDep) -> dict[str, Any]:
    """Accept an alert and start working on it.

    Args:
        body: The alert and the stage to run to.
        service: The process's resources.

    Returns:
        The session to poll, echoing the stage word the caller used.

    Raises:
        HTTPException: 503 when HTTP intake is switched off, which is how a deployment
            that takes alerts only from its queue closes this door without losing the
            read routes.
    """
    if not service.deployment.settings.intake.http_enabled:
        raise HTTPException(
            status_code=503, detail="HTTP intake disabled (KIRA_INTAKE__HTTP_ENABLED=false)"
        )
    return start_incident(body.alert, target_stage=body.target_stage, service=service)


@router.get("/incidents/{session_id}")
async def read_incident(session_id: str, service: ServiceDep) -> dict[str, Any]:
    """Report where one incident got to.

    Args:
        session_id: The incident.
        service: The process's resources.

    Returns:
        The session record. A poller reads ``run_status`` and nothing else has to move
        for it to work.

    Raises:
        HTTPException: 404 when there is no such incident.
    """
    record = service.deployment.storage.sessions.get_session(session_id)
    if record is None:
        raise HTTPException(status_code=404, detail="incident not found")
    return {
        "session_id": record.session_id,
        "status": record.state,
        "run_status": record.run_status,
        "current_stage": record.current_stage,
        "awaiting_request_id": record.awaiting_request_id,
        "started_at": record.started_at,
        "ended_at": record.ended_at,
        "alert_name": record.alert.get("alertname", "-"),
        "alert": record.alert,
        "root_cause": record.root_cause,
        "plan": record.plan,
        "execution_result": record.execution_result,
        "verify_result": record.verify_result,
        "distilled": record.distilled,
        "notes": record.notes,
        "error": record.notes if record.state == LifecycleState.INVALIDATED.value else None,
    }


def start_incident(alert: dict[str, Any], *, target_stage: str, service: Service) -> dict[str, Any]:
    """Open a session, put a task on it, and hand back what to poll.

    Args:
        alert: The payload that raised the incident.
        target_stage: The requested stage, in the wire's vocabulary.
        service: The process's resources.

    Returns:
        The intake receipt: session id, acceptance, the stage as asked for, and when the
        session was opened.
    """
    sessions = service.deployment.storage.sessions
    record = sessions.create_session(alert)
    # A created session and a session being worked on are different things, and the
    # field a poller keys on has to tell them apart from the moment the task exists.
    sessions.set_run_status(record.session_id, RunStatus.RUNNING)

    # Registered before the task, so a run that finishes inside one loop iteration
    # cannot be unregistered before it was ever registered.
    service.runs.register(record.session_id)
    task = asyncio.create_task(
        run_incident_task(
            session_id=record.session_id,
            alert=alert,
            target_stage=_TARGET_STAGES[target_stage],
            service=service,
        )
    )
    service.tasks.add(task)
    task.add_done_callback(service.tasks.discard)
    return {
        "session_id": record.session_id,
        "status": "accepted",
        "target_stage": target_stage,
        "started_at": record.started_at,
    }


async def run_incident_task(
    *, session_id: str, alert: dict[str, Any], target_stage: TargetStage, service: Service
) -> None:
    """Run one incident, and leave the session readable whatever happens to it.

    Args:
        session_id: The session opened for this alert.
        alert: The payload that raised it.
        target_stage: How far to run.
        service: The process's resources.
    """
    try:
        await run_incident(alert, session_id, service.deployment, target_stage=target_stage)
    except ApprovalPending as pending:
        # Not a failure. The incident is parked on a person, with everything needed to
        # continue it already written down.
        _log.info("session %s is awaiting approval request %s", session_id, pending.request_id)
    except asyncio.CancelledError:
        # Re-raised rather than absorbed: cooperative cancellation is what a drain and
        # an operator's stop both depend on.
        _log.info("session %s was cancelled", session_id)
        raise
    except Exception as exc:
        _log.exception("session %s crashed", session_id)
        _record_failure(service.deployment, session_id, exc)
    finally:
        service.runs.unregister(session_id)


def _record_failure(deployment: Deployment, session_id: str, exc: BaseException) -> None:
    """Leave a crashed incident in a state a reader can tell is over."""
    detail = f"{type(exc).__name__}: {exc}"
    try:
        sessions = deployment.storage.sessions
        sessions.update_artefacts(session_id, notes=detail)
        sessions.set_state(session_id, LifecycleState.INVALIDATED)
        # Polling adapters key on run_status. Leaving it as RUNNING after the task has
        # unwound strands the benchmark environment until its outer timeout even though
        # the failure is already terminal.
        sessions.set_run_status(session_id, RunStatus.FAILED)
        for run in reversed(deployment.storage.runs.runs_for(session_id)):
            if run.status in _UNFINISHED:
                deployment.storage.runs.finish_run(run.agent_run_id, status="failed", error=detail)
                break
    except Exception:
        # The prototype swallowed this silently. A failure to record a failure is
        # exactly the case an operator needs told about.
        _log.exception("could not record the failure of session %s", session_id)
