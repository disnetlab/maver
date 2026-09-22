"""Incident intake, and the background task that carries one run.

The request and response shapes here are held byte-compatible with the prototype's,
because an evaluation harness outside this repository posts to them and polls them
unchanged. Where the prototype's vocabulary and this system's disagree — it called the
diagnosis-only run ``rca`` — the wire keeps the old word and the pipeline gets the new
one, rather than either side being asked to move.

Accepting an incident returns immediately. The run itself is minutes to hours of model
calls, so the caller is given a session id and polls it.

Which is why the store calls on these routes go to a thread. The run is a task on this
same event loop, created here, so a read made inline by the poller is not a cost the API
pays on its own — it stops the very incident being polled, and a harness polls one every
few seconds for hours. The exception is :func:`_record_failure`, which runs once as a
crashed run unwinds and is discussed there.
"""

import asyncio
import hashlib
import json
import logging
from collections.abc import Mapping
from typing import Any, Literal

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from maver.api.deps import Service, ServiceDep, admit
from maver.contracts.lifecycle import LifecycleState, RunStatus
from maver.contracts.runtime import RunState
from maver.contracts.state import TargetStage
from maver.middleware.authority import ApprovalPending
from maver.orchestration.jobs import record_cancellation
from maver.orchestration.runner import Deployment, run_incident
from maver.orchestration.runtime import load_runtime, save_runtime
from maver.pipelines import SelectedPipeline, select_pipeline
from maver.roles.prompting import visible_alert

__all__ = ["IncidentRequest", "fingerprint", "router", "run_incident_task", "start_incident"]

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
    pipeline: str | None = Field(default=None, min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
    """Private routing and middleware context, never merged into the model-visible alert."""


@router.post("/incidents", status_code=202)
async def create_incident(
    body: IncidentRequest,
    service: ServiceDep,
    idempotency_key: str | None = Header(default=None),
) -> dict[str, Any]:
    """Accept an alert and start working on it.

    Args:
        body: The alert and the stage to run to.
        service: The process's resources.
        idempotency_key: Optional durable retry key.

    Returns:
        The session to poll, echoing the stage word the caller used.

    Raises:
        HTTPException: 503 when HTTP intake is switched off, which is how a deployment
            that takes alerts only from its queue closes this door without losing the
            read routes; and whatever :func:`~maver.api.deps.admit` refuses with.
    """
    if not service.deployment.settings.intake.http_enabled:
        raise HTTPException(
            status_code=503, detail="HTTP intake disabled (MAVER_INTAKE__HTTP_ENABLED=false)"
        )

    if idempotency_key is not None and (not idempotency_key or len(idempotency_key) > 255):
        raise HTTPException(status_code=422, detail="Idempotency-Key must contain 1-255 characters")
    return await accept_incident(
        body, service, intake_id=("http", idempotency_key) if idempotency_key else None
    )


async def accept_incident(
    body: IncidentRequest,
    service: Service,
    *,
    intake_id: tuple[str, str] | None = None,
) -> dict[str, Any]:
    """Admit an already validated alert from HTTP or a queue."""
    sessions = service.deployment.storage.sessions
    request_hash = _request_hash(body)
    # An exact delivery replay owns its original session even when that session is
    # parked for reconciliation or its selected pipeline is no longer configured.
    if intake_id is not None:
        try:
            prior = await asyncio.to_thread(sessions.get_intake_receipt, *intake_id, request_hash)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if prior is not None:
            return _receipt(body, service.deployment, prior)
    unresolved = await asyncio.to_thread(sessions.reconciliation_sessions)
    key = fingerprint(body.alert)
    for parked in unresolved:
        if fingerprint(parked.alert) == key:
            raise HTTPException(
                status_code=409,
                detail=f"alert is held by incident {parked.session_id} pending reconciliation",
                headers={"Retry-After": "30"},
            )
    # Validate explicit names before reserving capacity or opening a session. Resolving
    # selectors is deferred until this request owns the alert, so retries cannot run
    # selection middleware repeatedly for one incident.
    explicit = _requested_pipeline(body, service.deployment)
    claim = admit(service, fingerprint=fingerprint(body.alert))

    # A retrying alert source posting the same alert twice wants one incident, not two
    # agents remediating the same fault against each other. Answered with the run that
    # is already on it, in the same shape, so the caller needs no new branch.
    if not claim.ours:
        existing = await claim.session()
        if existing is None:
            raise HTTPException(
                status_code=503,
                detail="the incident this alert raised could not be opened",
                headers={"Retry-After": "30"},
            )
        record = await asyncio.to_thread(sessions.get_session, existing)
        runtime = await asyncio.to_thread(
            load_runtime, service.deployment.storage.sessions, existing
        )
        selected = runtime.selection.get("name") if runtime is not None else None
        if explicit is not None and selected != explicit:
            raise HTTPException(
                status_code=409,
                detail=f"this alert is already handled by pipeline {selected!r} in incident {existing}",
            )
        if intake_id is not None:
            try:
                record = await asyncio.to_thread(
                    sessions.bind_intake_receipt, *intake_id, request_hash, existing
                )
            except ValueError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
        _log.info("alert already being handled by %s; not starting a second run", existing)
        receipt = {
            "session_id": existing,
            "status": "accepted",
            "target_stage": body.target_stage,
            "started_at": record.started_at if record is not None else "",
        }
        if _show_pipeline(body, service.deployment):
            receipt["pipeline"] = str(selected or "")
        return receipt

    try:
        selection = await _select(body, service.deployment, explicit)
        admission = asyncio.create_task(
            start_incident(
                body.alert,
                target_stage=body.target_stage,
                service=service,
                selection=selection,
                metadata=body.metadata,
                intake_id=intake_id,
                request_hash=request_hash,
            ),
            name="maver-incident-admission",
        )
        # A cancelled HTTP request must not cancel a thread after it commits the
        # session but before this coroutine registers its run. Keep the claim until
        # initialization either finishes or records a failure.
        cancelled = False
        while not admission.done():
            try:
                await asyncio.shield(admission)
            except asyncio.CancelledError:
                cancelled = True
            except Exception:
                break
        if cancelled:
            if not admission.cancelled():
                error = admission.exception()
                if error is not None:
                    _log.error(
                        "incident admission failed after request cancellation", exc_info=error
                    )
            raise asyncio.CancelledError
        receipt = admission.result()
        if _show_pipeline(body, service.deployment):
            receipt["pipeline"] = selection.pipeline.name
        return receipt
    finally:
        service.runs.release(claim)


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
    record = await asyncio.to_thread(service.deployment.storage.sessions.get_session, session_id)
    if record is None:
        raise HTTPException(status_code=404, detail="incident not found")
    result = {
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
    if record.runtime.get("selection"):
        result["pipeline"] = record.runtime["selection"].get("name")
    return result


@router.post("/incidents/{session_id}/cancel", status_code=202)
async def cancel_incident(session_id: str, service: ServiceDep) -> dict[str, Any]:
    """Stop a run that is going wrong, without killing the process.

    An operator watching a run spiral had no way to stop it short of deleting the pod,
    which skips every ``finally`` the run is holding: the terminal record, and an
    environment adapter's release of the application it is grading.

    Args:
        session_id: The incident to stop.
        service: The process's resources.

    Returns:
        The incident, and that it is being stopped. Answered before the run has unwound,
        like the intake it mirrors: a poller reads ``run_status`` for the end.

    Raises:
        HTTPException: 404 when there is no such incident, and 409 when this process is
            not carrying it — it has already finished, or it is parked on a person,
            which is ended by deciding the approval rather than here.
    """
    record = await asyncio.to_thread(service.deployment.storage.sessions.get_session, session_id)
    if record is None:
        raise HTTPException(status_code=404, detail="incident not found")
    if not service.runs.cancel(session_id):
        raise HTTPException(
            status_code=409, detail=f"incident is {record.run_status}, not running here"
        )
    _log.info("session %s was cancelled by an operator", session_id)
    return {"session_id": session_id, "status": "cancelling"}


async def start_incident(
    alert: dict[str, Any],
    *,
    target_stage: str,
    service: Service,
    selection: SelectedPipeline | None = None,
    metadata: dict[str, Any] | None = None,
    intake_id: tuple[str, str] | None = None,
    request_hash: str | None = None,
) -> dict[str, Any]:
    """Open a session, put a task on it, and hand back what to poll.

    A coroutine because it always was one in effect: it creates the incident's task, and
    :func:`asyncio.create_task` has never been callable without a running loop. Saying so
    is what lets the two writes below go to a thread.

    Args:
        alert: The payload that raised the incident.
        target_stage: The requested stage, in the wire's vocabulary.
        service: The process's resources.
        selection: The already resolved admission choice, if the caller selected it.
        metadata: Private middleware context, separate from the alert.
        intake_id: Durable source and message key, when supplied.
        request_hash: Digest of the complete validated request.

    Returns:
        The intake receipt: session id, acceptance, the stage as asked for, and when the
        session was opened.
    """
    if selection is None:
        body = IncidentRequest.model_validate(
            {"alert": alert, "target_stage": target_stage, "metadata": metadata or {}}
        )
        selection = await _select(
            body, service.deployment, _requested_pipeline(body, service.deployment)
        )
    # Validate private state before opening anything. A malformed custom caller must
    # not leave a session which has neither a task nor a complete selection record.
    runtime = RunState(selection=selection.manifest(), metadata=metadata or {})
    sessions = service.deployment.storage.sessions
    if intake_id is not None:
        assert request_hash is not None
        try:
            record, created = await asyncio.to_thread(
                sessions.open_idempotent_session,
                *intake_id,
                request_hash,
                alert,
                runtime.model_dump(mode="json"),
            )
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if not created:
            return _receipt(
                IncidentRequest(alert=alert, target_stage=target_stage, metadata=metadata or {}),
                service.deployment,
                record,
            )
    else:
        record = await asyncio.to_thread(sessions.create_session, alert)
    try:
        # Saved before register wakes deduplicated requests and before any model call.
        if intake_id is None:
            await asyncio.to_thread(save_runtime, sessions, record.session_id, runtime)
        # Pollers must see the run as started from the moment its task exists.
        if intake_id is None:
            await asyncio.to_thread(sessions.set_run_status, record.session_id, RunStatus.RUNNING)
    except Exception as exc:
        _log.exception("initial persistence failed for incident %s", record.session_id)
        # Database errors can quote rejected JSON, including private metadata. The
        # operator log keeps the cause; publicly readable notes carry only the phase.
        _record_failure(
            service.deployment,
            record.session_id,
            RuntimeError("could not persist initial incident state"),
        )
        raise HTTPException(
            status_code=503,
            detail="the incident could not be initialized",
            headers={"Retry-After": "30"},
        ) from exc

    # Registered before the task, so a run that finishes inside one loop iteration
    # cannot be unregistered before it was ever registered.
    service.runs.register(record.session_id, fingerprint=fingerprint(alert))
    task = asyncio.create_task(
        run_incident_task(
            session_id=record.session_id,
            alert=alert,
            target_stage=_TARGET_STAGES[target_stage],
            service=service,
            selection=selection,
            metadata=metadata,
        )
    )
    service.runs.attach(record.session_id, task)
    return {
        "session_id": record.session_id,
        "status": "accepted",
        "target_stage": target_stage,
        "started_at": record.started_at,
    }


def _receipt(body: IncidentRequest, deployment: Deployment, record: Any) -> dict[str, Any]:
    """Return the original admission without launching another run."""
    result = {
        "session_id": record.session_id,
        "status": "accepted",
        "target_stage": body.target_stage,
        "started_at": record.started_at,
    }
    if _show_pipeline(body, deployment):
        result["pipeline"] = record.runtime.get("selection", {}).get("name", "")
    return result


def _request_hash(body: IncidentRequest) -> str:
    """Digest every validated request field for durable key conflict detection."""
    canonical = json.dumps(body.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


async def run_incident_task(
    *,
    session_id: str,
    alert: dict[str, Any],
    target_stage: TargetStage,
    service: Service,
    selection: SelectedPipeline | None = None,
    metadata: dict[str, Any] | None = None,
    recovery: dict[str, Any] | None = None,
) -> None:
    """Run one incident, and leave the session readable whatever happens to it.

    Args:
        session_id: The session opened for this alert.
        alert: The payload that raised it.
        target_stage: How far to run.
        service: The process's resources.
        selection: The pinned admission choice; the runtime must not select it again.
        metadata: Private context, separate from all prompt input.
        recovery: A validated pre-stage checkpoint from a previous process.
    """
    try:
        await run_incident(
            alert,
            session_id,
            service.deployment,
            target_stage=target_stage,
            selection=selection,
            metadata=metadata,
            recovery=recovery,
        )
    except ApprovalPending as pending:
        # Not a failure. The incident is parked on a person, with everything needed to
        # continue it already written down.
        _log.info("session %s is awaiting approval request %s", session_id, pending.request_id)
    except asyncio.CancelledError:
        # A drain with opt-in recovery leaves the durable stage checkpoint active for
        # the next process. An operator cancellation still closes the run immediately.
        _log.info("session %s was cancelled", session_id)
        if not (
            service.runs.draining
            and service.deployment.settings.server.resume_interrupted_incidents
        ):
            record_cancellation(service.deployment, session_id)
        raise
    except Exception as exc:
        _log.exception("session %s crashed", session_id)
        _record_failure(service.deployment, session_id, exc)
    finally:
        service.runs.unregister(session_id)


def _requested_pipeline(body: IncidentRequest, deployment: Deployment) -> str | None:
    """Validate a caller's selection and preserve the historical diagnosis-only route."""
    catalog = deployment.pipelines
    assert catalog is not None, "deployment must resolve its pipeline catalog before intake"
    explicit = body.pipeline
    if explicit is None and body.target_stage in ("rca", "diagnose"):
        explicit = "diagnose"
    if explicit is not None:
        try:
            pipeline = catalog.get(explicit)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        stages = pipeline.workflow.stages
        if body.target_stage in ("rca", "diagnose") and not (
            len(stages) == 1 and stages[0].role == "rca"
        ):
            raise HTTPException(
                status_code=422,
                detail="target_stage=diagnose requires a pipeline with one RCA stage",
            )
    return explicit


async def _select(
    body: IncidentRequest, deployment: Deployment, explicit: str | None
) -> SelectedPipeline:
    """Select for the claim owner, translating configured refusals into intake errors."""
    catalog = deployment.pipelines
    assert catalog is not None, "deployment must resolve its pipeline catalog before intake"
    try:
        return await select_pipeline(
            catalog,
            alert=body.alert,
            metadata=body.metadata,
            explicit=explicit,
            selectors=deployment.pipeline_selectors,
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _show_pipeline(body: IncidentRequest, deployment: Deployment) -> bool:
    """Keep the held receipt shape for callers using only the original request fields."""
    return body.pipeline is not None or deployment.settings.customisation is not None


def _record_failure(deployment: Deployment, session_id: str, exc: BaseException) -> None:
    """Leave a crashed incident in a state a reader can tell is over.

    Written inline, unlike the routes above. This runs once, as the task unwinds, and
    the incident it is closing has already stopped: there is nothing left for the loop
    to be freed up for that these five writes are in the way of, and a hop off it would
    only add a way for the record of a crash to be lost.
    """
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


def fingerprint(alert: Mapping[str, Any]) -> str:
    """Identify an alert by what it says, not by when it arrived.

    Over the visible payload only, so an evaluation harness's own bookkeeping keys
    cannot make two posts of the same alert look different — which would defeat the
    deduplication exactly where a harness reposts on a timeout.

    Args:
        alert: The payload.

    Returns:
        A short stable hash.
    """
    canonical = json.dumps(visible_alert(alert), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]
