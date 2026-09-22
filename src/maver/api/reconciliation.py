"""Operator decisions for interrupted runs with uncertain external effects."""

import asyncio
import logging
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from maver.api.deps import ServiceDep, admit
from maver.api.incidents import fingerprint, schedule_prepared_incident
from maver.contracts.lifecycle import RunStatus
from maver.contracts.runtime import RunState
from maver.pipelines import restore_selection

router = APIRouter()
_log = logging.getLogger(__name__)


class ReconciliationDecision(BaseModel):
    """The human's fresh observation and chosen disposition."""

    action: Literal["close", "replan"]
    reason: str = Field(min_length=1)
    observed_state: str = Field(min_length=1)
    decided_by: str = Field(min_length=1)


@router.get("/incidents/{session_id}/reconciliation")
async def read_reconciliation(session_id: str, service: ServiceDep) -> dict[str, Any]:
    """Show safe metadata; detailed evidence is available from the local CLI."""
    record = await asyncio.to_thread(service.deployment.storage.sessions.get_session, session_id)
    if record is None:
        raise HTTPException(status_code=404, detail="incident not found")
    if record.run_status != RunStatus.NEEDS_RECONCILIATION:
        raise HTTPException(status_code=409, detail=f"incident is {record.run_status}")
    return {
        "session_id": session_id,
        "run_status": record.run_status,
        "interrupted_stage": record.current_stage,
        "has_checkpoint": record.checkpoint is not None,
        "has_execution_result": record.execution_result is not None,
        "inspection": "Use maver incidents reconcile-show locally for private evidence.",
    }


@router.post("/incidents/{session_id}/reconciliation")
async def decide_reconciliation(
    session_id: str, body: ReconciliationDecision, service: ServiceDep
) -> dict[str, Any]:
    """Keep the decision alive through shutdown, including its database commit."""
    if service.runs.draining:
        raise HTTPException(
            status_code=503, detail="service is draining", headers={"Retry-After": "30"}
        )
    decision = asyncio.create_task(
        _decide_reconciliation(session_id, body, service),
        name="maver-reconciliation-decision",
    )
    service.runs.track_admission(decision)
    cancelled = False
    while not decision.done():
        try:
            await asyncio.shield(decision)
        except asyncio.CancelledError:
            cancelled = True
        except Exception:
            break
    if cancelled:
        if not decision.cancelled() and (error := decision.exception()) is not None:
            _log.error("reconciliation failed after request cancellation", exc_info=error)
        raise asyncio.CancelledError
    return decision.result()


async def _decide_reconciliation(
    session_id: str, body: ReconciliationDecision, service: ServiceDep
) -> dict[str, Any]:
    """Close the uncertain run, optionally starting a fresh linked incident."""
    sessions = service.deployment.storage.sessions
    record = await asyncio.to_thread(sessions.get_session, session_id)
    if record is None:
        raise HTTPException(status_code=404, detail="incident not found")
    if record.run_status != RunStatus.NEEDS_RECONCILIATION:
        raise HTTPException(status_code=409, detail=f"incident is {record.run_status}")

    selected = None
    target = "full"
    if body.action == "replan":
        checkpoint = record.checkpoint or {}
        state = checkpoint.get("state")
        if checkpoint.get("adapter") or service.deployment.adapter is not None:
            raise HTTPException(
                status_code=409,
                detail="adapter episode cannot be reattached; close it and start a new episode through its driver",
            )
        if not isinstance(state, dict) or state.get("session_id") != session_id:
            raise HTTPException(status_code=409, detail="saved checkpoint is incomplete")
        # Approval checkpoints store this explicitly as well as in the state. The
        # latter can be absent for a continuation of an older incident.
        saved_target = state.get("target_stage", checkpoint.get("target_stage"))
        if saved_target not in {"full", "diagnose"}:
            raise HTTPException(status_code=409, detail="saved target stage cannot be restored")
        target = str(saved_target)
        saved_runtime = state.get("runtime")
        if not isinstance(saved_runtime, dict):
            raise HTTPException(status_code=409, detail="saved runtime is incomplete")
        if saved_runtime.get("deployment_digest") != service.deployment.runtime_digest:
            raise HTTPException(
                status_code=409, detail="deployment behavior changed since admission"
            )
        try:
            catalog = service.deployment.pipelines
            if catalog is None:
                raise ValueError("pipeline catalog unavailable")
            selected = restore_selection(catalog, state["runtime"]["selection"])
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=409, detail=f"saved pipeline cannot be restored: {exc}"
            ) from exc

    note = (
        f"Reconciliation by {body.decided_by}: {body.action}; "
        f"observed state: {body.observed_state}; reason: {body.reason}"
    )
    if selected is None:
        decided = await asyncio.to_thread(sessions.decide_reconciliation, session_id, notes=note)
        if not decided:
            raise HTTPException(status_code=409, detail="reconciliation already decided")
        return {"session_id": session_id, "status": "closed"}

    claim = admit(service, fingerprint=fingerprint(record.alert))
    metadata = {"reconciliation_of": session_id}
    runtime = RunState(selection=selected.manifest(), metadata=metadata)

    async def commit_and_schedule() -> dict[str, Any]:
        try:
            prepared = await asyncio.to_thread(
                sessions.open_replan_session,
                session_id,
                runtime=runtime.model_dump(mode="json"),
                notes=note,
            )
            if prepared is None:
                raise HTTPException(status_code=409, detail="reconciliation already decided")
            receipt = schedule_prepared_incident(
                prepared,
                target_stage=target,
                service=service,
                selection=selected,
                metadata=metadata,
            )
            return {"session_id": session_id, "status": "replanned", "new_incident": receipt}
        finally:
            service.runs.release(claim)

    return await commit_and_schedule()
