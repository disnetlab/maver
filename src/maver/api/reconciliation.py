"""Operator decisions for interrupted runs with uncertain external effects."""

import asyncio
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from maver.api.deps import ServiceDep, admit
from maver.api.incidents import fingerprint, start_incident
from maver.contracts.lifecycle import RunStatus
from maver.pipelines import restore_selection

router = APIRouter()


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
        target = state.get("target_stage")
        if target not in {"full", "diagnose"}:
            raise HTTPException(status_code=409, detail="saved target stage cannot be restored")
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
    claim = admit(service, fingerprint=fingerprint(record.alert)) if selected else None
    try:
        decided = await asyncio.to_thread(sessions.decide_reconciliation, session_id, notes=note)
        if not decided:
            raise HTTPException(status_code=409, detail="reconciliation already decided")
        if selected is None:
            return {"session_id": session_id, "status": "closed"}
        receipt = await start_incident(
            record.alert,
            target_stage=target,
            service=service,
            selection=selected,
            metadata={"reconciliation_of": session_id},
        )
        await asyncio.to_thread(
            sessions.update_artefacts,
            session_id,
            notes=f"{note}; new incident: {receipt['session_id']}",
        )
        return {"session_id": session_id, "status": "replanned", "new_incident": receipt}
    finally:
        if claim is not None:
            service.runs.release(claim)
