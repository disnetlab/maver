"""Explicitly retry a quarantined Kafka record through ordinary admission."""

import asyncio
import base64
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import ValidationError

from maver.api.deps import ServiceDep
from maver.api.incidents import IncidentRequest, accept_incident

router = APIRouter()


@router.post("/quarantine/{record_id}/retry", status_code=202)
async def retry_quarantine(
    record_id: str,
    service: ServiceDep,
    body: IncidentRequest | None = None,
) -> dict[str, Any]:
    """Retry one saved delivery, optionally replacing its malformed request body.

    Args:
        record_id: The locally inspected quarantine record.
        service: The process carrying the incident runtime.
        body: Corrected request. If omitted, the original bytes must parse.

    Returns:
        The durable admission receipt.

    Raises:
        HTTPException: 404 for an unknown record; 422 if the original body is invalid.
    """
    sessions = service.deployment.storage.sessions
    record = await asyncio.to_thread(sessions.get_quarantine, record_id)
    if record is None:
        raise HTTPException(status_code=404, detail="quarantine record not found")
    if body is None:
        try:
            body = IncidentRequest.model_validate_json(base64.b64decode(record.raw_base64))
        except ValidationError as exc:
            raise HTTPException(
                status_code=422, detail="provide a corrected incident body"
            ) from exc
    receipt = await accept_incident(body, service, intake_id=("quarantine", record_id))
    await asyncio.to_thread(sessions.mark_quarantine_retried, record_id, receipt["session_id"])
    return receipt
