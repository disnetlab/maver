"""Operator outcome review after an incident has closed."""

import asyncio
from dataclasses import asdict
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, model_validator

from maver.api.deps import ServiceDep
from maver.memory.lifecycle import IllegalTransition
from maver.orchestration.outcome_feedback import submit_feedback

router = APIRouter()


class FeedbackRequest(BaseModel):
    """An explicitly attributed later verdict and its supporting context."""

    verdict: Literal["confirmed", "rejected", "corrected"]
    decided_by: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    evidence: str = Field(min_length=1)
    corrected_cause: str = ""

    @model_validator(mode="after")
    def require_correction(self) -> "FeedbackRequest":
        """A corrected verdict must supply the corrected cause."""
        if not all(
            value.replace("\x00", "").strip()
            for value in (self.decided_by, self.reason, self.evidence)
        ):
            raise ValueError("decided_by, reason, and evidence must contain text")
        if self.verdict == "corrected" and not self.corrected_cause.replace("\x00", "").strip():
            raise ValueError("corrected_cause is required for a corrected verdict")
        if self.verdict != "corrected" and self.corrected_cause:
            raise ValueError("corrected_cause is only accepted with a corrected verdict")
        return self


@router.post("/incidents/{session_id}/feedback", status_code=201)
async def record_feedback(
    session_id: str, body: FeedbackRequest, service: ServiceDep
) -> dict[str, object]:
    """Record a later outcome verdict without rewriting the agent's artifacts."""
    storage = service.deployment.storage
    try:
        feedback = await asyncio.to_thread(
            submit_feedback,
            storage.sessions,
            storage.index,
            session_id=session_id,
            **body.model_dump(),
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ValueError, IllegalTransition) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return asdict(feedback)


@router.get("/incidents/{session_id}/feedback")
async def list_feedback(session_id: str, service: ServiceDep) -> list[dict[str, object]]:
    """Read the append-only history of operator outcome verdicts."""
    sessions = service.deployment.storage.sessions
    record = await asyncio.to_thread(sessions.get_session, session_id)
    if record is None:
        raise HTTPException(status_code=404, detail="incident not found")
    history = await asyncio.to_thread(sessions.feedback_for, session_id)
    return [asdict(entry) for entry in history]
