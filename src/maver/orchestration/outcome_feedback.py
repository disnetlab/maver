"""Operator verdicts on closed incidents, kept apart from run artifacts."""

from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

from maver.memory.ports import OutcomeFeedback, SessionStore, TextSearch
from maver.memory.review import reviewed_memory


def submit_feedback(
    sessions: SessionStore,
    index: TextSearch,
    *,
    session_id: str,
    verdict: Literal["confirmed", "rejected", "corrected"],
    decided_by: str,
    reason: str,
    evidence: str,
    corrected_cause: str = "",
) -> OutcomeFeedback:
    """Persist a human verdict and refresh an existing distilled record's metadata."""
    feedback = OutcomeFeedback(
        feedback_id=str(uuid4()),
        session_id=session_id,
        verdict=verdict,
        decided_by=decided_by,
        reason=reason,
        evidence=evidence,
        corrected_cause=corrected_cause,
        created_at=datetime.now(UTC).isoformat(),
    )
    feedback = sessions.record_feedback(feedback)
    memory = index.open_record(session_id)
    if memory is not None:
        state = sessions.get_states([session_id])[session_id]
        index.upsert_summary(
            reviewed_memory(memory, state=state, verdict=verdict, corrected_cause=corrected_cause)
        )
    return feedback


def latest_feedback_packet(sessions: SessionStore, session_id: str) -> dict[str, object] | None:
    """Render the latest human verdict for delayed Learning without changing observations."""
    history = sessions.feedback_for(session_id)
    if not history:
        return None
    latest = history[-1]
    packet: dict[str, object] = {
        "source": "operator",
        "operator": latest.decided_by,
        "reason": latest.reason,
        "evidence": latest.evidence,
        "mitigation": {"success": latest.verdict == "confirmed"},
    }
    if latest.verdict == "corrected":
        packet["diagnosis"] = {"success": False}
        packet["ground_truth_root_cause"] = latest.corrected_cause
    return packet
