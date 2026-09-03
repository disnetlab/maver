"""Turning one distilled incident into a record a later incident can find.

The write half of the experience store. What reaches the corpus is a projection of what
the offline pass concluded, never a transcript: a retrieval corpus of chat logs is not a
memory system, and the two searchable documents here are the whole of what a search can
match on.

The record is keyed by the incident, so re-distilling a session replaces its record
rather than adding a second one. That matters more than it sounds: a corpus that
accumulated one record per attempt would let a session that was distilled twice outvote
one that was distilled once, purely by being retried.
"""

import logging
from collections.abc import Mapping
from typing import Any

from kira.contracts.artifacts import LearningPacket, LearningResult
from kira.memory.ports import SessionMemory, SessionStore, TextSearch
from kira.memory.projections import build_projections
from kira.settings.groups import MemorySettings

__all__ = ["distil"]

_log = logging.getLogger(__name__)


def distil(
    packet: LearningPacket,
    result: LearningResult,
    *,
    index: TextSearch,
    sessions: SessionStore,
    settings: MemorySettings | None = None,
) -> SessionMemory:
    """Write one incident's lesson into the corpus and mark the session distilled.

    Args:
        packet: What the offline pass was given about the incident.
        result: What it concluded.
        index: The corpus. Which campaign's corpus it is was decided when the store
            was opened, so nothing here has to carry it.
        sessions: Where the session lives, so it can be marked distilled.
        settings: The memory group, which decides whether this record is indexed under
            its own identifiers or under placeholders. None uses the shipped defaults,
            which is what a caller with nothing to say about it has.

    Returns:
        The record that was written.
    """
    memory = settings if settings is not None else MemorySettings()
    alert = dict(packet.alert)
    root_cause = packet.root_cause or {}
    projections = build_projections(
        result,
        alert=alert,
        external_feedback=packet.external_feedback,
        evidence_observations=_pre_feedback_observations(packet),
        mask_identifiers=memory.mask_identifiers,
        mask_namespaces=sorted(memory.mask_namespaces),
        affected_resource=str(root_cause.get("affected_resource") or ""),
    )
    record = SessionMemory(
        session_id=packet.session_id,
        evidence_text=projections.evidence_text,
        lesson_text=projections.lesson_text,
        incident_summary=result.summary,
        # A snapshot at distillation time. Retrieval resolves the current state from the
        # session store at rank time and falls back to this only for a record whose
        # session is gone.
        lifecycle_state=packet.lifecycle_state,
        fault_category=result.fault_category,
        alert_name=str(alert.get("alertname") or ""),
        namespace=str(alert.get("namespace") or ""),
        affected_resource=str(root_cause.get("affected_resource") or ""),
        verification_verdict=_verdict(result, packet),
        corrected_cause=projections.corrected_cause,
        **_evaluation(packet.external_feedback),
    )
    index.upsert_summary(record)
    sessions.mark_distilled(packet.session_id)
    _log.info("distilled session %s into the corpus", packet.session_id)
    return record


def _verdict(result: LearningResult, packet: LearningPacket) -> str:
    """Prefer the pass's own verdict, and fall back to the check that actually ran."""
    stated = (result.verification_verdict or "").strip()
    if stated:
        return stated
    verify = packet.verify_result or {}
    return str(verify.get("verdict") or "")


def _pre_feedback_observations(packet: LearningPacket) -> list[str] | None:
    """The observations the diagnosis made before any external grade was visible.

    An empty list is meaningful and is not None: for a real incident it says to index
    the identifiers alone rather than falling back to text the model wrote after it had
    been told the answer.
    """
    observations: list[str] = []
    seen: set[str] = set()
    for summary in packet.trace_summaries:
        if summary.role != "rca":
            continue
        for item in summary.key_observations:
            cleaned = " ".join(str(item or "").split())
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                observations.append(cleaned)
    return observations


def _evaluation(feedback: Mapping[str, Any] | None) -> dict[str, Any]:
    """Summarise an external grade without flattening partial credit into a verdict.

    A lifecycle state describes the pipeline's own run and must not be overwritten by a
    grader, so this is separate. Keeping the component verdicts beside the coarse label
    is what stops a half-right cell from reading as a failure.
    """
    if not feedback:
        return {}
    diagnosis = _section(feedback, "diagnosis")
    mitigation = _section(feedback, "mitigation")
    dx = _flag(diagnosis.get("success"))
    mit = _flag(mitigation.get("success"))
    score = _score(diagnosis.get("score"))

    if dx is None or mit is None:
        label = "unknown"
    elif dx and mit:
        label = "success"
    elif dx or mit or (score is not None and score > 0):
        label = "partial"
    else:
        label = "failure"

    out: dict[str, Any] = {
        "evaluation_label": label,
        "evaluation_source": str(feedback.get("source") or "external"),
    }
    if dx is not None:
        out["evaluation_diagnosis_success"] = dx
    if score is not None:
        out["evaluation_diagnosis_score"] = score
    if mit is not None:
        out["evaluation_mitigation_success"] = mit
    return out


def _section(feedback: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    part = feedback.get(name)
    return part if isinstance(part, Mapping) else {}


def _flag(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _score(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if number == number and abs(number) != float("inf") else None
