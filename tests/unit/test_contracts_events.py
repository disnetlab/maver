"""The audit event log: discrimination, ordering, and full cost coverage."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from maver.contracts import AuditEvent, ModelCallEvent, PolicyDecisionEvent

WHEN = datetime(2026, 8, 31, 12, 0, tzinfo=UTC)


def _event(payload: dict[str, object], **overrides: object) -> AuditEvent:
    base: dict[str, object] = {
        "session_id": "s-1",
        "sequence": 1,
        "stage": "rca",
        "occurred_at": WHEN,
        "payload": payload,
    }
    base.update(overrides)
    return AuditEvent(**base)  # type: ignore[arg-type]


def test_payload_is_discriminated_by_kind() -> None:
    event = _event(
        {
            "kind": "tool_call",
            "tool": "get_metrics",
            "arguments": {"query": "up"},
            "rationale": "check scrape health",
            "succeeded": True,
            "result_chars": 412,
        }
    )
    assert event.payload.kind == "tool_call"


def test_an_unknown_kind_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _event({"kind": "not_a_kind"})


def test_helper_calls_are_attributable_so_none_escapes_cost_accounting() -> None:
    """The prototype counted only role-issued calls and had to disclose the gap."""
    call = ModelCallEvent(
        caller="quality_gate",
        model="openai/qwen3.6-35b-a3b",
        profile="qwen36-35b",
        sampling={"temperature": 0.0, "top_p": 1.0, "frequency_penalty": 0.3},
        prompt_tokens=1200,
        completion_tokens=180,
        finish_reason="stop",
        duration_ms=2400,
    )
    assert call.caller == "quality_gate"


def test_the_whole_sampling_set_is_recorded_together() -> None:
    call = ModelCallEvent(
        caller="rca",
        model="m",
        profile="p",
        sampling={"temperature": 1.0, "top_p": 0.95, "frequency_penalty": 0.0},
        duration_ms=1,
    )
    assert set(call.sampling) == {"temperature", "top_p", "frequency_penalty"}


def test_unreported_cached_tokens_stay_null_rather_than_zero() -> None:
    """An endpoint that does not report the field has not reported a zero."""
    call = ModelCallEvent(caller="rca", model="m", profile="p", sampling={}, duration_ms=1)
    assert call.cached_tokens is None


def test_a_refusal_is_a_recorded_decision() -> None:
    event = _event(
        {
            "kind": "policy_decision",
            "tool": "exec_kubectl",
            "policy": "deny",
            "decision": "denied",
            "reason": "write attempted outside the execution stage",
        },
        stage="planning",
    )
    assert isinstance(event.payload, PolicyDecisionEvent)
    assert event.payload.decision == "denied"
