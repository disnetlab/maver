"""The ordered per-incident event log.

An operator must be able to reconstruct, after the fact, what was decided, on what
evidence, and before which action. That property is the reason this log exists; it is
not a debugging convenience, and it is what the typed artifacts alone cannot provide,
because they record conclusions and not the process that reached them.

Every model call is recorded, including the calls made by gates, nominators and
selectors rather than by a role. The research prototype counted only role-issued
calls, so its published cost figures had to disclose the gap; ``ModelCallEvent.caller``
exists specifically to close it.

The log is append-only. Records are never rewritten, so a correction is a new event.
"""

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

__all__ = [
    "AuditEvent",
    "EnvironmentEvent",
    "EventPayload",
    "InterventionEvent",
    "ModelCallEvent",
    "PolicyDecisionEvent",
    "SecondPassEvent",
    "StageEvent",
    "SubmissionEvent",
    "ToolCallEvent",
]


class ModelCallEvent(BaseModel):
    """One completion request, as reported by the serving endpoint."""

    kind: Literal["model_call"] = "model_call"
    caller: str = Field(
        description=(
            "What issued the call. A role name for the main loop, or the mechanism "
            "name for a helper call such as a quality gate, nominator or selector. "
            "Every call has a caller so that none escapes cost accounting."
        )
    )
    model: str = Field(description="The resolved model identifier sent to the provider.")
    profile: str = Field(description="The model profile this call was configured from.")
    sampling: dict[str, Any] = Field(
        description=(
            "The complete sampling set applied to this call, recorded together. A "
            "partial record is indistinguishable from a complete one after the fact, "
            "which is how a configuration drift went unnoticed for weeks."
        )
    )
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None
    cached_tokens: int | None = Field(
        default=None,
        description=(
            "Server-reported cached prompt tokens. Null when the endpoint does not "
            "report the field, which is not the same as zero."
        ),
    )
    finish_reason: str | None = Field(
        default=None,
        description=(
            "The provider's stop reason. A length stop means the generation hit the "
            "cap and rarely carries a complete tool call."
        ),
    )
    duration_ms: int
    attempt: int = Field(default=1, description="1 for the first try, higher on retry.")
    error: str | None = None


class ToolCallEvent(BaseModel):
    """One tool invocation and the shape of its result."""

    kind: Literal["tool_call"] = "tool_call"
    tool: str
    arguments: dict[str, Any]
    rationale: str | None = Field(
        default=None,
        description=(
            "The model's stated reason for the call. Required on every non-submission "
            "tool so that intent survives across turns even when the provider "
            "discards its reasoning channel."
        ),
    )
    succeeded: bool
    result_chars: int
    truncated: bool = False
    duration_ms: int = 0
    transport: str | None = Field(default=None, description="Which transport served the call.")


class SubmissionEvent(BaseModel):
    """An attempt by a role to exit through its submission tool."""

    kind: Literal["submission"] = "submission"
    artifact: str = Field(description="The artifact type submitted, e.g. root_cause.")
    accepted: bool
    attempt: int
    validation_error: str | None = Field(
        default=None,
        description="The schema or invariant failure, verbatim, when rejected.",
    )


class InterventionEvent(BaseModel):
    """A runtime control acting on a stalled or looping generation."""

    kind: Literal["intervention"] = "intervention"
    control: Literal[
        "generation_cap",
        "no_tool_nudge",
        "repeat_breaker",
        "finalizer_rescue",
        "turn_limit",
        "timeout",
    ]
    action: Literal["warned", "refused", "rescued", "ended_run"]
    detail: str


class PolicyDecisionEvent(BaseModel):
    """An authority decision about a write-capable tool call."""

    kind: Literal["policy_decision"] = "policy_decision"
    tool: str
    policy: Literal["auto", "prompt", "deny"]
    decision: Literal["approved", "denied", "timed_out", "pending"]
    reason: str = ""
    request_id: str | None = None
    decided_by: str | None = Field(default=None, description="Who decided, when a human did.")


class StageEvent(BaseModel):
    """A role entering, leaving, or being handed work back."""

    kind: Literal["stage"] = "stage"
    stage: str
    event: Literal["entered", "exited", "handed_back", "halted", "failed"]
    detail: str = ""


class SecondPassEvent(BaseModel):
    """One decision taken by the gated second pass.

    Recorded for every incident where the mechanism is enabled, including the incidents
    where the gate declines and no model call is made. A skip that leaves no record
    makes the mechanism's dose unmeasurable, and the dose is what an arm gap has to be
    divided by.
    """

    kind: Literal["second_pass"] = "second_pass"
    policy: str = Field(description="The named composition, e.g. ma-s.")
    step: Literal["gate", "search", "nomination", "brief", "selection", "skip"]
    fired: bool | None = None
    reason: str = ""
    queries: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Per query: its name, its text, and the ids it returned. Enough to replay "
            "the retrieval offline against a frozen corpus."
        ),
    )
    precedent_ids: list[str] = Field(default_factory=list)
    withheld_count: int = Field(
        default=0,
        description=(
            "Size of the leave-one-out set applied. Zero on a generalisation arm means "
            "the cell served its own precedent and is void."
        ),
    )
    family: str | None = None
    confidence: float | None = None
    weakness: str = Field(
        default="",
        description=(
            "The quality audit's own words for the single biggest weakness it found. "
            "Kept because it is the only text the gate produces about why it fired."
        ),
    )
    winner: str | None = None
    label_to_source: dict[str, str] = Field(default_factory=dict)
    selector_prompt: str | None = Field(
        default=None,
        description=(
            "The exact blinded prompt, stored on the selection event only, so a "
            "selection can be re-judged offline."
        ),
    )
    selector_response: str | None = None


class EnvironmentEvent(BaseModel):
    """An adapter acting on the environment that hosts this incident.

    Named for the environment rather than for any one of them, so the closed union
    below stays free of a benchmark's name while an adapter can still write into the
    incident's own ordered log instead of a ledger beside it.
    """

    kind: Literal["environment"] = "environment"
    adapter: str
    action: Literal["acquired", "submitted", "graded", "released", "failed"]
    detail: str = ""
    ok: bool = True


EventPayload = Annotated[
    ModelCallEvent
    | ToolCallEvent
    | SubmissionEvent
    | InterventionEvent
    | PolicyDecisionEvent
    | StageEvent
    | SecondPassEvent
    | EnvironmentEvent,
    Field(discriminator="kind"),
]


class AuditEvent(BaseModel):
    """One entry in an incident's ordered event log."""

    session_id: str
    sequence: int = Field(
        description=(
            "Monotonic position within the session. Ordering is by this field and not "
            "by timestamp: timestamps have collided at the resolution the stores use, "
            "and text-typed timestamp columns have silently broken range filters."
        )
    )
    stage: str = Field(description="The role or mechanism the event occurred under.")
    run_id: str | None = Field(
        default=None, description="The role run this event belongs to, when there is one."
    )
    occurred_at: datetime
    payload: EventPayload
