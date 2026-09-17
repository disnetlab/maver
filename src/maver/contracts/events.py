"""The ordered per-incident event log.

An operator must be able to reconstruct, after the fact, what was decided, on what
evidence, and before which action. That property is the reason this log exists; it is
not a debugging convenience, and it is what the typed artifacts alone cannot provide,
because they record conclusions and not the process that reached them.

Every model call is recorded, including the calls made by gates, nominators and
selectors rather than by a role. The research prototype counted only role-issued
calls, so its published cost figures had to disclose the gap; ``ModelCallEvent.caller``
exists specifically to close it.

Large request, response and transcript bodies are captured in a private local artifact
store. Events reference content-addressed manifests; repeated messages and tool schemas
share their captured bodies. The short legacy prompt digest remains useful for matching
old records, while the request reference captures the complete effective input layer.

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
    "ModelAttemptEvent",
    "ModelCallEvent",
    "ModelCallStartedEvent",
    "PolicyDecisionEvent",
    "SecondPassEvent",
    "StageEvent",
    "SubmissionEvent",
    "ToolCallEvent",
    "ToolCallStartedEvent",
]


class ModelCallStartedEvent(BaseModel):
    """One completion request, written down before it is sent.

    Two things are knowable only from before the call. A process killed mid-generation —
    a container restart, an OOM kill, a bridge that wedged and was eventually noticed —
    writes no completion event at all, so without this row the turn it died in leaves no
    trace and the run reads as having stopped cleanly between turns. And the request
    itself: :class:`ModelCallEvent` reports what came back and what it cost, never what
    went out.

    ``call_id`` joins the started record, provider attempts and terminal summary.
    ``request`` references ordered message and tool-schema blocks, plus provider options
    when the client exposes them. Capture identifies the SDK invocation or neutral
    request layer explicitly; it does not claim to record the HTTP wire payload.
    The legacy ``digest`` covers only system and messages and remains compatible with
    earlier records. The captured manifest has its own full SHA-256 identity.
    """

    kind: Literal["model_call_started"] = "model_call_started"
    call_id: str | None = None
    request: dict[str, Any] = Field(
        default_factory=dict,
        description="Reference to the captured request manifest; SDK API-key options excluded.",
    )
    caller: str = Field(description="What issued the call, as on the completion event.")
    model: str = Field(description="The resolved model identifier sent to the provider.")
    profile: str = Field(description="The model profile this call was configured from.")
    message_count: int = Field(
        description=(
            "How many messages the request carried, the system prompt aside. It locates "
            "the transcript prefix this request should be; the digest then confirms it. "
            "Full messages are retained through content-addressed blocks in request, "
            "so repeated transcript prefixes do not duplicate their bodies."
        )
    )
    digest: str = Field(
        description=(
            "Twelve hex characters over the system prompt and the messages, taken the "
            "way a sampling set is. It detects a difference and does not explain one, "
            "which is why what replaced the transcript is recorded where it happened."
        )
    )


class ModelCallEvent(BaseModel):
    """One logical call summary; token counts describe only its final attempt.

    Use model_attempt events for provider usage across retries. Never sum both event
    kinds together; their separate durations and token counts describe overlapping work.
    """

    kind: Literal["model_call"] = "model_call"
    call_id: str | None = None
    response: dict[str, Any] = Field(default_factory=dict)
    error_detail: dict[str, Any] = Field(default_factory=dict)
    usage_scope: Literal["final_attempt"] = "final_attempt"
    attempts_observed: int | None = Field(
        default=None,
        description="Provider attempts captured separately; zero when a custom client exposes no attempt observations.",
    )
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
    duration_ms: int = Field(
        description=(
            "Wall clock the caller waited on this call, every retry and every backoff "
            "included. One meaning on both paths: recorded as the last attempt on "
            "success and the whole call on failure, the same column answered two "
            "questions and a mixed aggregate was neither."
        )
    )
    last_attempt_ms: int | None = Field(
        default=None,
        description=(
            "How long the attempt this row's outcome came from took. Null where no "
            "single attempt produced it."
        ),
    )
    attempt: int = Field(default=1, description="1 for the first try, higher on retry.")
    error: str | None = None
    notes: list[str] = Field(
        default_factory=list,
        description=(
            "What the client repaired in this response before anything above it saw "
            "it. A repeated tool-call identifier drops a result wherever results are "
            "matched to calls by identifier, and the transcript then shows an "
            "identifier the provider never sent, so the repair is a harness "
            "intervention on the record and belongs in it."
        ),
    )


class ModelAttemptEvent(BaseModel):
    """One provider attempt; separate from the once-per-logical-call summary.

    Sum this event's usage to include usage reported on failed retries. A null count
    means unreported; it does not establish that the failed attempt consumed no tokens.
    """

    kind: Literal["model_attempt"] = "model_attempt"
    call_id: str
    caller: str
    model: str
    profile: str
    attempt: int
    status: Literal["succeeded", "failed", "cancelled"]
    duration_ms: int
    usage: dict[str, int | None] = Field(default_factory=dict)
    response: dict[str, Any] = Field(default_factory=dict)
    error: dict[str, Any] = Field(default_factory=dict)


class ToolCallStartedEvent(BaseModel):
    """A tool attempt recorded before middleware, dispatch or external I/O."""

    kind: Literal["tool_call_started"] = "tool_call_started"
    tool: str
    call_id: str | None = None
    invocation_id: str
    arguments: dict[str, Any]
    rationale: str | None = None


class ToolCallEvent(BaseModel):
    """One tool invocation and the shape of its result."""

    kind: Literal["tool_call"] = "tool_call"
    tool: str
    call_id: str | None = Field(
        default=None,
        description=(
            "The provider's identifier for the call this event describes, which is how "
            "it aligns to a position in the stored transcript and how a refusal joins "
            "to the call it refused. Null for a call the harness issued rather than "
            "the model."
        ),
    )
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
    invocation_id: str | None = None
    status: (
        Literal["completed", "failed", "incomplete", "refused", "paused", "cancelled"] | None
    ) = None
    truncated: bool | None = None
    duration_ms: int | None = None
    full_result_chars: int | None = None
    result: dict[str, Any] = Field(default_factory=dict)
    output_artifacts: list[dict[str, Any]] = Field(default_factory=list)
    incomplete: bool = False
    unavailable: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    upstream_truncated: bool = False
    error: str | None = None
    transport: str | None = Field(default=None, description="Which transport served the call.")
    retrieval: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "What a retrieval call returned: per record its ``record_id``, the "
            "``handle`` the model was shown in its place, and its ``score``. The gated "
            "second pass records the same thing as ``record_ids`` under "
            "``SecondPassEvent.queries``; one concept keeps one vocabulary. The query "
            "is not repeated because it is already in ``arguments``. Empty for a tool "
            "that retrieves nothing. Without it a role's retrieval leaves only the "
            "query and a character count, and which precedents a role was actually "
            "served — the question every memory-delivery analysis asks — is answerable "
            "for the second pass and not for the roles."
        ),
    )


class SubmissionEvent(BaseModel):
    """An attempt by a role to exit through its submission tool."""

    kind: Literal["submission"] = "submission"
    source: Literal["workflow", "role_loop", "learning"] | None = None
    artifact: str = Field(description="The artifact type submitted, e.g. root_cause.")
    accepted: bool
    attempt: int
    validation_error: str | None = Field(
        default=None,
        description="The schema or invariant failure, verbatim, when rejected.",
    )


class InterventionEvent(BaseModel):
    """A runtime control acting on a stalled or looping generation, or failing to."""

    kind: Literal["intervention"] = "intervention"
    control: Literal[
        "generation_cap",
        "no_tool_nudge",
        "repeat_breaker",
        "finalizer_rescue",
        "submit_guard",
        "submit_normalization",
        "tool_choice",
        "turn_limit",
        "timeout",
        "hook_failure",
        "middleware_context",
    ]
    action: Literal["warned", "refused", "rescued", "bypassed", "ended_run", "injected"]
    """``bypassed`` is a control that was reached and not applied — because its own
    budget was spent, because the endpoint being served cannot accept it, or because it
    raised and was skipped. It is not ``warned``: nothing was said to the model and
    nothing was refused. It exists as its
    own value because a control's dose is a divisor, and a bypass counted as a
    consultation inflates the denominator."""
    detail: str
    context: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "The messages this control sent in place of the run's transcript, verbatim "
            "and once. Empty for every control that only appends to the transcript, "
            "since there the transcript is the record. The finalizer rescue does not "
            "append: it compacts the run into a short submit-only brief and sends that "
            "instead, so a rescued run's decisive turn was made on messages that, "
            "without this, exist nowhere once the process has ended. Only the brief is "
            "here; whatever the rescued turns then produced is appended to the "
            "transcript as any other turn's is."
        ),
    )


class PolicyDecisionEvent(BaseModel):
    """An authority decision about a write-capable tool call."""

    kind: Literal["policy_decision"] = "policy_decision"
    tool: str
    call_id: str | None = Field(
        default=None,
        description=(
            "The provider's identifier for the call this event describes, which is how "
            "it aligns to a position in the stored transcript and how a refusal joins "
            "to the call it refused. Null for a call the harness issued rather than "
            "the model."
        ),
    )
    policy: Literal["auto", "prompt", "deny"]
    decision: Literal["approved", "denied", "timed_out", "pending"]
    reason: str = ""
    request_id: str | None = None
    decided_by: str | None = Field(default=None, description="Who decided, when a human did.")


class StageEvent(BaseModel):
    """A role entering, leaving, or being handed work back."""

    kind: Literal["stage"] = "stage"
    run: dict[str, Any] = Field(
        default_factory=dict,
        description="Captured stage/run manifest, including private transcript and identity references.",
    )
    source: Literal["workflow", "role_loop", "learning"] | None = None
    stage: str
    event: Literal["entered", "exited", "handed_back", "halted", "failed", "cancelled", "paused"]
    detail: str = ""


class SecondPassEvent(BaseModel):
    """One decision taken by the gated second pass.

    Recorded for every incident where the mechanism is enabled, including the incidents
    where the gate declines and no model call is made. A skip that leaves no record
    makes the mechanism's dose unmeasurable, and the dose is what an arm gap has to be
    divided by.
    """

    kind: Literal["second_pass"] = "second_pass"
    policy: str = Field(description="The composition, named for the settings that select it.")
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
    ModelCallStartedEvent
    | ModelCallEvent
    | ModelAttemptEvent
    | ToolCallEvent
    | ToolCallStartedEvent
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
            "Monotonic position within the session, allocated once for the session "
            "and shared by every stage's journal. Ordering is by this field and not by "
            "timestamp: timestamps have collided at the resolution the stores use, and "
            "text-typed timestamp columns have silently broken range filters. A process "
            "that did not run the incident — a resume after a decision, or a distilling "
            "job — begins a new order, so its events sort among the original ones."
        )
    )
    stage: str = Field(description="The role or mechanism the event occurred under.")
    run_id: str | None = Field(
        default=None, description="The role run this event belongs to, when there is one."
    )
    occurred_at: datetime
    payload: EventPayload
