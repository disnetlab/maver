"""The loop's runtime controls, as decisions separated from the loop that applies them.

Everything here is pure: given counters and a turn's calls, it says what the loop should
do and what text to say. The loop owns the state; this module owns the thresholds and
the wording.

Two departures from the prototype, both deliberate:

* Limits are a per-run parameter object rather than module-level constants frozen at
  import. The prototype read its environment once when ``loop_policy`` was first
  imported, so a setting changed after that had no effect and tests had to monkeypatch
  private module attributes.
* A repeated call's signature excludes the model's ``rationale``. See
  :class:`RepeatedCallTracker`.
"""

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from kira.llm.types import Message, ToolCall
from kira.prompts.loader import read_asset, render
from kira.settings.groups import RuntimeSettings

__all__ = [
    "RATIONALE_ARG",
    "LoopLimits",
    "RepeatDecision",
    "RepeatedCallTracker",
    "compact_rescue_context",
    "control",
    "finalizer_rescue_message",
    "mixed_submit_message",
    "no_tool_nudge_message",
    "repair_submission",
    "repeat_breaker_message",
    "role_system_prompt",
    "truncate_text",
]

RATIONALE_ARG = "rationale"
"""The working-trace argument injected into every non-submit tool schema.

It is the model's carried-forward reasoning, not a tool parameter, so it is stripped
before dispatch and before a repeat signature is taken.
"""


def control(name: str, /, **values: str) -> str:
    """Render one control block.

    Args:
        name: File stem under ``prompts/control``.
        **values: Substitutions for the block's ``{{MARKER}}`` placeholders.

    Returns:
        The rendered text.

    Raises:
        PromptError: When a marker is unfilled or a value is unused.
    """
    return render(read_asset(f"control/{name}.txt"), values)


@dataclass(frozen=True, slots=True)
class LoopLimits:
    """Every threshold the loop consults, resolved once per run.

    Defaults match the values every published measurement used.
    """

    max_turns: int = 20
    no_tool_nudges: int = 2
    submit_guard_rejections: int = 2
    submit_validation_rejections: int = 3
    repeat_soft: int = 3
    repeat_hard: int = 6
    repeat_stop: int = 9
    max_tool_calls_per_turn: int = 8
    keep_reasoning_max_chars: int = 6000
    finalizer_rescues: int = 1
    finalizer_min_content_chars: int = 800
    finalizer_tool_results: int = 8
    finalizer_tool_result_chars: int = 1600
    finalizer_assistant_chars: int = 6000

    @classmethod
    def from_settings(cls, runtime: RuntimeSettings, *, max_turns: int) -> "LoopLimits":
        """Build the limits from configuration.

        Args:
            runtime: The resolved runtime settings group.
            max_turns: The role's normal turn budget.

        Returns:
            The limits for one run.
        """
        return cls(
            max_turns=max_turns,
            no_tool_nudges=runtime.no_tool_nudges,
            submit_guard_rejections=runtime.submit_guard_rejections,
            submit_validation_rejections=runtime.submit_validation_rejections,
            repeat_soft=runtime.repeat_call_soft,
            repeat_hard=runtime.repeat_call_hard,
            repeat_stop=runtime.repeat_call_stop,
            max_tool_calls_per_turn=max(1, runtime.max_tool_calls_per_turn),
            keep_reasoning_max_chars=runtime.keep_reasoning_max_chars,
            finalizer_rescues=runtime.finalizer_rescues,
            finalizer_min_content_chars=runtime.finalizer_min_content_chars,
            finalizer_tool_results=runtime.finalizer_tool_results,
            finalizer_tool_result_chars=runtime.finalizer_tool_result_chars,
            finalizer_assistant_chars=runtime.finalizer_assistant_chars,
        )

    @property
    def turn_budget(self) -> int:
        """Iterations the loop may take, including those spent on a rescue.

        A rescue does not consume the normal turn budget, so the ceiling has to leave
        room for it or requesting one after the last normal turn could not be answered.
        """
        return self.max_turns + max(0, self.finalizer_rescues) * max(
            1, self.submit_validation_rejections
        )


def truncate_text(text: str, limit: int) -> str:
    """Cut text to a limit, saying how much was removed.

    Args:
        text: The text to cut.
        limit: Maximum characters to keep. Zero or less returns the text unchanged.

    Returns:
        The text, with a marker appended when it was cut.
    """
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit] + f"\n[truncated {len(text) - limit} chars]"


def role_system_prompt(base: str, *, now: datetime | None = None) -> str:
    """Assemble a role's full system prompt.

    The timestamp is prepended because several role prompts reason about recency, and
    the tool-call protocol is appended because it is harness plumbing every role needs
    and no role prompt should have to restate.

    Args:
        base: The composed role prompt.
        now: Clock override, for tests.

    Returns:
        The system prompt as sent.
    """
    stamp = (now or datetime.now(UTC)).strftime("%Y-%m-%dT%H:%M:%SZ")
    return f"Current time (UTC): {stamp}\n\n{base}\n\n{control('tool_failure_guidance')}"


def no_tool_nudge_message(
    *,
    submit_tool: str,
    finish_reason: str | None,
    assistant_content: str,
    tool_call_truncated: bool,
    token_limited: bool,
) -> str:
    """The corrective for a turn that produced no usable tool call.

    Three wordings, because the three causes need different instructions: a cut-off call
    should be re-emitted smaller, a cut-off generation should be continued rather than
    restarted, and a plain prose turn just needs the protocol restated.

    Args:
        submit_tool: Name of the role's submit tool.
        finish_reason: The provider's finish reason.
        assistant_content: Content of the turn just taken.
        tool_call_truncated: Whether a tool call was cut off and discarded.
        token_limited: Whether the finish reason indicates the generation cap.

    Returns:
        The user-turn text to append.
    """
    if tool_call_truncated:
        return control("no_tool_truncated", SUBMIT_TOOL=submit_tool)
    if token_limited:
        note = (
            "The partial assistant generation is included immediately above."
            if assistant_content.strip()
            else "The previous assistant generation returned no visible content before truncation."
        )
        return control("no_tool_max_tokens", PREVIOUS_NOTE=note, SUBMIT_TOOL=submit_tool)
    return control("no_tool", SUBMIT_TOOL=submit_tool)


def repeat_breaker_message(
    *,
    tools: Sequence[str],
    count: int,
    submit_tool: str,
    alternatives: Sequence[str] = (),
    strong: bool = False,
) -> str:
    """The corrective for a call that keeps returning the same result.

    The strong tier names other tools but leaves every tool available: it is a redirect,
    not a forced submission, because a model pushed straight to submit gives up an
    investigation that a different query might still have completed.

    Args:
        tools: Names of the repeated tools.
        count: How many identical results have come back.
        submit_tool: Name of the role's submit tool.
        alternatives: Tools not yet exhausted, named in the strong tier only.
        strong: Whether to use the stronger wording.

    Returns:
        The user-turn text to append.
    """
    names = ", ".join(tools)
    if not strong:
        return control(
            "repeat_breaker", TOOL_NAMES=names, COUNT=str(count), SUBMIT_TOOL=submit_tool
        )
    listed = ", ".join(f"`{name}`" for name in alternatives)
    sentence = (
        f" Other tools are available: {listed} — use one of them to make progress."
        if alternatives
        else ""
    )
    return control(
        "repeat_breaker_strong",
        TOOL_NAMES=names,
        COUNT=str(count),
        ALTERNATIVES=sentence,
        SUBMIT_TOOL=submit_tool,
    )


def mixed_submit_message(submit_tool: str) -> str:
    """The rejection for a submission emitted alongside other calls.

    Args:
        submit_tool: Name of the role's submit tool.

    Returns:
        The rejection text, used both as the tool result and as the corrective.
    """
    return control("mixed_submit", SUBMIT_TOOL=submit_tool)


def finalizer_rescue_message(*, submit_tool: str, reason: str) -> str:
    """The message that opens a submit-only rescue.

    Args:
        submit_tool: Name of the role's submit tool.
        reason: Why the loop stalled.

    Returns:
        The user-turn text to append.
    """
    return control("finalizer_rescue", REASON=reason, SUBMIT_TOOL=submit_tool)


def compact_rescue_context(
    *,
    initial_messages: Sequence[Message],
    transcript: Sequence[Message],
    submit_tool: str,
    reason: str,
    limits: LoopLimits,
) -> list[Message]:
    """Rebuild the transcript as a short, submit-only brief.

    A stalled run is usually stalled because the transcript has grown past what the
    model can hold in view. Replaying it would reproduce the stall, so the rescue sends
    the task, the last few tool results and the role's own most recent conclusion, and
    asks for the submission alone.

    Args:
        initial_messages: The messages the run opened with.
        transcript: The live transcript.
        submit_tool: Name of the role's submit tool.
        reason: Why the loop stalled.
        limits: The run's thresholds.

    Returns:
        A short message list to send in place of the transcript.
    """
    opening: list[Message] = [
        {
            "role": str(m.get("role", "user")),
            "content": truncate_text(str(m.get("content", "")), 8000),
        }
        for m in list(initial_messages)[-2:]
    ]

    kept = max(0, limits.finalizer_tool_results)
    results = [m for m in transcript if m.get("role") == "tool"]
    results = results[-kept:] if kept else []

    conclusion = ""
    for message in reversed(transcript):
        if message.get("role") == "assistant" and str(message.get("content") or "").strip():
            conclusion = str(message["content"])
            break

    sections = [
        control("finalizer_context_header", REASON=reason),
        control("finalizer_context_required", SUBMIT_TOOL=submit_tool),
    ]
    if results:
        sections.append(control("finalizer_recent_results"))
        sections.extend(
            f"[tool result {i}]\n"
            f"{truncate_text(str(m.get('content', '')), limits.finalizer_tool_result_chars)}"
            for i, m in enumerate(results, start=1)
        )
    if conclusion:
        sections.append(
            control(
                "finalizer_latest_conclusion",
                CONCLUSION=truncate_text(conclusion, limits.finalizer_assistant_chars),
            )
        )
    sections.append(control("finalizer_call_now", SUBMIT_TOOL=submit_tool))

    return [*opening, {"role": "user", "content": "\n\n".join(sections)}]


def repair_submission(arguments: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Repair only the narrow, telemetry-visible schema slips worth repairing.

    Exactly one: a confidence outside its range is clipped. The alternative is to hand
    it back, which costs one of three shared submission retries and, three times over,
    ends the stage — so a model writing 1.4 instead of 0.9 loses the whole cell, and the
    loss is indistinguishable in the results from a model that had nothing to say.

    The repair is recorded rather than silent, so the number a run reports is never
    quietly different from the number the model produced.

    Args:
        arguments: The submitted arguments.

    Returns:
        The arguments to validate, and a list of the repairs applied.
    """
    if "confidence" not in arguments:
        return dict(arguments), []
    raw = arguments["confidence"]
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return dict(arguments), []
    if not math.isfinite(value) or 0.0 <= value <= 1.0:
        return dict(arguments), []
    clipped = max(0.0, min(1.0, value))
    repaired = dict(arguments)
    repaired["confidence"] = clipped
    return repaired, [
        {"field": "confidence", "original": raw, "repaired": clipped, "reason": "clipped_to_0_1"}
    ]


@dataclass(frozen=True, slots=True)
class RepeatDecision:
    """Which tools have stalled, and how hard to push back."""

    soft: tuple[str, ...] = ()
    strong: tuple[str, ...] = ()
    stop: tuple[str, ...] = ()
    count: int = 0
    """The highest repeat count among tools that reached at least the soft threshold."""

    @property
    def repeated(self) -> tuple[str, ...]:
        """Every tool at or past the soft threshold.

        Returns:
            The union of the three tiers, sorted.
        """
        return tuple(sorted({*self.soft, *self.strong, *self.stop}))


class RepeatedCallTracker:
    """Counts consecutive identical (call, result) pairs.

    The count advances only when a call returns *the same result as last time*. A call
    repeated with a different result is progress, however slow, and resets to one.

    The signature excludes ``rationale``. The prototype included it, and since every
    non-submit tool schema carries a required rationale, a model that reissued the same
    call while rewording one sentence of its reasoning produced a new signature every
    time and never tripped the breaker. The breaker is the primary defence against the
    spiral it exists to catch, so it keyed on the one field guaranteed to vary.
    """

    def __init__(self, limits: LoopLimits) -> None:
        """Store the thresholds and start with no history.

        Args:
            limits: The run's thresholds.
        """
        self._limits = limits
        self._seen: dict[str, tuple[str, int]] = {}

    @staticmethod
    def signature(call: ToolCall) -> str:
        """Identify a call by name and arguments, ignoring the rationale.

        Args:
            call: The call to identify.

        Returns:
            A stable signature string.
        """
        arguments = {k: v for k, v in call.arguments.items() if k != RATIONALE_ARG}
        return f"{call.name}|{json.dumps(arguments, sort_keys=True, default=str)}"

    def observe(self, outcomes: Sequence[tuple[ToolCall, str]]) -> RepeatDecision:
        """Record a turn's results and report which tools have stalled.

        Args:
            outcomes: The turn's calls paired with what each returned.

        Returns:
            The tiers tripped by this turn.
        """
        soft: set[str] = set()
        strong: set[str] = set()
        stop: set[str] = set()
        highest = 0

        for call, content in outcomes:
            signature = self.signature(call)
            digest = hashlib.sha256(content.encode("utf-8", "replace")).hexdigest()
            previous = self._seen.get(signature)
            count = previous[1] + 1 if previous is not None and previous[0] == digest else 1
            self._seen[signature] = (digest, count)

            if count >= self._limits.repeat_stop:
                stop.add(call.name)
            elif count >= self._limits.repeat_hard:
                strong.add(call.name)
            elif count >= self._limits.repeat_soft:
                soft.add(call.name)
            if count >= self._limits.repeat_soft:
                highest = max(highest, count)

        return RepeatDecision(
            soft=tuple(sorted(soft)),
            strong=tuple(sorted(strong)),
            stop=tuple(sorted(stop)),
            count=highest,
        )


@dataclass(slots=True)
class RescueTranscript:
    """The compact message list a rescue turn actually sends.

    Modelled as an object rather than a second list beside the transcript because the
    two must stay consistent in a way that scattered append sites did not guarantee: in
    the prototype an assistant message went into the rescue context whenever it carried
    tool calls, while the matching results went only to the main transcript. A rescue
    turn on which the model called a non-submit tool therefore built a context holding a
    tool call with no result, which an OpenAI-compatible server rejects outright.

    Appending assistant messages and results through one object keeps that pairing an
    invariant instead of a convention.
    """

    messages: list[Message] = field(default_factory=list)
    _pending: bool = False

    def append_assistant(self, message: Message) -> None:
        """Add an assistant turn, remembering whether it awaits results.

        Args:
            message: The serialised assistant message.
        """
        self.messages.append(message)
        self._pending = bool(message.get("tool_calls"))

    def append_results(self, results: Sequence[Message]) -> None:
        """Add the results for the assistant turn just appended.

        Args:
            results: Serialised tool-result messages.
        """
        if not self._pending:
            return
        self.messages.extend(results)
        self._pending = False

    def append_user(self, content: str) -> None:
        """Add a corrective user turn.

        Args:
            content: The message text.
        """
        self.messages.append({"role": "user", "content": content})

    def unanswered(self) -> bool:
        """Report whether a tool call is still awaiting its result.

        Returns:
            True when sending now would leave a call unanswered.
        """
        return self._pending
