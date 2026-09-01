"""The bounded model/tool loop every role runs.

One role, one submit tool, a budget of turns. The loop calls the model, dispatches the
tools it asks for, feeds the results back, and ends when the model calls its submit tool
as the only call in a turn.

Most of the code here is not that sequence. It is what keeps an open-weight model
inside the protocol: a turn that produced prose instead of a call, a call cut off by the
generation cap, twenty calls in one turn, a submission emitted alongside investigation
tools, or the same call repeated with the same result until the budget is gone. Each of
those has a specific corrective, and none of them can be a middleware hook because each
one reads counters that span turns. The concerns that *are* per-turn — authority checks,
tracing, prompt shaping — go through the hook stack instead.
"""

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from kira.audit.journal import Journal
from kira.contracts.events import InterventionEvent, StageEvent, SubmissionEvent, ToolCallEvent
from kira.llm.client import carry_reasoning_forward, serialise_assistant
from kira.llm.sampling import is_token_limit_finish
from kira.llm.types import (
    Completion,
    Message,
    ModelRequest,
    ModelResponse,
    ToolCall,
    ToolChoice,
    ToolResult,
    ToolSpec,
    serialise_tool_results,
)
from kira.middleware.authority import ApprovalPending
from kira.middleware.base import MiddlewareStack, ToolRequest, TurnContext
from kira.roles.policy import (
    RATIONALE_ARG,
    LoopLimits,
    RepeatedCallTracker,
    RescueTranscript,
    compact_rescue_context,
    control,
    finalizer_rescue_message,
    mixed_submit_message,
    no_tool_nudge_message,
    repair_submission,
    repeat_breaker_message,
)
from kira.tools.errors import call_failed

__all__ = ["LoopResult", "SubmitGuard", "ToolFn", "run_until_submit"]

_log = logging.getLogger(__name__)

ToolFn = Callable[..., Awaitable[Any]]
"""A dispatchable tool. Called with the model's arguments, minus the rationale."""

SubmitGuard = Callable[[dict[str, Any], int, list[Message]], Any]
"""Stage-specific policy consulted before a submission is validated.

Returns None to accept, or text to hand back. Text starting with ``note:`` is passed to
the model as material to weigh rather than framed as an error, because a guard that
returns content the model should consider is not reporting a fault.
"""

_GUARD_CONTENT_MARK = "note:"


@dataclass(frozen=True, slots=True)
class LoopResult:
    """How a role's run ended.

    ``submission`` is set only on the success path. Callers branch on ``status`` rather
    than on the presence of a submission, because the ways a run can fail differ in what
    the caller should do next: a validation failure is worth reporting to the operator,
    while an exhausted turn budget usually is not.
    """

    status: str
    """One of: submitted, no_tool_calls, submit_validation_failed,
    repeated_call_no_progress, max_turns_exceeded."""
    transcript: list[Message]
    submission: dict[str, Any] | None = None
    text: str | None = None
    """The model's prose, when it ended the run by writing instead of calling."""
    reason: str | None = None
    tool_calls_made: int = 0
    turns_used: int = 0

    @property
    def submitted(self) -> bool:
        """Whether the role produced a valid submission.

        Returns:
            True when the run ended through the submit tool.
        """
        return self.status == "submitted"


@dataclass(slots=True)
class _State:
    """The counters the controls read. Deliberately not visible to hooks."""

    nudges: int = 0
    tool_calls_made: int = 0
    normal_turns_used: int = 0
    guard_rejections: int = 0
    submit_rejections: int = 0
    rescues_used: int = 0
    rescue_turns_left: int = 0
    rescue: RescueTranscript | None = None
    last_assistant_content: str = ""
    transcript: list[Message] = field(default_factory=list)


def _forced(name: str) -> dict[str, Any]:
    return {"type": "function", "function": {"name": name}}


def _turn_tool_choice(
    *,
    mode: str,
    rescue: bool,
    submit_tool: str,
    first_turn: bool,
    initial: ToolChoice | None,
) -> ToolChoice | None:
    """Decide what to tell the provider about tool use this turn.

    Not every endpoint accepts a forced choice, so the mode is configurable and
    ``omit`` sends nothing at all.
    """
    if mode == "omit":
        return None
    if mode == "auto":
        return "auto"
    if rescue:
        return _forced(submit_tool)
    if first_turn and initial is not None:
        return initial
    return "required"


async def _maybe_await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


def _hand_back(
    state: "_State",
    call: ToolCall,
    message: str,
    *,
    note: bool = False,
    rescue: RescueTranscript | None = None,
) -> None:
    """Return a rejected submission to the model as that call's tool result.

    Args:
        state: The run's state.
        call: The rejected submit call.
        message: What to tell the model.
        note: Frame the text as material rather than as an error.
        rescue: The rescue context, when this turn was a rescue turn.
    """
    content = message if note else f"error: {message}"
    rendered = serialise_tool_results(
        [ToolResult(tool_call_id=call.id, name=call.name, content=content)]
    )
    state.transcript.extend(rendered)
    if rescue is not None:
        rescue.append_results(rendered)


async def run_until_submit(
    *,
    model: Completion,
    role: str,
    system: str,
    tools: Sequence[ToolSpec],
    dispatch: Mapping[str, ToolFn],
    submit_tool: str,
    messages: Sequence[Message],
    limits: LoopLimits | None = None,
    session_id: str = "",
    journal: Journal | None = None,
    middleware: MiddlewareStack | None = None,
    submit_validator: Callable[[dict[str, Any]], Any] | None = None,
    submit_guard: SubmitGuard | None = None,
    serial_tool_calls: bool = False,
    keep_reasoning: bool = False,
    initial_tool_choice: ToolChoice | None = None,
    tool_choice_mode: str = "required",
) -> LoopResult:
    """Run one role until it submits, gives up, or runs out of turns.

    Args:
        model: The completion port.
        role: The role's name, recorded as the caller of every model call.
        system: The composed system prompt.
        tools: The tools this role may call, including its submit tool.
        dispatch: Tool name to implementation.
        submit_tool: The tool whose arguments are this role's output.
        messages: The opening transcript.
        limits: Thresholds for this run. Defaults to the published values.
        journal: Where audit events go.
        session_id: The incident session, for hooks and the audit log.
        middleware: Cross-cutting hooks.
        submit_validator: Validates a submission, raising to reject it.
        submit_guard: Stage policy consulted before validation.
        serial_tool_calls: Run a turn's calls one at a time. Required wherever calls
            mutate a cluster, since concurrent writes to one namespace are not ordered.
        keep_reasoning: Carry a bounded tail of the reasoning channel between turns.
        initial_tool_choice: What to force on the first turn, if anything.
        tool_choice_mode: ``required``, ``auto`` or ``omit``.

    Returns:
        How the run ended, with the transcript.

    Raises:
        ModelCallError: When a model call fails and retrying did not help.
    """
    limits = limits or LoopLimits()
    stack = middleware or MiddlewareStack()
    log = journal or Journal(session_id=session_id, stage=role)

    state = _State(transcript=list(messages))
    opening = list(messages)
    submit_only = [t for t in tools if t.name == submit_tool]
    repeats = RepeatedCallTracker(limits)

    log.record(StageEvent(stage=role, event="entered"))

    def finish(
        status: str,
        *,
        submission: dict[str, Any] | None = None,
        text: str | None = None,
        reason: str | None = None,
    ) -> LoopResult:
        log.record(
            StageEvent(
                stage=role,
                event="exited" if status == "submitted" else "failed",
                detail=reason or status,
            )
        )
        return LoopResult(
            status=status,
            transcript=state.transcript,
            submission=submission,
            text=text,
            reason=reason,
            tool_calls_made=state.tool_calls_made,
            turns_used=state.normal_turns_used,
        )

    def request_rescue(reason: str) -> bool:
        """Open a submit-only recovery turn, if one is warranted and still available.

        Refused when there is nothing to submit *from*: a run that has produced neither
        a tool result nor a substantial written conclusion has no findings to convert,
        and rescuing it would only manufacture a confident-looking empty answer.
        """
        has_evidence = (
            state.tool_calls_made > 0
            or len(state.last_assistant_content) >= limits.finalizer_min_content_chars
        )
        if (
            state.rescues_used >= limits.finalizer_rescues
            or not submit_only
            or not has_evidence
            or (submit_guard is not None and state.tool_calls_made == 0)
        ):
            return False

        state.rescues_used += 1
        state.rescue_turns_left = max(1, limits.submit_validation_rejections)
        state.rescue = RescueTranscript(
            compact_rescue_context(
                initial_messages=opening,
                transcript=state.transcript,
                submit_tool=submit_tool,
                reason=reason,
                limits=limits,
            )
        )
        state.nudges = 0
        state.transcript.append(
            {
                "role": "user",
                "content": finalizer_rescue_message(submit_tool=submit_tool, reason=reason),
            }
        )
        log.record(InterventionEvent(control="finalizer_rescue", action="rescued", detail=reason))
        return True

    for turn_index in range(limits.turn_budget):
        rescue_turn = state.rescue_turns_left > 0
        if rescue_turn:
            state.rescue_turns_left -= 1
        else:
            if state.normal_turns_used >= limits.max_turns:
                if request_rescue("max_turns_exceeded"):
                    continue
                log.record(
                    InterventionEvent(
                        control="turn_limit", action="ended_run", detail="max_turns_exceeded"
                    )
                )
                return finish("max_turns_exceeded", reason="max_turns_exceeded")
            state.normal_turns_used += 1

        ctx = TurnContext(role=role, session_id=session_id, turn=turn_index + 1, caller=role)
        rescue_ctx = state.rescue
        request = ModelRequest(
            system=system,
            messages=tuple(rescue_ctx.messages if rescue_turn and rescue_ctx else state.transcript),
            tools=tuple(submit_only if rescue_turn else tools),
            tool_choice=_turn_tool_choice(
                mode=tool_choice_mode,
                rescue=rescue_turn,
                submit_tool=submit_tool,
                first_turn=turn_index == 0,
                initial=initial_tool_choice,
            ),
            lite_reasoning=rescue_turn,
        )
        request = stack.before_model(ctx, request)

        async def call(req: ModelRequest, _caller: str = role) -> ModelResponse:
            return await model.complete(req, caller=_caller)

        response = await stack.run_model(ctx, request, call)

        correction = stack.after_model(ctx, response)
        if correction is not None:
            log.record(
                InterventionEvent(
                    control="no_tool_nudge",
                    action="ended_run" if correction.end_run else "warned",
                    detail=f"{correction.control}: {correction.detail}",
                )
            )
            if correction.message is not None:
                state.transcript.append(correction.message)
            if correction.end_run:
                return finish("no_tool_calls", reason=correction.detail)
            continue

        # An oversized batch is treated as a malformed generation: nothing is executed
        # and nothing is serialised. Running a subset would make the outcome depend on
        # whatever order the provider happened to emit the calls in.
        if len(response.tool_calls) > limits.max_tool_calls_per_turn:
            state.transcript.append(
                {"role": "assistant", "content": control("oversized_batch_discarded")}
            )
            state.transcript.append(
                {
                    "role": "user",
                    "content": control(
                        "oversized_batch_correction",
                        CALL_COUNT=str(len(response.tool_calls)),
                        LIMIT=str(limits.max_tool_calls_per_turn),
                    ),
                }
            )
            log.record(
                InterventionEvent(
                    control="generation_cap",
                    action="refused",
                    detail=f"{len(response.tool_calls)} calls in one turn, limit {limits.max_tool_calls_per_turn}",
                )
            )
            continue

        token_limited = is_token_limit_finish(response.finish_reason)
        truncated_call = bool(response.tool_calls) and token_limited

        assistant = serialise_assistant(response)
        if keep_reasoning and assistant.get("tool_calls"):
            assistant = carry_reasoning_forward(
                assistant, response.reasoning, limits.keep_reasoning_max_chars
            )
        state.transcript.append(assistant)
        state.last_assistant_content = str(assistant.get("content") or response.text or "").strip()
        if rescue_turn and rescue_ctx is not None:
            rescue_ctx.append_assistant(assistant)

        # A truncated call was dropped during serialisation, so this turn produced no
        # usable call even though the response carried one.
        if not response.tool_calls or truncated_call:
            if state.nudges < limits.no_tool_nudges and not rescue_turn:
                state.nudges += 1
                nudge = no_tool_nudge_message(
                    submit_tool=submit_tool,
                    finish_reason=response.finish_reason,
                    assistant_content=state.last_assistant_content,
                    tool_call_truncated=truncated_call,
                    token_limited=token_limited,
                )
                state.transcript.append({"role": "user", "content": nudge})
                log.record(
                    InterventionEvent(
                        control="no_tool_nudge",
                        action="warned",
                        detail=f"nudge {state.nudges} of {limits.no_tool_nudges}",
                    )
                )
                continue
            if rescue_turn and rescue_ctx is not None and state.rescue_turns_left > 0:
                retry = control("finalizer_retry", SUBMIT_TOOL=submit_tool)
                state.transcript.append({"role": "user", "content": retry})
                rescue_ctx.append_user(retry)
                continue
            reason = "truncated_tool_call" if truncated_call else "no_tool_calls"
            if not rescue_turn and request_rescue(reason):
                continue
            return finish("no_tool_calls", text=response.text, reason=reason)

        state.nudges = 0

        submit_calls = [c for c in response.tool_calls if c.name == submit_tool]

        async def execute(call_: ToolCall) -> ToolResult:
            """Dispatch one tool call through the hook stack."""
            if call_.parse_error:
                return ToolResult(
                    tool_call_id=call_.id,
                    name=call_.name,
                    content=f"error: invalid {call_.name} arguments: {call_.parse_error}",
                )

            async def dispatch_one(req: ToolRequest) -> ToolResult:
                fn = dispatch.get(req.name)
                if fn is None:
                    content = f"error: unknown tool '{req.name}'"
                else:
                    try:
                        content = str(await fn(**req.arguments))
                    except Exception as exc:  # surfaced to the model as a tool error
                        content = f"error: {type(exc).__name__}: {exc}"
                return ToolResult(tool_call_id=req.call_id, name=req.name, content=content)

            # The rationale is the model's carried-forward working trace, injected into
            # every non-submit schema. It stays in the transcript, so the trace survives
            # to the next turn, but it is not a parameter any tool accepts.
            arguments = {k: v for k, v in call_.arguments.items() if k != RATIONALE_ARG}
            try:
                result = await stack.run_tool(
                    ToolRequest(
                        call_id=call_.id,
                        name=call_.name,
                        arguments=arguments,
                        role=role,
                        session_id=session_id,
                        rationale=call_.arguments.get(RATIONALE_ARG),
                    ),
                    dispatch_one,
                )
            except ApprovalPending as pending:
                # The transcript is the loop's, and the run has to be restartable from
                # where it stopped rather than from its opening turn.
                pending.messages = list(state.transcript)
                raise
            log.record(
                ToolCallEvent(
                    tool=call_.name,
                    arguments=arguments,
                    rationale=call_.arguments.get(RATIONALE_ARG),
                    succeeded=not call_failed(result.content),
                    result_chars=len(result.content),
                )
            )
            return result

        async def execute_all(calls: Sequence[ToolCall]) -> list[ToolResult]:
            if serial_tool_calls:
                return [await execute(c) for c in calls]
            return list(await asyncio.gather(*(execute(c) for c in calls)))

        # A submission is terminal, so it must be the only call in its turn. Emitted
        # alongside investigation tools it is premature by construction: the model has
        # not seen those results yet. Two submissions in one turn land here too.
        if submit_calls and len(response.tool_calls) != 1:
            rejection = mixed_submit_message(submit_tool)
            others = [c for c in response.tool_calls if c.name != submit_tool]
            executed = await execute_all(others)
            by_id = {r.tool_call_id: r for r in executed}
            results = [
                by_id.get(
                    c.id,
                    ToolResult(tool_call_id=c.id, name=c.name, content=f"error: {rejection}"),
                )
                for c in response.tool_calls
            ]
            state.tool_calls_made += len(executed)
            rendered = serialise_tool_results(results)
            state.transcript.extend(rendered)
            if rescue_turn and rescue_ctx is not None:
                rescue_ctx.append_results(rendered)
            state.transcript.append({"role": "user", "content": rejection})
            for c in submit_calls:
                log.record(
                    SubmissionEvent(
                        artifact=c.name,
                        accepted=False,
                        attempt=state.submit_rejections + 1,
                        validation_error="mixed_tool_calls",
                    )
                )
            continue

        if submit_calls:
            call_ = submit_calls[0]

            handed_back = rescue_ctx if rescue_turn else None

            if call_.parse_error:
                state.submit_rejections += 1
                _hand_back(
                    state,
                    call_,
                    f"invalid {call_.name} arguments: {call_.parse_error}",
                    rescue=handed_back,
                )
                log.record(
                    SubmissionEvent(
                        artifact=call_.name,
                        accepted=False,
                        attempt=state.submit_rejections,
                        validation_error=call_.parse_error,
                    )
                )
                if state.submit_rejections >= limits.submit_validation_rejections:
                    return finish(
                        "submit_validation_failed",
                        reason=f"submit_validation_failed: {call_.parse_error}",
                    )
                continue

            # The guard runs before schema validation on purpose: it expresses stage
            # policy, and letting it answer first means an incomplete-but-recognisable
            # submission can be turned into useful feedback instead of consuming the
            # generic validation-retry budget.
            if submit_guard is not None and state.guard_rejections < limits.submit_guard_rejections:
                verdict = await _maybe_await(
                    submit_guard(dict(call_.arguments), state.tool_calls_made, state.transcript)
                )
                if verdict:
                    state.guard_rejections += 1
                    is_note = str(verdict).startswith(_GUARD_CONTENT_MARK)
                    _hand_back(state, call_, str(verdict), note=is_note, rescue=handed_back)
                    log.record(
                        SubmissionEvent(
                            artifact=call_.name,
                            accepted=False,
                            attempt=state.guard_rejections,
                            validation_error=str(verdict),
                        )
                    )
                    continue

            # Repair the narrow slips that are worth repairing before validating. A
            # confidence of 1.4 is a formatting mistake, not a failure to diagnose, and
            # handing it back spends one of three shared retries.
            arguments, repairs = repair_submission(call_.arguments)
            if repairs:
                log.record(
                    SubmissionEvent(
                        artifact=call_.name,
                        accepted=True,
                        attempt=state.submit_rejections + 1,
                        validation_error=f"repaired: {repairs}",
                    )
                )

            if submit_validator is not None:
                try:
                    submit_validator(arguments)
                except Exception as exc:  # handed back to the model for repair
                    state.submit_rejections += 1
                    detail = f"{type(exc).__name__}: {exc}"
                    _hand_back(
                        state,
                        call_,
                        f"invalid {call_.name} arguments: {detail}",
                        rescue=handed_back,
                    )
                    log.record(
                        SubmissionEvent(
                            artifact=call_.name,
                            accepted=False,
                            attempt=state.submit_rejections,
                            validation_error=detail,
                        )
                    )
                    if state.submit_rejections >= limits.submit_validation_rejections:
                        return finish(
                            "submit_validation_failed",
                            reason=f"submit_validation_failed: {detail}",
                        )
                    continue

            state.transcript.extend(
                serialise_tool_results(
                    [ToolResult(tool_call_id=call_.id, name=call_.name, content="submitted")]
                )
            )
            log.record(
                SubmissionEvent(
                    artifact=call_.name, accepted=True, attempt=state.submit_rejections + 1
                )
            )
            return finish("submitted", submission=arguments)

        results = await execute_all(response.tool_calls)
        state.tool_calls_made += len(results)
        rendered = serialise_tool_results(results)
        state.transcript.extend(rendered)
        if rescue_turn and rescue_ctx is not None:
            rescue_ctx.append_results(rendered)

        decision = repeats.observe(
            [(c, r.content) for c, r in zip(response.tool_calls, results, strict=True)]
        )
        if decision.stop:
            names = ", ".join(decision.stop)
            log.record(
                InterventionEvent(control="repeat_breaker", action="ended_run", detail=names)
            )
            if request_rescue("repeated_call_no_progress"):
                continue
            return finish("repeated_call_no_progress", reason=f"repeated_call_no_progress: {names}")
        if decision.strong or decision.soft:
            strong = bool(decision.strong)
            tripped = decision.strong if strong else decision.soft
            alternatives = [
                t.name for t in tools if t.name != submit_tool and t.name not in decision.repeated
            ]
            message = repeat_breaker_message(
                tools=tripped,
                count=decision.count,
                submit_tool=submit_tool,
                alternatives=alternatives,
                strong=strong,
            )
            state.transcript.append({"role": "user", "content": message})
            log.record(
                InterventionEvent(
                    control="repeat_breaker",
                    action="refused" if strong else "warned",
                    detail=f"{', '.join(tripped)} repeated {decision.count} times",
                )
            )

    log.record(
        InterventionEvent(control="turn_limit", action="ended_run", detail="max_turns_exceeded")
    )
    return finish("max_turns_exceeded", reason="max_turns_exceeded")
