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
import json
import logging
import textwrap
from collections.abc import Awaitable, Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Any

from maver.adapters.base import SubmitGuard
from maver.audit.journal import Journal
from maver.contracts.events import InterventionEvent, StageEvent, SubmissionEvent
from maver.llm.client import carry_reasoning_forward, serialise_assistant
from maver.llm.sampling import is_token_limit_finish
from maver.llm.types import (
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
from maver.middleware.authority import ApprovalPending
from maver.middleware.base import MiddlewareStack, ToolRequest, TurnContext
from maver.roles.policy import (
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
from maver.tools.auditing import ToolCallRecorder
from maver.tools.errors import call_timed_out
from maver.tools.registry import RETRIEVAL_PROVENANCE, ToolOutput

__all__ = [
    "GuardRecord",
    "LoopResult",
    "SubmitArgsNormalizer",
    "SubmitGuard",
    "ToolFn",
    "run_until_submit",
]

_log = logging.getLogger(__name__)

ToolFn = Callable[..., Awaitable[Any]]
"""A dispatchable tool. Called with the model's arguments, minus the rationale."""

SubmitArgsNormalizer = Callable[
    [dict[str, Any], list[Message]], tuple[dict[str, Any], list[dict[str, Any]]]
]
"""A synchronous, deterministic repair using evidence already in the transcript.

The callback returns normalized arguments and an explicit description of each repair.
The loop audits both payloads and leaves the model's original transcript intact.
"""

# Re-exported from the adapter contract, which owns the type because an environment
# supplies one through its role overlay. Text starting with ``note:`` is passed to the
# model as material to weigh rather than framed as an error, because a guard that
# returns content the model should consider is not reporting a fault.

_GUARD_CONTENT_MARK = "note:"

_FINDING_WIDTH = 160
"""How much of a guard's rejection the record keeps. Long enough to identify the
finding, short enough that a run row is not a second copy of the prompt asset."""


def _finding(rejection: str) -> str:
    """Condense a guard's rejection into the one line the record keeps."""
    first = next((line.strip() for line in rejection.splitlines() if line.strip()), "")
    return textwrap.shorten(first, width=_FINDING_WIDTH, placeholder=" ...")


@dataclass(frozen=True, slots=True)
class GuardRecord:
    """What the submit guard did to one run.

    A record, not a control: nothing reads it to decide anything. A guard whose budget
    is spent accepts the next submission unconsulted, and that acceptance is otherwise
    indistinguishable from one the guard approved — same artifact, same submission
    event, same run status. So the run carries what happened to it, because a reader
    reconstructing the run has no other way to tell the two apart.

    Attributes:
        rejections: How many submissions the guard handed back.
        bypassed: Whether a submission was then accepted with the budget spent.
        finding: The guard's last finding, which the bypass overrode.
    """

    rejections: int = 0
    bypassed: bool = False
    finding: str | None = None

    def metadata(self) -> dict[str, Any]:
        """Build the run row's ``submit_guard`` entry.

        Returns:
            The counts, and the overridden finding when there was a bypass.
        """
        entry: dict[str, Any] = {"rejections": self.rejections, "bypassed": self.bypassed}
        if self.bypassed and self.finding:
            entry["finding"] = self.finding
        return entry


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
    guard: GuardRecord = field(default_factory=GuardRecord)
    """What the submit guard did, for the record. All zeros when the role runs without
    one, or when it ran and never refused."""

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
    guard_bypassed: bool = False
    guard_finding: str = ""
    submit_rejections: int = 0
    mixed_rejections: int = 0
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


def _normalize_submission(
    call: ToolCall,
    messages: list[Message],
    normalizer: SubmitArgsNormalizer,
    log: Journal,
) -> dict[str, Any]:
    """Audit role-local repairs before the unchanged guard and schema checks.

    A broken normalizer leaves the original payload available to those checks.
    Copies protect the transcript and raw audit payload even if a callback mutates
    its inputs before failing.
    """
    original = deepcopy(call.arguments)
    try:
        normalized, repairs = normalizer(deepcopy(original), deepcopy(messages))
        if not repairs:
            return dict(call.arguments)
        detail = json.dumps(
            {
                "call_id": call.id,
                "tool": call.name,
                "raw_arguments": original,
                "normalized_arguments": normalized,
                "repairs": repairs,
            },
            sort_keys=True,
        )
    except Exception as exc:
        _log.warning("submit normalizer failed for %s: %s", call.name, exc)
        log.record(
            InterventionEvent(
                control="submit_normalization",
                action="bypassed",
                detail=json.dumps(
                    {
                        "call_id": call.id,
                        "tool": call.name,
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                    sort_keys=True,
                ),
            )
        )
        return dict(call.arguments)
    log.record(InterventionEvent(control="submit_normalization", action="rescued", detail=detail))
    return normalized


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
    submit_args_normalizer: SubmitArgsNormalizer | None = None,
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
        submit_args_normalizer: Evidence-backed mechanical repairs before the guard;
            each repair records both payloads without rewriting the transcript.
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

    # A continuation opens on the transcript it was paused in, so the commands it
    # already ran are in front of the model. Counting from zero told the guards that
    # read this that nothing had been run, and Execution's answer to that is to instruct
    # the model to apply a mutation the cluster has already taken.
    state = _State(
        transcript=list(messages),
        tool_calls_made=sum(1 for message in messages if message.get("role") == "tool"),
    )
    if stack.agent_state is not None:
        stack.agent_state.messages = state.transcript
        # Pydantic validates by copying: keep the live transcript after validation.
        state.transcript = stack.agent_state.messages
    scratch: dict[str, Any] = {}
    opening = list(messages)
    submit_only = [t for t in tools if t.name == submit_tool]
    repeats = RepeatedCallTracker(limits)

    log.record(StageEvent(source="role_loop", stage=role, event="entered"))

    def finish(
        status: str,
        *,
        submission: dict[str, Any] | None = None,
        text: str | None = None,
        reason: str | None = None,
    ) -> LoopResult:
        log.record(
            StageEvent(
                source="role_loop",
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
            guard=GuardRecord(
                rejections=state.guard_rejections,
                bypassed=state.guard_bypassed,
                finding=state.guard_finding or None,
            ),
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
        context = compact_rescue_context(
            initial_messages=opening,
            transcript=state.transcript,
            submit_tool=submit_tool,
            reason=reason,
            limits=limits,
        )
        state.rescue = RescueTranscript(list(context))
        state.nudges = 0
        state.transcript.append(
            {
                "role": "user",
                "content": finalizer_rescue_message(submit_tool=submit_tool, reason=reason),
            }
        )
        # Recorded here because this is the one control that does not add to the
        # transcript but replaces it. The kickoff the run returns is not what the rescue
        # turn was sent, and the brief that was sent is built from the transcript, kept
        # beside it and discarded with the run — so a rescued submission, which is the
        # answer a whole cell is graded on, was made on messages nobody could read back.
        log.record(
            InterventionEvent(
                control="finalizer_rescue",
                action="rescued",
                detail=reason,
                context=[dict(message) for message in context],
            )
        )
        return True

    def note_repeats(calls: Sequence[ToolCall], results: Sequence[ToolResult]) -> LoopResult | None:
        """Score the calls a turn actually dispatched, and act on what the breaker says.

        Shared by both dispatching paths rather than written out on each. A turn that
        also carried a submission is still a turn that ran these calls, and scoring it
        on one path only let a model repeat one query until its whole budget was gone
        with the breaker never counting past one.

        Args:
            calls: The calls that were dispatched.
            results: What each returned, in the same order.

        Returns:
            The run's result when the breaker ends it, or None to take another turn.
        """
        decision = repeats.observe([(c, r.content) for c, r in zip(calls, results, strict=True)])
        if decision.stop:
            names = ", ".join(decision.stop)
            log.record(
                InterventionEvent(control="repeat_breaker", action="ended_run", detail=names)
            )
            if request_rescue("repeated_call_no_progress"):
                return None
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
        return None

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

        ctx = TurnContext(
            role=role,
            session_id=session_id,
            turn=turn_index + 1,
            caller=role,
            run_state=stack.run_state,
            agent_state=stack.agent_state,
            scratch=scratch,
        )
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
        additions = (
            stack.run_state.context_for(stack.agent_state)
            if stack.run_state is not None and stack.agent_state is not None
            else ()
        )
        if additions:
            context = [{"role": "user", "content": item.text} for item in additions]
            request = replace(request, messages=(*request.messages, *context))
            log.record(
                InterventionEvent(
                    control="middleware_context",
                    action="injected",
                    detail=", ".join(f"{a.source}:{a.id}" for a in additions),
                    context=context,
                )
            )

        async def call(req: ModelRequest, _caller: str = role) -> ModelResponse:
            return await model.complete(req, caller=_caller)

        response = await stack.run_model(ctx, request, call)
        if stack.run_state is not None and stack.agent_state is not None:
            stack.run_state.delivered(stack.agent_state, additions)

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
            """Record the attempt around the existing middleware and dispatch path."""
            arguments = {
                key: value for key, value in call_.arguments.items() if key != RATIONALE_ARG
            }
            recorder = ToolCallRecorder(
                log, call_.name, arguments, call_.id, call_.arguments.get(RATIONALE_ARG)
            )
            with recorder.active():
                try:
                    result = await execute_inner(call_)
                except ApprovalPending as pending:
                    recorder.finish("", status="paused", error=str(pending))
                    raise
                recorder.observation.retrieval = result.metadata.get(RETRIEVAL_PROVENANCE, [])
                recorder.finish(result.content)
                return result

        async def execute_inner(call_: ToolCall) -> ToolResult:
            """Dispatch one tool call through the hook stack."""
            if call_.parse_error:
                return ToolResult(
                    tool_call_id=call_.id,
                    name=call_.name,
                    content=f"error: invalid {call_.name} arguments: {call_.parse_error}",
                )

            async def dispatch_one(req: ToolRequest) -> ToolResult:
                fn = dispatch.get(req.name)
                returned: Any
                if fn is None:
                    returned = f"error: unknown tool '{req.name}'"
                else:
                    try:
                        returned = await fn(**req.arguments)
                    except ApprovalPending:
                        # A pause is not a failed call. Reported to the model as one, the
                        # run would carry on past the decision it is waiting for, and the
                        # request an operator answers would belong to nothing.
                        raise
                    except Exception as exc:  # surfaced to the model as a tool error
                        returned = f"error: {type(exc).__name__}: {exc}"
                provenance = list(returned.retrieval) if isinstance(returned, ToolOutput) else []
                return ToolResult(
                    tool_call_id=req.call_id,
                    name=req.name,
                    content=str(returned),
                    # The audit half of what the tool returned, which the transcript
                    # never carries: `serialise_tool_results` drops it on purpose.
                    metadata={RETRIEVAL_PROVENANCE: provenance} if provenance else {},
                )

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
            if call_timed_out(result.content):
                # A bound the harness imposed and enforced, which the tool-call event
                # records only as a failure like any other. Grouping interventions by
                # control is how a run is read back, and the archive's forty-seven
                # minute blocking delete is the reason an empty timeout bucket must not
                # be easy to mistake for a run in which nothing was ever killed.
                log.record(
                    InterventionEvent(
                        control="timeout",
                        action="refused",
                        detail=f"{call_.name}: {result.content.split(chr(10), 1)[0]}",
                    )
                )
            return result

        async def execute_all(calls: Sequence[ToolCall]) -> list[ToolResult]:
            pending: ApprovalPending | None = None
            results: list[ToolResult] = []
            if serial_tool_calls:
                for call_ in calls:
                    try:
                        results.append(await execute(call_))
                    except ApprovalPending as paused:
                        pending = paused
                        break
            else:
                settled = await asyncio.gather(*(execute(c) for c in calls), return_exceptions=True)
                for outcome in settled:
                    if isinstance(outcome, ToolResult):
                        results.append(outcome)
                    elif isinstance(outcome, ApprovalPending):
                        pending = pending or outcome
                    else:
                        raise outcome
            if pending is None:
                return results
            # A sibling of the paused call has already run to completion — neither
            # dispatch path unwinds what it already did — so whatever it wrote is
            # already written. Letting the pause propagate on its own left that call
            # unanswered in the transcript the resume replays from, which is both a
            # malformed request and an invitation for the role to reissue a write the
            # cluster has taken.
            state.transcript.extend(serialise_tool_results(results))
            pending.messages = list(state.transcript)
            raise pending

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
                # Counted on its own, and not against the schema-repair budget. The
                # submission was never read, so charging it there turned three attempts
                # at a fixable slip into one — where the archive charged nothing at all.
                # It still has to advance something, or every rejection in a spiral
                # reads as the first one.
                state.mixed_rejections += 1
                log.record(
                    SubmissionEvent(
                        source="role_loop",
                        artifact=c.name,
                        accepted=False,
                        attempt=state.mixed_rejections,
                        validation_error="mixed_tool_calls",
                    )
                )
            ended = note_repeats(others, executed)
            if ended is not None:
                return ended
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
                        source="role_loop",
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

            arguments = (
                _normalize_submission(call_, state.transcript, submit_args_normalizer, log)
                if submit_args_normalizer is not None
                else dict(call_.arguments)
            )

            # The guard runs before schema validation on purpose: it expresses stage
            # policy, and letting it answer first means an incomplete-but-recognisable
            # submission can be turned into useful feedback instead of consuming the
            # generic validation-retry budget.
            if submit_guard is not None:
                if state.guard_rejections >= limits.submit_guard_rejections:
                    # Taken anyway: a run whose guard will never be satisfied still has
                    # to end, and this is what every published measurement was taken
                    # under. The acceptance that follows is otherwise identical to a
                    # guard's approval, and the guard's dose is what an arm's gap is
                    # divided by — so the run says here, and on its row, which role was
                    # let through, after how many rejections, and over what finding.
                    state.guard_bypassed = True
                    log.record(
                        InterventionEvent(
                            control="submit_guard",
                            action="bypassed",
                            detail=(
                                f"{role}: guard budget spent after "
                                f"{state.guard_rejections} rejections; submission "
                                f"accepted unconsulted, overriding: {state.guard_finding}"
                            ),
                        )
                    )
                else:
                    verdict = await _maybe_await(
                        submit_guard(dict(arguments), state.tool_calls_made, state.transcript)
                    )
                    if verdict:
                        state.guard_rejections += 1
                        state.guard_finding = _finding(str(verdict))
                        is_note = str(verdict).startswith(_GUARD_CONTENT_MARK)
                        _hand_back(state, call_, str(verdict), note=is_note, rescue=handed_back)
                        log.record(
                            SubmissionEvent(
                                source="role_loop",
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
            arguments, repairs = repair_submission(arguments)
            if repairs:
                log.record(
                    SubmissionEvent(
                        source="role_loop",
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
                            source="role_loop",
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
                    source="role_loop",
                    artifact=call_.name,
                    accepted=True,
                    attempt=state.submit_rejections + 1,
                )
            )
            return finish("submitted", submission=arguments)

        results = await execute_all(response.tool_calls)
        state.tool_calls_made += len(results)
        rendered = serialise_tool_results(results)
        state.transcript.extend(rendered)
        if rescue_turn and rescue_ctx is not None:
            rescue_ctx.append_results(rendered)

        ended = note_repeats(response.tool_calls, results)
        if ended is not None:
            return ended

    log.record(
        InterventionEvent(control="turn_limit", action="ended_run", detail="max_turns_exceeded")
    )
    return finish("max_turns_exceeded", reason="max_turns_exceeded")
