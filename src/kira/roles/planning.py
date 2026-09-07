"""The remediating role: a diagnosis in, one executable `Plan` out.

Two things make this role different. Its opening message carries the diagnosis it must
not revisit — a re-entry here means the plan failed, never the cause — and it holds a
submit guard, because the way this role fails quietly is an empty plan.

The retry budget is deliberately absent. The prototype spent it here, on entry, and
carried a `carry` dict into every return path so the count survived a failed run. In
Kira the node wrapper has already spent it and already cleared the stale artifacts
before the role runs, so the behaviour is the same and none of the arithmetic is here.
"""

import json
import logging
from typing import Any

from kira.contracts.artifacts import Plan
from kira.llm.types import Message
from kira.prompts.loader import compose_role_prompt, read_runtime, render
from kira.roles.base import RoleContext, RoleOutcome, RoleSpec
from kira.roles.loop import run_until_submit
from kira.roles.policy import LoopLimits, opening, overlay_guard, role_system_prompt
from kira.roles.prompting import (
    alert_json,
    memory_guidance,
    overlay_context,
    overlay_system,
    retrieval_enabled,
)
from kira.roles.registry import role

__all__ = ["PLANNING"]

_log = logging.getLogger(__name__)

_REPLAN_KINDS = frozenset({"replan_from_verify", "replan_from_execution"})


def _replan_block(prior_plans: list[dict[str, Any]]) -> str:
    """Build the re-plan task: the prior plan(s) could not be carried out.

    Mirrors RCA's reassessment block but keeps the root cause fixed: the failures here
    are infeasible or ambiguous steps and commands that could not run, not a wrong
    cause.
    """
    lines = read_runtime("planning", "replan_header.txt").splitlines()
    for i, attempt in enumerate(prior_plans, 1):
        plan = attempt.get("plan") or {}
        steps = plan.get("steps") or []
        actions = (
            "; ".join(f"{s.get('action', '?')} → {s.get('target', '?')}" for s in steps[:5])
            or "(no steps)"
        )
        lines.append(f"{i}. {actions}")
        note = attempt.get("execution_note")
        if note:
            lines.append(f"   Execution could not complete it: {note}")
    lines += ["", read_runtime("planning", "replan_footer.txt")]
    return "\n".join(lines)


def _submit_guard(
    args: dict[str, Any], tool_calls_made: int, messages: list[Message]
) -> str | None:
    """Reject an empty plan that is not an explicit halt.

    A `submit_plan` with no steps and `halt=false` is a silent give-up: Execution then
    skips with nothing applied and the stage fails with no usable signal. Force the
    model to either produce an actionable step or halt explicitly with a reason.

    It is a guard rather than a schema rule because a guard comes back to the model as a
    turn it can answer, where a validation error spends the generic repair budget.
    """
    steps = args.get("steps")
    if (not isinstance(steps, list) or len(steps) == 0) and not bool(args.get("halt")):
        return read_runtime("planning", "submit_empty.txt")
    return None


def _initial_message(ctx: RoleContext, root_cause: Any, *, replanning: bool) -> str:
    """Render the opening user turn: the alert, the fixed diagnosis, and the task."""
    task = (
        _replan_block(list(ctx.state.get("prior_plans") or []))
        if replanning
        else read_runtime("planning", "produce_plan.txt")
    )
    return render(
        read_runtime("planning", "initial.txt"),
        {
            # The playbook and ACE layers are gone; empty is what the prototype
            # rendered with both features disabled, and the template is byte-pinned.
            "PLAYBOOK_BLOCK": "",
            "PROCEDURAL_CONTEXT": "",
            "MEMORY_GUIDANCE": memory_guidance(retrieval_enabled(ctx)),
            "ALERT_JSON": alert_json(ctx.state.get("alert") or {}),
            "ROOT_CAUSE_JSON": json.dumps(root_cause.model_dump(mode="json"), indent=2),
            "TASK": task,
        },
    )


async def _run(ctx: RoleContext) -> RoleOutcome:
    root_cause = ctx.state.get("root_cause")
    if root_cause is None:
        # Unreachable through the graph, which ends on a missing diagnosis. Kept for the
        # paths that reach a role directly: the single-agent mode and a resumed state.
        return RoleOutcome(
            artifact=None, transcript=[], error="Planning skipped: no root cause from RCA"
        )

    replanning = ctx.handback is not None and ctx.handback.kind in _REPLAN_KINDS
    initial: list[Message] = [
        {
            "role": "user",
            "content": overlay_context(
                _initial_message(ctx, root_cause, replanning=replanning), ctx.overlay
            ),
        }
    ]

    result = await run_until_submit(
        model=ctx.completion,
        role=ctx.spec.name,
        system=role_system_prompt(
            overlay_system(
                compose_role_prompt(
                    ctx.spec.prompt,
                    retrieval_enabled=retrieval_enabled(ctx),
                    override_dir=ctx.settings.prompts.override_dir,
                ),
                ctx.overlay,
            )
        ),
        tools=ctx.toolset.definitions,
        dispatch=ctx.toolset.dispatch,
        submit_tool=ctx.spec.submit_tool,
        messages=opening(ctx, initial),
        limits=LoopLimits.from_settings(
            ctx.settings.runtime,
            max_turns=ctx.settings.pipeline.max_turns,
            overlay=ctx.overlay,
        ),
        session_id=ctx.session_id,
        journal=ctx.journal,
        middleware=ctx.middleware,
        tool_choice_mode=ctx.profile.quirks.tool_choice,
        keep_reasoning=ctx.profile.quirks.keep_reasoning,
        submit_validator=ctx.spec.submit_model.model_validate,
        submit_guard=overlay_guard(_submit_guard, ctx.overlay),
        serial_tool_calls=ctx.spec.serial_tool_calls,
    )

    artifact = (
        ctx.spec.submit_model.model_validate(result.submission)
        if result.submission is not None
        else None
    )
    return RoleOutcome(
        artifact=artifact,
        transcript=result.transcript,
        error=None if artifact is not None else (result.reason or result.status),
        metadata={"replan": replanning, "submit_guard": result.guard.metadata()},
    )


PLANNING = role(
    RoleSpec(
        name="planning",
        submit_tool="submit_plan",
        submit_model=Plan,
        state_slot="plan",
        prompt="planning",
        run=_run,
    )
)
"""The remediating role. Read-only cluster access plus retrieval, an empty-plan guard,
batches its tool calls, and halts the incident when it cannot submit."""
