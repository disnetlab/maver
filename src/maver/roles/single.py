"""The single-agent configuration: one agent, one prompt, no memory.

The experiment's minimal baseline, and the null hypothesis against the multi-agent,
memory-augmented pipeline. One goal-only prompt, the full read and write toolset, and no
retrieval. It answers the benchmark's own two-task contract as two passes of the shared
loop over one continuous transcript:

    TASK 1 (diagnosis)  -> submit_diagnosis
    TASK 2 (mitigation) -> submit_mitigation

Both phases are one role run and one run row, so the arm's cost is one role's worth and
not two. The two phases write into the same diagnosis and remediation state slots the
pipeline uses, as their own unstructured artifacts, which is what lets persistence,
outcome resolution and an adapter's episode driver work against either configuration
without branching.

Continuity is the point of the arm. Phase two opens on the phase-one transcript rather
than on a fresh one, so the agent keeps everything it learned while diagnosing; a
baseline that re-read the namespace from scratch would be a different and weaker system
than the one being compared against.
"""

import json
import logging
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from maver.contracts.artifacts import SingleDiagnosis, SingleResult
from maver.llm.types import Message, ToolSpec
from maver.prompts.loader import compose_role_prompt, read_runtime, render
from maver.roles.base import RoleContext, RoleOutcome, RoleSpec, StateDelta
from maver.roles.loop import run_until_submit
from maver.roles.policy import (
    LoopLimits,
    close_dangling_tool_calls,
    overlay_guard,
    role_system_prompt,
)
from maver.roles.prompting import (
    alert_json,
    kubectl_route,
    overlay_context,
    overlay_system,
)
from maver.roles.registry import role
from maver.roles.trace_summary import build_trace_summary
from maver.tools.registry import BoundTool

__all__ = ["SINGLE"]

_log = logging.getLogger(__name__)


def _hide(tools: list[ToolSpec], name: str) -> list[ToolSpec]:
    """Withhold the finalizer that is not terminal in this phase.

    Only one of the two ends a phase. Offering both lets the model short-circuit the
    sequence by submitting the mitigation it has not performed while still diagnosing.
    """
    return [tool for tool in tools if tool.name != name]


def _submitted_diagnosis(messages: list[Message]) -> tuple[SingleDiagnosis, list[Message]] | None:
    """Read the first phase's answer back out of the transcript it was submitted in.

    A continuation re-enters the phase it paused in, so the diagnosis is not produced
    again in this run, and the state slot it would occupy is filled only when the role
    returns — which a run that paused has not done. The transcript is therefore the only
    record of it the role is handed, and it is a complete one.

    Args:
        messages: The transcript the run was paused in.

    Returns:
        The diagnosis and the prefix of the transcript that produced it, or None when
        the transcript carries no accepted diagnosis.
    """
    for index in range(len(messages) - 1, -1, -1):
        for tool_call in messages[index].get("tool_calls") or []:
            function = tool_call.get("function") or {}
            if function.get("name") != "submit_diagnosis":
                continue
            try:
                arguments = json.loads(function.get("arguments") or "{}")
                return SingleDiagnosis.model_validate(arguments), messages[: index + 1]
            except Exception:
                # A rejected submission is still in the transcript. Keep scanning back:
                # only the accepted one ends a phase, and it is the last of its kind.
                continue
    return None


def _for_stage(dispatch: Mapping[str, BoundTool], stage: str) -> dict[str, BoundTool]:
    """Rebind every tool in a dispatch map to the phase about to run.

    Args:
        dispatch: The role's resolved dispatch.
        stage: The phase label the loop reports to the approval gate.

    Returns:
        The same tools, each bound to a context naming that phase.
    """
    return {
        name: replace(bound, ctx=replace(bound.ctx, role=stage)) for name, bound in dispatch.items()
    }


async def _run(ctx: RoleContext) -> RoleOutcome:
    system = role_system_prompt(
        overlay_system(
            # No retrieval in this arm by construction: the allowlist carries no memory
            # tools, so the composed prompt must not prescribe them either.
            compose_role_prompt(
                ctx.spec.prompt,
                retrieval_enabled=False,
                override_dir=ctx.settings.prompts.override_dir,
                kubectl_route=kubectl_route(ctx),
                block_overrides=ctx.overlay.prompt_blocks if ctx.overlay is not None else None,
            ),
            ctx.overlay,
        )
    )
    limits = LoopLimits.from_settings(
        ctx.settings.runtime,
        max_turns=ctx.settings.pipeline.max_turns,
        overlay=ctx.overlay,
    )
    shared: dict[str, Any] = {
        "model": ctx.completion,
        "system": system,
        "limits": limits,
        "session_id": ctx.session_id,
        "journal": ctx.journal,
        "middleware": ctx.middleware,
        "submit_guard": overlay_guard(None, ctx.overlay),
        "tool_choice_mode": ctx.profile.quirks.tool_choice,
        "keep_reasoning": ctx.profile.quirks.keep_reasoning,
        "serial_tool_calls": ctx.spec.serial_tool_calls,
    }

    # A continuation re-enters the phase that paused. Rebuilding phase one would re-ask
    # a question already answered, discard the result of the write the operator has just
    # approved, and let the agent reissue a mutation the cluster has taken.
    continuation = (
        _submitted_diagnosis(list(ctx.resume))
        if ctx.resume is not None and ctx.resume_stage == "single.mitigation"
        else None
    )
    if ctx.resume is not None and continuation is not None:
        diagnosis, diagnosis_transcript = continuation
        mitigation_opening: list[Message] = list(ctx.resume)
    else:
        opening = overlay_context(
            render(
                read_runtime("single", "diagnosis_task.txt"),
                {
                    "INCIDENT_CONTEXT": render(
                        read_runtime("single", "incident_context.txt"),
                        {"ALERT_JSON": alert_json(ctx.state.get("alert") or {})},
                    )
                },
            ),
            ctx.overlay,
        )
        diagnosis_run = await run_until_submit(
            role="single.diagnosis",
            dispatch=ctx.toolset.dispatch,
            tools=_hide(ctx.toolset.definitions, "submit_mitigation"),
            submit_tool="submit_diagnosis",
            messages=list(ctx.resume)
            if ctx.resume is not None
            else [{"role": "user", "content": opening}],
            submit_validator=SingleDiagnosis.model_validate,
            **shared,
        )
        if diagnosis_run.submission is None:
            return RoleOutcome(
                artifact=None,
                transcript=diagnosis_run.transcript,
                error=(
                    f"single-agent diagnosis failed: {diagnosis_run.reason or diagnosis_run.status}"
                ),
            )

        diagnosis = SingleDiagnosis.model_validate(diagnosis_run.submission)
        diagnosis_transcript = diagnosis_run.transcript
        # Continue the same transcript so the agent keeps its diagnosis context. Its
        # last turn is the submit call, which nothing answered, and a server rejects a
        # request whose tool call has no reply — so the dangling call is closed first.
        mitigation_opening = [
            *close_dangling_tool_calls(diagnosis_transcript),
            {"role": "user", "content": read_runtime("single", "mitigation_task.txt")},
        ]

    mitigation_run = await run_until_submit(
        role="single.mitigation",
        # The phase, not the role. This pass answers a separately graded question and
        # the gate is told which one: the write policy names `single.mitigation`, and
        # that is the label the loop hands it. A tool that screens a call by the stage
        # has to be looking at the same label, and the only one a tool can see is the
        # role its context was bound to — one role for both passes, so the writing pass
        # rebinds it. The diagnosis pass deliberately does not: its label is what says
        # it may not write.
        dispatch=_for_stage(ctx.toolset.dispatch, "single.mitigation"),
        tools=_hide(ctx.toolset.definitions, "submit_diagnosis"),
        submit_tool="submit_mitigation",
        messages=mitigation_opening,
        submit_validator=SingleResult.model_validate,
        **shared,
    )

    unfinalised: str | None = None
    if mitigation_run.submission is None:
        # No clean finalize, but the agent may already have applied changes. Record a
        # failure result and still fill the remediation slot, so the episode signals
        # mitigation and the benchmark grades the cluster state rather than the absence
        # of a submission.
        unfinalised = mitigation_run.reason or mitigation_run.status
        result = SingleResult(succeeded=False, summary=f"mitigation not finalized: {unfinalised}")
    else:
        result = SingleResult.model_validate(mitigation_run.submission)

    # Two summaries, one per graded phase, replacing the single one the wrapper would
    # otherwise write. Every other role produces one artifact and one account of itself;
    # this one answers two separately graded questions over one transcript, and a
    # summary that covered only the second would tell the offline learning pass what was
    # applied without saying what it was applied for.
    summaries = [
        *ctx.state.get("trace_summaries", []),
        build_trace_summary(
            "single.diagnosis", transcript=diagnosis_transcript, output=diagnosis
        ).model_dump(mode="json"),
        build_trace_summary(
            "single.mitigation",
            transcript=mitigation_run.transcript,
            output=result,
            error=unfinalised,
        ).model_dump(mode="json"),
    ]
    delta: StateDelta = {
        "root_cause": diagnosis,
        "execution_result": result,
        "trace_summaries": summaries,
    }
    return RoleOutcome(
        artifact=result,
        transcript=mitigation_run.transcript,
        error=None if mitigation_run.submission is not None else "mitigation not finalized",
        delta=delta,
    )


SINGLE = role(
    RoleSpec(
        name="single",
        submit_tool="submit_mitigation",
        submit_model=SingleResult,
        # Two slots, filled by the role itself, because this role produces two artifacts
        # and the wrapper's single-slot merge cannot express that.
        state_slot=None,
        prompt="single",
        run=_run,
        serial_tool_calls=True,
    )
)
"""The minimal baseline. One agent, the full toolset, no memory, two graded phases over
one transcript."""
