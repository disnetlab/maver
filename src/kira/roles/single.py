"""The single-agent configuration: one agent, one prompt, no memory.

The experiment's minimal baseline, and the null hypothesis against the multi-agent,
memory-augmented pipeline. One goal-only prompt, the full read and write toolset, and no
retrieval. It answers the benchmark's own two-task contract as two passes of the shared
loop over one continuous transcript:

    TASK 1 (diagnosis)  -> submit_diagnosis   -> the `rca` milestone
    TASK 2 (mitigation) -> submit_mitigation  -> the `execution` milestone

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

import logging
from typing import Any

from kira.contracts.artifacts import SingleDiagnosis, SingleResult
from kira.llm.types import ToolSpec
from kira.prompts.loader import compose_role_prompt, read_runtime, render
from kira.roles.base import RoleContext, RoleOutcome, RoleSpec, StateDelta
from kira.roles.loop import run_until_submit
from kira.roles.policy import LoopLimits, close_dangling_tool_calls, role_system_prompt
from kira.roles.prompting import alert_json
from kira.roles.registry import role
from kira.roles.trace_summary import build_trace_summary

__all__ = ["SINGLE"]

_log = logging.getLogger(__name__)


def _hide(tools: list[ToolSpec], name: str) -> list[ToolSpec]:
    """Withhold the finalizer that is not terminal in this phase.

    Only one of the two ends a phase. Offering both lets the model short-circuit the
    sequence by submitting the mitigation it has not performed while still diagnosing.
    """
    return [tool for tool in tools if tool.name != name]


async def _announce(ctx: RoleContext, stage: str, delta: StateDelta) -> None:
    if ctx.milestone is not None:
        await ctx.milestone(stage, delta)


async def _run(ctx: RoleContext) -> RoleOutcome:
    system = role_system_prompt(
        # No retrieval in this arm by construction: the allowlist carries no memory
        # tools, so the composed prompt must not prescribe them either.
        compose_role_prompt(ctx.spec.prompt, retrieval_enabled=False)
    )
    limits = LoopLimits.from_settings(
        ctx.settings.runtime, max_turns=ctx.settings.pipeline.max_turns
    )
    shared: dict[str, Any] = {
        "model": ctx.completion,
        "system": system,
        "dispatch": ctx.toolset.dispatch,
        "limits": limits,
        "session_id": ctx.session_id,
        "journal": ctx.journal,
        "middleware": ctx.middleware,
        "tool_choice_mode": ctx.profile.quirks.tool_choice,
        "keep_reasoning": ctx.profile.quirks.keep_reasoning,
        "serial_tool_calls": ctx.spec.serial_tool_calls,
    }

    opening = render(
        read_runtime("single", "diagnosis_task.txt"),
        {
            "INCIDENT_CONTEXT": render(
                read_runtime("single", "incident_context.txt"),
                {"ALERT_JSON": alert_json(ctx.state.get("alert") or {})},
            )
        },
    )
    diagnosis_run = await run_until_submit(
        role="single.diagnosis",
        tools=_hide(ctx.toolset.definitions, "submit_mitigation"),
        submit_tool="submit_diagnosis",
        messages=[{"role": "user", "content": opening}],
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
    await _announce(ctx, "rca", {"root_cause": diagnosis})

    # Continue the same transcript so the agent keeps its diagnosis context. Its last
    # turn is the submit call, which nothing answered, and a server rejects a request
    # whose tool call has no reply — so the dangling call is closed before the replay.
    mitigation_run = await run_until_submit(
        role="single.mitigation",
        tools=_hide(ctx.toolset.definitions, "submit_diagnosis"),
        submit_tool="submit_mitigation",
        messages=[
            *close_dangling_tool_calls(diagnosis_run.transcript),
            {"role": "user", "content": read_runtime("single", "mitigation_task.txt")},
        ],
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
            "single.diagnosis", transcript=diagnosis_run.transcript, output=diagnosis
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
    await _announce(ctx, "execution", delta)
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
