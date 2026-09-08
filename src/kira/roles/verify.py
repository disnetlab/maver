"""The checking role: did the incident actually end?

Runs only after a clean apply, and only when verification is enabled. It re-evaluates
the firing alerts or the degraded SLI and exercises the affected request path; pods
being Ready is not the bar. It then owns the routing decision — end, hand back to
planning for an insufficient plan, or hand back to diagnosis for a wrong cause — and it
is read-only throughout: it observes and judges, it does not repair.

**It is not told what was diagnosed or what was applied.** Its dominant failure was
chain-anchoring: it validated the causal story instead of testing the system, so when
the diagnosis was wrong the real fault sat outside the chain being checked and could not
be seen. Asking for independence in prose did not produce it — instructing the model to
probe the request path first left the probe's position in the run statistically
unchanged. Withholding the diagnosis makes the independence structural, because there is
no chain available to anchor to. The cost is routing precision: without the diagnosis
this role cannot tell "cause right, fix incomplete" from "cause wrong", so a persisting
symptom routes to re-diagnosis. That was the dominant route regardless.

The verdict's own consistency — recovery needs an entry-point observation and a check
that exercises the request path, a not-recovered verdict must name a direction back, the
two hand-back flags are exclusive — is enforced by the schema rather than here. The
prototype filled several of those fields in code when the model left them blank; the
schema rejects the submission instead, so the model answers for its own verdict.
"""

import logging
from typing import Any

from kira.contracts.artifacts import VerifyResult
from kira.llm.types import Message
from kira.prompts.loader import compose_role_prompt, read_runtime, render
from kira.roles.base import RoleContext, RoleOutcome, RoleSpec
from kira.roles.health_gate import HealthGate, sweep_namespace_health
from kira.roles.loop import SubmitGuard, run_until_submit
from kira.roles.policy import LoopLimits, opening, overlay_guard, role_system_prompt
from kira.roles.prompting import (
    alert_json,
    kubectl_route,
    overlay_context,
    overlay_system,
    retrieval_enabled,
)
from kira.roles.registry import role

__all__ = ["VERIFY"]

_log = logging.getLogger(__name__)


def _build_submit_guard(gate: HealthGate) -> SubmitGuard:
    """Reject a verdict submitted before any end-to-end check has run.

    Also enforces the namespace health gate. That lives here rather than in the schema
    because the model gets a guard rejection back as a turn it can revise: the point is
    not to fail the submit but to make recovery unavailable while the namespace is
    measurably broken, so the pipeline routes instead of closing.

    Args:
        gate: The namespace sweep taken before the run started.

    Returns:
        The guard, bound to that sweep.
    """

    def guard(args: dict[str, Any], tool_calls_made: int, messages: list[Message]) -> str | None:
        if tool_calls_made == 0:
            return read_runtime("verify", "submit_without_check.txt")
        if not str(args.get("summary") or "").strip():
            return read_runtime("verify", "submit_without_summary.txt")
        checks = args.get("checks")
        if not isinstance(checks, list) or len(checks) == 0:
            return read_runtime("verify", "submit_without_checks.txt")
        # Either field claiming recovery is enough to trigger the last two checks. The
        # schema forces the two into agreement afterwards; here the cheaper reading is
        # the safer one, because it catches the claim before validation sees it.
        claims_recovered = (
            bool(args.get("recovered"))
            or str(args.get("verdict") or "").strip().lower() == "recovered"
        )
        if claims_recovered and not str(args.get("entry_point_status") or "").strip():
            return read_runtime("verify", "submit_without_entry_point.txt")
        if claims_recovered and gate.blocks_recovered:
            return gate.rejection_text()
        return None

    return guard


def _initial_message(ctx: RoleContext, gate: HealthGate) -> str:
    """Render the opening user turn: the alert, and nothing about the remediation."""
    body = render(
        read_runtime("verify", "context_blind.txt"),
        {
            # The ACE layer is gone. Empty is what the prototype rendered for it
            # outside the environment that supplied it.
            "PROCEDURAL_CONTEXT": "",
            "ALERT_JSON": alert_json(ctx.state.get("alert") or {}),
        },
    )
    return f"{body}\n\n{gate.context_text()}" if gate.checked else body


async def _run(ctx: RoleContext) -> RoleOutcome:
    # Swept before the model sees anything, so the namespace's real state is established
    # independently of any causal chain rather than through one.
    gate = HealthGate(checked=False)
    if ctx.settings.verify.health_gate:
        namespace = str((ctx.state.get("alert") or {}).get("namespace") or "").strip()
        if namespace:
            gate = await sweep_namespace_health(ctx.tool_context, namespace)

    initial: list[Message] = [
        {"role": "user", "content": overlay_context(_initial_message(ctx, gate), ctx.overlay)}
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
                    kubectl_route=kubectl_route(ctx),
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
        submit_guard=overlay_guard(_build_submit_guard(gate), ctx.overlay),
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
        # Persisted because the eval container is recreated per cell: without this, a
        # run with no blocks cannot be told apart from a gate that never managed to look.
        metadata={
            "health_gate": ctx.settings.verify.health_gate,
            "gate": gate.audit(),
            "submit_guard": result.guard.metadata(),
        },
    )


VERIFY = role(
    RoleSpec(
        name="verify",
        submit_tool="submit_verify_result",
        submit_model=VerifyResult,
        state_slot="verify_result",
        prompt="verify",
        run=_run,
        serial_tool_calls=True,
        # A check that could not run must not invalidate a remediation that applied
        # cleanly. This role returns no result and the incident is then decided on the
        # remediation's own claim, which is the one role where that asymmetry is right.
        failure_sets_error=False,
    )
)
"""The checking role. Read-only end to end, blind to the diagnosis it is checking, and
the owner of the hand-back decision."""
