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
normalizer fills only mechanical omissions using observed probe results or the
model's own submitted evidence. The verdict, guards and artifact schema remain authoritative.
"""

import logging
import re
from typing import Any, cast

from maver.contracts.artifacts import VerifyResult
from maver.llm.types import Message
from maver.prompts.loader import compose_role_prompt, read_runtime, render
from maver.roles.base import RoleContext, RoleOutcome, RoleSpec
from maver.roles.health_gate import HealthGate, sweep_namespace_health
from maver.roles.loop import SubmitGuard, run_until_submit
from maver.roles.policy import LoopLimits, opening, overlay_guard, role_system_prompt
from maver.roles.prompting import (
    alert_json,
    kubectl_route,
    overlay_context,
    overlay_system,
    retrieval_enabled,
)
from maver.roles.registry import role

__all__ = ["VERIFY"]

_log = logging.getLogger(__name__)


_HTTP_SUCCESS_PROBE = re.compile(
    r"\bHTTP\s+(?:2\d\d|3\d\d)\s+via\s+service/[a-z0-9-]+(?::\d+)?[^\s]*",
    re.IGNORECASE,
)


def _latest_successful_probe_status(messages: list[Message]) -> str | None:
    """Return the last successful Verify-only HTTP probe recorded in the transcript."""
    tool_names: dict[str, str] = {}
    successful: str | None = None
    for message in messages:
        if message.get("role") == "assistant":
            for call in message.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                function = call.get("function") or {}
                call_id = call.get("id")
                if isinstance(call_id, str) and isinstance(function, dict):
                    name = function.get("name")
                    if isinstance(name, str):
                        tool_names[call_id] = name
        elif (
            message.get("role") == "tool"
            and tool_names.get(cast(str, message.get("tool_call_id"))) == "probe_service"
        ):
            content = str(message.get("content") or "").strip()
            if content.startswith("error:"):
                continue
            match = _HTTP_SUCCESS_PROBE.search(content)
            if match:
                successful = match.group(0)
    return successful


def _submitted_evidence(args: dict[str, Any]) -> str:
    """Bounded restatement of the model's own evidence for an RCA hand-back."""
    parts = [str(args.get("summary") or "").strip()]
    checks = args.get("checks")
    if isinstance(checks, list):
        parts.extend(str(check).strip() for check in checks[:3] if str(check).strip())
    evidence = " ".join(part for part in parts if part)
    return evidence[:1800]


def _normalize_verify_submit_args(
    args: dict[str, Any], messages: list[Message]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Fill only mechanical Verify finalizer omissions from already-observed facts.

    The model retains ownership of the recovery verdict. A claimed recovery gets
    an entry-point field only when its transcript contains a successful HTTP
    `probe_service` result; a not-recovered verdict without a route uses the
    prompt's conservative RCA-reassessment route and repeats the model's own
    submitted evidence. Repairs are returned for durable audit by the loop.
    """
    normalized = dict(args)
    repairs: list[dict[str, Any]] = []
    recovered = bool(normalized.get("recovered")) and (
        str(normalized.get("verdict") or "").strip().lower() == "recovered"
    )
    if recovered and not str(normalized.get("entry_point_status") or "").strip():
        status = _latest_successful_probe_status(messages)
        if status:
            normalized["entry_point_status"] = status
            repairs.append(
                {
                    "field": "entry_point_status",
                    "source": "last_successful_probe_service_result",
                }
            )

    if str(normalized.get("verdict") or "").strip().lower() == "not_recovered":
        has_replan = bool(normalized.get("needs_replan"))
        has_reassessment = bool(normalized.get("needs_reassessment"))
        if not has_replan and not has_reassessment:
            normalized["needs_reassessment"] = True
            normalized["reassessment_reason"] = (
                "Verify submitted not_recovered. Evidence from its own checks: "
                f"{_submitted_evidence(normalized)}"
            )
            repairs.append(
                {
                    "field": "needs_reassessment,reassessment_reason",
                    "source": "not_recovered_default_to_rca_with_submitted_evidence",
                }
            )
        elif has_replan and not str(normalized.get("replan_reason") or "").strip():
            normalized["replan_reason"] = _submitted_evidence(normalized)
            repairs.append(
                {
                    "field": "replan_reason",
                    "source": "submitted_summary_and_checks",
                }
            )
        elif has_reassessment and not str(normalized.get("reassessment_reason") or "").strip():
            normalized["reassessment_reason"] = _submitted_evidence(normalized)
            repairs.append(
                {
                    "field": "reassessment_reason",
                    "source": "submitted_summary_and_checks",
                }
            )
    return normalized, repairs


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
                    block_overrides=ctx.overlay.prompt_blocks if ctx.overlay is not None else None,
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
        submit_args_normalizer=_normalize_verify_submit_args,
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
