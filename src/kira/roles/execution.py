"""The applying role: an approved plan in, one `ExecutionResult` out.

This is the only role that changes anything, which accounts for everything unusual
about it. Its tool calls run one at a time, because concurrent writes to one namespace
are not ordered. Its first call is forced to be a read, so the model cannot open on a
mutation. It holds the cluster-write lock for the length of its run. And its submit
guard is the strictest of any role's, because the way this role fails quietly is by
reporting a plan as applied without having touched the cluster.

Check-and-Set — re-read the target immediately before mutating, report the drift if the
precondition no longer holds, and stop the dependent steps — is instructed in the prompt
rather than enforced here. Four properties of this module are what make the instruction
followable: the forced opening read, the serial calls, the guard against a submit with
no commands, and the schema's demand that each step observation carry a before value, a
result and an after value.

`drift_detected` is a record field and not a control signal. It is written by the model
and rendered into the trace summary, and no routing predicate reads it; carrying it as a
control would imply a behaviour the system does not have.

Approval is not handled here. When a write parks on a human the gate raises, and this
role's part is to let it propagate: the orchestration layer catches it, checkpoints and
exits, and the resumed state re-enters the same router. The prototype instead imported
the graph, the outcome resolver and the post-incident hooks into this file in order to
drive the pipeline itself after a resume, which is why its approval path was the least
testable part of the system.
"""

import json
import logging
from typing import Any

from kira.contracts.artifacts import ExecutionResult, RootCause
from kira.llm.types import Message, ToolChoice
from kira.prompts.loader import compose_role_prompt, read_runtime, render
from kira.roles.base import RoleContext, RoleOutcome, RoleSpec
from kira.roles.loop import run_until_submit
from kira.roles.policy import LoopLimits, opening, role_system_prompt
from kira.roles.prompting import retrieval_enabled
from kira.roles.registry import role

__all__ = ["EXECUTION", "initial_execution_tool_choice", "requires_kubelet_recovery"]

_log = logging.getLogger(__name__)

_INITIAL_EXECUTION_TOOL = "exec_kubectl_cmd_safely"

_KUBELET_MARKERS = ("stopped", "not posting", "notready", "not ready", "nodestatusunknown")


def requires_kubelet_recovery(root_cause: Any) -> bool:
    """Say whether the diagnosis specifically identifies a stopped kubelet.

    A generic node failure can still be a storage, network or cloud-host problem, so
    those are not routed automatically. Both the node fault classification and concrete
    kubelet evidence are required before the run is directed at the constrained
    host-recovery capability.

    Args:
        root_cause: The diagnosis, or None.

    Returns:
        True when the evidence names a kubelet and says it is down.
    """
    if not isinstance(root_cause, RootCause):
        return False
    if root_cause.fault_type.strip().lower() != "node_failure":
        return False
    evidence = " ".join([root_cause.summary, *root_cause.evidence]).lower()
    return "kubelet" in evidence and any(marker in evidence for marker in _KUBELET_MARKERS)


def initial_execution_tool_choice(root_cause: Any, tool_names: set[str]) -> ToolChoice | None:
    """Choose what the first turn is forced to call.

    Ordinarily a Kubernetes read, so the model cannot open with a mutation. A stopped
    kubelet is the exception: cordoning, draining or patching Kubernetes objects cannot
    restore a node whose kubelet is down, so that case routes to the least-privileged
    repair the system has. Only the tool selection is forced; the model still supplies
    an allowed node and the documented command, and still verifies the result.

    Args:
        root_cause: The diagnosis, or None.
        tool_names: What this run actually resolved, which may exclude either tool.

    Returns:
        A forced choice, or None when neither tool is available.
    """
    if requires_kubelet_recovery(root_cause) and "ssh_exec" in tool_names:
        return _forced("ssh_exec")
    if _INITIAL_EXECUTION_TOOL in tool_names:
        return _forced(_INITIAL_EXECUTION_TOOL)
    return None


def _forced(name: str) -> ToolChoice:
    return {"type": "function", "function": {"name": name}}


def _submit_guard(
    args: dict[str, Any], tool_calls_made: int, messages: list[Message]
) -> str | None:
    """Reject a submit that finalises the remediation before any command has run.

    The approved plan reaches this role only after diagnosis and planning succeeded, so
    a submit on the very first turn — with zero reads and zero mutations — discards that
    work and files an empty mitigation. Small models occasionally emit the terminal tool
    while their own reasoning says to read the resource first. One corrective turn
    usually recovers a real attempt, and the loop caps the retries so this cannot wedge
    the stage.
    """
    if tool_calls_made == 0:
        return read_runtime("execution", "submit_without_command.txt")
    if not str(args.get("final_state_summary") or "").strip():
        return read_runtime("execution", "submit_without_summary.txt")
    steps = args.get("steps")
    if not isinstance(steps, list) or len(steps) == 0:
        return read_runtime("execution", "submit_without_steps.txt")
    return None


def _empty_plan_result() -> ExecutionResult:
    """What a plan with no steps resolves to, without spending a model call."""
    return ExecutionResult(
        succeeded=False,
        steps=[],
        final_state_summary="Execution skipped: approved plan contains no remediation steps.",
        drift_detected=False,
        confidence=0.0,
    )


def _initial_message(plan: Any, root_cause: Any) -> str:
    """Render the opening user turn: the diagnosis, any directive, and the plan."""
    root_cause_block = (
        render(
            read_runtime("execution", "root_cause_block.txt"),
            {"ROOT_CAUSE_JSON": json.dumps(root_cause.model_dump(mode="json"), indent=2)},
        )
        + "\n\n"
        if root_cause is not None
        else ""
    )
    directive = (
        read_runtime("execution", "kubelet_recovery.txt") + "\n\n"
        if requires_kubelet_recovery(root_cause)
        else ""
    )
    return render(
        read_runtime("execution", "initial.txt"),
        {
            # The playbook layer is gone; empty is what the prototype rendered with the
            # feature disabled, and the template is byte-pinned.
            "PLAYBOOK_BLOCK": "",
            "ROOT_CAUSE_BLOCK": root_cause_block,
            "RECOVERY_DIRECTIVE": directive,
            "PLAN_JSON": json.dumps(plan.model_dump(mode="json"), indent=2),
        },
    )


async def _run(ctx: RoleContext) -> RoleOutcome:
    plan = ctx.state.get("plan")
    if plan is None:
        # Unreachable through the graph, which ends on a missing plan. Kept for the
        # paths that reach a role directly: a resumed state, and a caller driving one
        # role on its own.
        return RoleOutcome(
            artifact=None, transcript=[], error="Execution skipped: no plan from Planning"
        )

    if not plan.steps:
        # No model call, and no lock: there is nothing to serialise against.
        return RoleOutcome(
            artifact=_empty_plan_result(),
            transcript=[],
            error="execution skipped: approved plan contains no remediation steps",
        )

    root_cause = ctx.state.get("root_cause")
    initial: list[Message] = [{"role": "user", "content": _initial_message(plan, root_cause)}]

    # Acquired after the guards above, so a run with nothing to apply does not queue
    # behind a remediation that is still running.
    async with ctx.write_lock:
        result = await run_until_submit(
            model=ctx.completion,
            role=ctx.spec.name,
            system=role_system_prompt(
                compose_role_prompt(
                    ctx.spec.prompt,
                    retrieval_enabled=retrieval_enabled(ctx),
                    override_dir=ctx.settings.prompts.override_dir,
                )
            ),
            tools=ctx.toolset.definitions,
            dispatch=ctx.toolset.dispatch,
            submit_tool=ctx.spec.submit_tool,
            messages=opening(ctx, initial),
            limits=LoopLimits.from_settings(
                ctx.settings.runtime, max_turns=ctx.settings.pipeline.max_turns
            ),
            session_id=ctx.session_id,
            journal=ctx.journal,
            middleware=ctx.middleware,
            tool_choice_mode=ctx.profile.quirks.tool_choice,
            keep_reasoning=ctx.profile.quirks.keep_reasoning,
            submit_validator=ctx.spec.submit_model.model_validate,
            submit_guard=_submit_guard,
            serial_tool_calls=ctx.spec.serial_tool_calls,
            initial_tool_choice=initial_execution_tool_choice(
                root_cause, {tool.name for tool in ctx.toolset.definitions}
            ),
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
        metadata={"plan_steps": len(plan.steps)},
    )


EXECUTION = role(
    RoleSpec(
        name="execution",
        submit_tool="submit_execution_result",
        submit_model=ExecutionResult,
        state_slot="execution_result",
        prompt="execution",
        run=_run,
        serial_tool_calls=True,
    )
)
"""The applying role. The only writer, serialised against itself, opening on a forced
read and guarded against submitting a remediation it never attempted."""
