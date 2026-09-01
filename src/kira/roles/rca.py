"""The diagnosing role: read-only telemetry in, one `RootCause` out.

What makes this role different from the others is the message it opens with, and there
are three things in it:

* a namespace snapshot, collected in code and prepended, so that localization does not
  depend on the model remembering to enumerate the namespace;
* a task block that is either the incident or, on a hand-back, a re-diagnosis: a
  three-way discrimination between an insufficient fix, a real-but-unrelated fault and
  a wrong cause;
* the memory line, whose marker name is shared with a different block in the system
  prompt. :func:`~kira.roles.prompting.memory_guidance` says which is which.

Everything else a run needs — the run row, the audit events, the trace summary, the
retry budget, the state merge, the hand-back application — belongs to the node wrapper
above, so none of it appears here. The role never halts by returning early either:
``halt`` is a field on a submitted ``RootCause``, so halting is a submission and the
incident is still answered downstream.
"""

import logging
import time
from typing import Any

from kira.contracts.artifacts import RootCause
from kira.contracts.events import ToolCallEvent
from kira.llm.types import Message
from kira.prompts.loader import compose_role_prompt, read_runtime, render
from kira.roles.base import RoleContext, RoleOutcome, RoleSpec
from kira.roles.loop import run_until_submit
from kira.roles.policy import LoopLimits, close_dangling_tool_calls, role_system_prompt
from kira.roles.prompting import alert_json, memory_guidance, retrieval_enabled
from kira.roles.registry import role
from kira.tools.snapshot import namespace_snapshot

__all__ = ["RCA", "build_initial_message"]

_log = logging.getLogger(__name__)

_SNAPSHOT_TOOL = "namespace_snapshot"


def _reassessment_block(alert: dict[str, Any], prior_attempts: list[dict[str, Any]]) -> str:
    """Build the re-diagnosis task: prior diagnosis, what was fixed, then dig deeper."""
    lines = read_runtime("rca", "reassessment_header.txt").splitlines()
    for i, attempt in enumerate(prior_attempts, 1):
        rc = attempt.get("root_cause") or {}
        lines.append(
            f"{i}. Fault `{rc.get('fault_type', '?')}` on "
            f"`{rc.get('affected_resource', '?')}` — {rc.get('summary', '')}".rstrip()
        )
        if attempt.get("remediation"):
            lines.append(f"   What was fixed: {attempt['remediation']}")
        if attempt.get("execution_note"):
            lines.append(f"   Execution's note: {attempt['execution_note']}")
    footer = render(
        read_runtime("rca", "reassessment_footer.txt"), {"ALERT_JSON": alert_json(alert)}
    )
    lines += ["", *footer.splitlines()]
    return "\n".join(lines)


def _task_block(ctx: RoleContext, *, reassessing: bool) -> str:
    """The incident task, or the re-diagnosis task on a hand-back."""
    alert = ctx.state.get("alert") or {}
    if reassessing:
        return _reassessment_block(alert, list(ctx.state.get("prior_attempts") or []))
    return render(read_runtime("rca", "incident_task.txt"), {"ALERT_JSON": alert_json(alert)})


def _record_snapshot_failure(ctx: RoleContext, namespace: str, *, chars: int, ms: int) -> None:
    """Record a snapshot that produced nothing, so its absence is visible in the log.

    The prototype swallowed every failure and returned an empty block, which is how the
    snapshot came to be missing from roughly 93% of runs with nobody noticing. The event
    has no field for the reason, so that goes to the log; the event carries the fact.
    """
    ctx.journal.record(
        ToolCallEvent(
            tool=_SNAPSHOT_TOOL,
            arguments={"namespace": namespace},
            succeeded=False,
            result_chars=chars,
            duration_ms=ms,
        )
    )


async def _snapshot_block(ctx: RoleContext) -> str:
    """Auto-collected namespace snapshot prepended to the RCA task.

    Forced enumeration: the snapshot puts every candidate component (sibling workloads,
    Service/targetPort bindings, ConfigMap/Secret refs, recent changes, node pressure)
    in front of the model so localization no longer depends on it remembering to look.
    Best-effort — a failure never breaks the diagnosis, which still has the
    `namespace_snapshot` tool — but never silent.
    """
    if not ctx.settings.pipeline.auto_snapshot:
        return ""
    namespace = str((ctx.state.get("alert") or {}).get("namespace") or "").strip()
    if not namespace:
        return ""

    started = time.perf_counter()
    try:
        snapshot = await namespace_snapshot(ctx.tool_context, namespace)
    except Exception as exc:  # never let snapshotting break the diagnosis
        _log.warning("namespace snapshot failed for ns=%s: %s", namespace, exc, exc_info=exc)
        _record_snapshot_failure(
            ctx, namespace, chars=0, ms=round((time.perf_counter() - started) * 1000)
        )
        return ""
    elapsed_ms = round((time.perf_counter() - started) * 1000)

    if not snapshot or snapshot.startswith("error:"):
        _log.warning(
            "namespace snapshot returned nothing usable for ns=%s: %s", namespace, snapshot
        )
        _record_snapshot_failure(ctx, namespace, chars=len(snapshot), ms=elapsed_ms)
        return ""
    return render(read_runtime("rca", "namespace_snapshot.txt"), {"SNAPSHOT": snapshot})


async def build_initial_message(
    ctx: RoleContext, *, reassessing: bool = False, assignment: str = ""
) -> str:
    """Render the diagnosing role's opening user turn.

    Six markers are supplied because the template declares six and rendering is strict
    in both directions. Two of them are always empty: ``PLAYBOOK_BLOCK`` and
    ``PROCEDURAL_CONTEXT`` belong to the playbook and ACE layers this system does not
    have, and empty is exactly what the prototype rendered with those features off.
    ``ASSIGNMENT_BLOCK`` is empty here and filled by the second pass, which owns the
    assignment text and is the only caller that passes it.

    Args:
        ctx: The role run's context.
        reassessing: Whether this is a re-diagnosis after a verification hand-back.
        assignment: The second pass's brief, positioned after the snapshot and before
            the task.

    Returns:
        The message content, byte-identical to what the prototype rendered.
    """
    snapshot = await _snapshot_block(ctx)
    return render(
        read_runtime("rca", "initial.txt"),
        {
            "PLAYBOOK_BLOCK": "",
            "PROCEDURAL_CONTEXT": "",
            "MEMORY_GUIDANCE": memory_guidance(retrieval_enabled(ctx)),
            "SNAPSHOT_BLOCK": f"{snapshot}\n\n" if snapshot else "",
            "ASSIGNMENT_BLOCK": f"{assignment}\n" if assignment else "",
            "TASK_BLOCK": _task_block(ctx, reassessing=reassessing),
        },
    )


def _resolve_reassess_mode(ctx: RoleContext, prior: list[Message]) -> tuple[str, str | None]:
    """Decide between a fresh re-diagnosis and a talkback resume, and say why.

    ``talkback`` degrades to ``fresh`` rather than failing, because a re-entry with no
    usable transcript still has a diagnosis to redo.
    """
    mode = ctx.settings.pipeline.reassess_mode
    if mode != "talkback":
        return mode, None
    if not prior:
        return "fresh", "no_prior_transcript"
    size = sum(len(str(message.get("content") or "")) for message in prior)
    if size > ctx.settings.pipeline.talkback_max_chars:
        _log.warning(
            "talkback degraded to fresh — transcript %d chars exceeds %d (session=%s)",
            size,
            ctx.settings.pipeline.talkback_max_chars,
            ctx.session_id,
        )
        return "fresh", "transcript_too_large"
    return "talkback", None


def _talkback_messages(ctx: RoleContext, prior: list[Message]) -> list[Message]:
    """Resume the conversation that produced the superseded diagnosis.

    The prior transcript is the full message list the loop worked from — opening task,
    every assistant turn and tool result, and the submit call — so the model answers the
    critique with its own ruled-out hypotheses still in context rather than re-deriving
    them from a summary.
    """
    attempts = list(ctx.state.get("prior_attempts") or [])
    latest = attempts[-1] if attempts else {}
    verify = ctx.state.get("verify_result")
    checks = list(getattr(verify, "checks", None) or [])
    critique = render(
        read_runtime("rca", "talkback_critique.txt"),
        {
            "REMEDIATION_SUMMARY": latest.get("remediation") or "remediation details unavailable",
            "REASSESSMENT_REASON": (
                latest.get("execution_note") or "the incident symptom is still present"
            ),
            "VERIFY_CHECKS": (
                "\n".join(f"- {check}" for check in checks)
                if checks
                else "- (Verify recorded no individual checks)"
            ),
        },
    )
    return [*close_dangling_tool_calls(prior), {"role": "user", "content": critique}]


async def _run(ctx: RoleContext) -> RoleOutcome:
    reassessing = ctx.handback is not None and ctx.handback.kind == "reassessment"
    prior: list[Message] = []
    mode, fallback = "fresh", None
    if reassessing:
        prior = list((ctx.state.get("messages") or {}).get(ctx.spec.name) or [])
        mode, fallback = _resolve_reassess_mode(ctx, prior)

    if mode == "talkback":
        initial = _talkback_messages(ctx, prior)
    else:
        content = await build_initial_message(ctx, reassessing=reassessing)
        initial = [{"role": "user", "content": content}]

    result = await run_until_submit(
        model=ctx.completion,
        role=ctx.spec.name,
        system=role_system_prompt(
            compose_role_prompt(ctx.spec.prompt, retrieval_enabled=retrieval_enabled(ctx))
        ),
        tools=ctx.toolset.definitions,
        dispatch=ctx.toolset.dispatch,
        submit_tool=ctx.spec.submit_tool,
        messages=initial,
        limits=LoopLimits.from_settings(
            ctx.settings.runtime, max_turns=ctx.settings.pipeline.max_turns
        ),
        session_id=ctx.session_id,
        journal=ctx.journal,
        middleware=ctx.middleware,
        tool_choice_mode=ctx.profile.quirks.tool_choice,
        keep_reasoning=ctx.profile.quirks.keep_reasoning,
        submit_validator=ctx.spec.submit_model.model_validate,
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
        metadata={"reassess_mode": mode, "talkback_fallback": fallback},
    )


RCA = role(
    RoleSpec(
        name="rca",
        submit_tool="submit_root_cause",
        submit_model=RootCause,
        state_slot="root_cause",
        prompt="rca",
        run=_run,
    )
)
"""The diagnosing role. Read-only against the cluster, no submit guard, batches its
tool calls, and halts the incident when it cannot submit."""
