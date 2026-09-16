"""The distilling role: one closed incident in, one reusable record out.

It is a role by the same contract as the others and it is deliberately not a node in
the graph. It runs from a scheduler job minutes after the incident closed, on a fresh
task, which is what the published description means by distilling outside the incident
path. Three things follow, and each is a real constraint rather than an implementation
detail: it reads a packet assembled from stored records instead of live state, its job
function has to be importable by name so a scheduler can restore it across a restart,
and its turn budget comes from the trigger that started it rather than from the
pipeline's.

One trigger ships. The prototype also had an operator survey and a free-form operator
instruction, whose only storage path was a mutable playbook this system does not have,
and which no published claim rests on. The store-write rule that kept them apart from
the incident corpus survives their removal, because it is the rule that keeps the corpus
clean: only a post-incident pass writes a per-session record, since that corpus is the
precedent the planning role retrieves from and a task answer would pollute it.

Writing the result is not this role's job. It returns a `LearningResult`; the projection
into evidence and lesson text, the lifecycle state it is stored under and the weight
retrieval gives it belong to the memory layer.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import yaml

from maver.contracts.artifacts import LearningResult
from maver.llm.types import Message
from maver.prompts.loader import compose_role_prompt, read_runtime, render
from maver.roles.base import RoleContext, RoleOutcome, RoleSpec
from maver.roles.loop import run_until_submit
from maver.roles.policy import LoopLimits, overlay_guard, role_system_prompt
from maver.roles.prompting import (
    kubectl_route,
    overlay_context,
    overlay_system,
    retrieval_enabled,
)
from maver.roles.registry import role

__all__ = ["LEARNING", "Trigger", "load_trigger"]

_log = logging.getLogger(__name__)

POST_INCIDENT = "post_incident"
_DEFAULT_MAX_TURNS = 30


@dataclass(frozen=True, slots=True)
class Trigger:
    """What started a learning pass, and how long it may run."""

    name: str
    max_turns: int


def load_trigger(name: str, directory: Path) -> Trigger:
    """Read one trigger's configuration.

    Args:
        name: The trigger to load.
        directory: The directory holding ``triggers.yaml``.

    Returns:
        The trigger.

    Raises:
        ValueError: When the trigger is unknown or disabled. Both are refused rather
            than defaulted, because a pass that ran under a trigger nobody configured
            would write records nobody asked for.
    """
    raw = yaml.safe_load((directory / "triggers.yaml").read_text(encoding="utf-8")) or {}
    entry = (raw.get("triggers") or {}).get(name)
    if entry is None:
        raise ValueError(f"unknown learning trigger {name!r}")
    if not entry.get("enabled", True):
        raise ValueError(f"learning trigger {name!r} is disabled")
    return Trigger(name, int(entry.get("max_turns", _DEFAULT_MAX_TURNS)))


def _system_prompt(ctx: RoleContext, *, verify_live: bool) -> str:
    """Compose the base prompt, the trigger's half, and the live re-check when it runs.

    The re-check is appended only when the environment is expected to still be up and no
    external evaluation is present. An external source of truth supersedes a live
    self-check, and where one exists the environment has usually been torn down already,
    so the guidance would only spend turns on reads that cannot succeed.
    """
    parts = [
        compose_role_prompt(
            ctx.spec.prompt,
            retrieval_enabled=retrieval_enabled(ctx),
            override_dir=ctx.settings.prompts.override_dir,
            kubectl_route=kubectl_route(ctx),
        ),
        read_runtime("learning", f"{POST_INCIDENT}.txt"),
    ]
    if verify_live:
        parts.append(read_runtime("learning", "live_recovery_verification.txt"))
    return role_system_prompt(overlay_system("\n\n---\n\n".join(parts), ctx.overlay))


async def _run(ctx: RoleContext) -> RoleOutcome:
    packet = ctx.packet
    if packet is None:
        return RoleOutcome(
            artifact=None, transcript=[], error="learning ran with no incident packet"
        )

    external = bool(packet.external_feedback)
    content = render(
        read_runtime("learning", f"initial.{POST_INCIDENT}.txt"),
        {"PACKET_JSON": packet.model_dump_json(indent=2)},
    )
    if external:
        content += "\n\n" + read_runtime("learning", "external_feedback_guidance.txt")
    initial: list[Message] = [{"role": "user", "content": overlay_context(content, ctx.overlay)}]

    trigger = load_trigger(POST_INCIDENT, ctx.settings.memory.learning_config_dir)
    result = await run_until_submit(
        model=ctx.completion,
        role=ctx.spec.name,
        system=_system_prompt(
            ctx, verify_live=ctx.settings.memory.learning_verify and not external
        ),
        tools=ctx.toolset.definitions,
        dispatch=ctx.toolset.dispatch,
        submit_tool=ctx.spec.submit_tool,
        messages=initial,
        limits=LoopLimits.from_settings(
            ctx.settings.runtime, max_turns=trigger.max_turns, overlay=ctx.overlay
        ),
        session_id=ctx.session_id,
        journal=ctx.journal,
        middleware=ctx.middleware,
        tool_choice_mode=ctx.profile.quirks.tool_choice,
        keep_reasoning=ctx.profile.quirks.keep_reasoning,
        submit_validator=ctx.spec.submit_model.model_validate,
        submit_guard=overlay_guard(None, ctx.overlay),
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
        metadata={"trigger": trigger.name, "external_feedback": external},
    )


LEARNING = role(
    RoleSpec(
        name="learning",
        submit_tool="submit_learning_result",
        submit_model=LearningResult,
        # Not a graph node, so it fills no slot and its failure ends nothing: the
        # incident it is distilling closed minutes ago.
        state_slot=None,
        prompt="learning",
        run=_run,
        failure_sets_error=False,
    )
)
"""The distilling role. Read-only telemetry plus retrieval, run offline from a job."""
