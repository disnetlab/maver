"""What the approval gate asks a tool about one of its calls.

The gate has to decide read-or-write, name the stages allowed to write, render something
an operator can actually review, and produce selectors a policy rule can match — all
without importing kubectl or knowing what a Deployment is. So the tool supplies those as
data, registered beside itself.

Note that this is **not** ``ToolSpec.safety``. Safety says what a tool is capable of;
policy says what one call is asking for. Wiring the gate to safety would gate every
sandbox call, because the sandboxes are capable of running arbitrary code — and then
refuse them outside the execution stage, which would break diagnosis in every role.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

__all__ = ["ActionClass", "ToolPolicy", "policy_for", "register_policy"]


class ActionClass(StrEnum):
    """What one tool call does to the world.

    ``BLOCKED`` is here because the tool refuses those itself, and the gate must not
    offer them for approval: a command nobody may run is not a command an operator may
    authorise.
    """

    READ = "read"
    WRITE = "write"
    EXEC = "exec"
    """A command run inside a container.

    Its own class because it cannot be proven read-only and was permitted for every role
    in every measured run. Treating it as a read would hide it from the record; treating
    it as a write would change what was measured. It passes by default, is recorded as
    an unprovable-authority call, and a policy rule can hold or deny it.
    """
    BLOCKED = "blocked"


def _no_selectors(arguments: Mapping[str, Any]) -> tuple[str, ...]:
    return ()


def _default_stage_refusal(stage: str) -> str:
    return f"error: writes are not permitted for the {stage} agent (read-only)"


@dataclass(frozen=True)
class ToolPolicy:
    """How the gate reasons about one tool."""

    classify: Callable[[Mapping[str, Any]], ActionClass]
    write_stages: frozenset[str] = field(default_factory=frozenset)
    summarise: Callable[[Mapping[str, Any]], str] = repr
    selectors: Callable[[Mapping[str, Any]], tuple[str, ...]] = _no_selectors
    """Facets a policy rule can match, e.g. ``("verb:patch", "resource:secret")``.

    This is what makes per-verb policy possible without kubectl leaking upward into the
    settings layer.
    """
    targets: Callable[[Mapping[str, Any]], tuple[str, ...]] = _no_selectors
    stage_refusal: Callable[[str], str] = _default_stage_refusal


_POLICIES: dict[str, ToolPolicy] = {}


def register_policy(tool_name: str, policy: ToolPolicy) -> None:
    """Attach a policy to a tool.

    Args:
        tool_name: The tool it governs.
        policy: How the gate should reason about it.
    """
    _POLICIES[tool_name] = policy


def policy_for(tool_name: str) -> ToolPolicy | None:
    """Look up a tool's policy.

    A missing policy is not an error here. The gate treats it as a write, which is the
    fail-closed reading: a tool nobody classified is a tool nobody vouched for.

    Args:
        tool_name: The tool.

    Returns:
        Its policy, or None.
    """
    return _POLICIES.get(tool_name)


def read_only(tool_name: str) -> None:
    """Register a tool as reading only.

    Args:
        tool_name: The tool.
    """
    register_policy(tool_name, ToolPolicy(classify=lambda _: ActionClass.READ))


def always_writes(tool_name: str, *, stages: frozenset[str], refusal: Callable[[str], str]) -> None:
    """Register a tool whose every call is a write.

    Args:
        tool_name: The tool.
        stages: Stages permitted to call it.
        refusal: Renders the refusal shown when another stage tries.
    """
    register_policy(
        tool_name,
        ToolPolicy(
            classify=lambda _: ActionClass.WRITE,
            write_stages=stages,
            summarise=lambda args: " ".join(f"{k}={v}" for k, v in sorted(args.items())),
            stage_refusal=refusal,
        ),
    )
