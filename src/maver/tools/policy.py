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

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from maver.settings.groups import ToolSettings

__all__ = ["ActionClass", "ToolPolicy", "may_write", "policy_for", "register_policy"]


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

    It cannot be proven read-only. Diagnostic roles may request it, but human approval
    is required by default, independently of automatic approval for ordinary writes.
    Unattended experiments must explicitly disable that requirement.
    """
    BLOCKED = "blocked"


_EXEC_DISPATCH: ContextVar[tuple[str, str, str] | None] = ContextVar(
    "authorized_exec_dispatch", default=None
)


@contextmanager
def authorized_exec(session_id: str, stage: str, command: str) -> Iterator[None]:
    """Let the tool dispatch only the command approved by the enclosing gate."""
    token = _EXEC_DISPATCH.set((session_id, stage, command))
    try:
        yield
    finally:
        _EXEC_DISPATCH.reset(token)


def exec_is_authorized(session_id: str, stage: str, command: str) -> bool:
    """Check the gate's scoped authorization, including incident and role identity."""
    return _EXEC_DISPATCH.get() == (session_id, stage, command)


def _no_selectors(arguments: Mapping[str, Any]) -> tuple[str, ...]:
    return ()


def _default_stage_refusal(stage: str) -> str:
    return f"error: writes are not permitted for the {stage} agent (read-only)"


@dataclass(frozen=True)
class ToolPolicy:
    """How the gate reasons about one tool."""

    classify: Callable[[Mapping[str, Any], ToolSettings], ActionClass]
    """Says what one call is asking for, under this deployment's tool settings.

    The settings are a parameter rather than a module default because some of what a
    tool refuses is configurable. A classifier reading the shipped default while the
    dispatch reads the deployment's would disagree the moment an operator changed one:
    a screen the deployment had lifted still classified as ``BLOCKED``, the gate passed
    it through expecting the tool to refuse it, and the tool ran it with no authority
    decision anywhere.
    """
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


def may_write(tool_name: str, stage: str, widened: frozenset[str] = frozenset()) -> bool:
    """Report whether a stage may issue a write through one tool.

    A tool knows the role it was bound to, while the gate is told which phase of that
    role is running, and a write policy can name one phase because the single-agent
    configuration answers both graded questions over one transcript. So a role whose
    phase may write passes here, and which phase it is stays the gate's decision: this
    is the coarser of the two checks by construction, and it is the one that still holds
    when the gate is misconfigured.

    Args:
        tool_name: The tool being called.
        stage: The calling role or phase.
        widened: Stages this deployment opened writes to, beyond the tool's own set.
            Passed in rather than read from the environment: a deployment assembled in
            code widened the gate and the tool refused anyway, with nothing to say why.

    Returns:
        True when the tool's own policy, or the deployment's widening of it, allows it.
    """
    policy = policy_for(tool_name)
    allowed = (policy.write_stages if policy is not None else frozenset()) | widened
    return stage in allowed or any(phase.split(".", 1)[0] == stage for phase in allowed)


def read_only(tool_name: str) -> None:
    """Register a tool as reading only.

    Args:
        tool_name: The tool.
    """
    register_policy(tool_name, ToolPolicy(classify=lambda _args, _tools: ActionClass.READ))


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
            classify=lambda _args, _tools: ActionClass.WRITE,
            write_stages=stages,
            summarise=lambda args: " ".join(f"{k}={v}" for k, v in sorted(args.items())),
            stage_refusal=refusal,
        ),
    )
