"""The two tools that exist only under this harness.

Rolling a write back is not a Kubernetes operation. It needs a journal holding the
inverse of everything already applied, and that journal is the harness's own session
state, keyed on the correlation header this adapter sets. No transport talking straight
to an API server can reconstruct it, and the inverse of a delete does not exist in
general, so these are supplied by the adapter rather than registered in the core: under
any other environment they are simply absent, which is honest, where a tool that existed
and always failed would read to a model as a fault to work around.

Their roles are the ones the measured configuration gave them. Their descriptions travel
with them rather than sitting in the shipped catalogue, for the same reason their
allowlist entries do not exist.
"""

from kira.adapters.base import AdapterTool
from kira.tools.context import ToolContext
from kira.tools.policy import always_writes, read_only
from kira.tools.registry import ToolSpec

__all__ = ["adapter_tools"]

_WRITE_ROLES = frozenset({"execution", "single.mitigation"})
"""Stages the gate lets call the rollback. The single-agent configuration names its
remediation phase separately because one transcript answers both graded questions."""

_ROLES = frozenset({"execution", "single"})

_NO_ARGUMENTS = {"type": "object", "properties": {}}

_ROLLBACK_DESCRIPTION = {
    "rollback_command": {
        "default": (
            "Roll back the last kubectl write you successfully executed with "
            "exec_kubectl_cmd_safely. Takes no arguments. Routed through the HITL gate "
            "like any cluster write. Call get_previous_rollbackable_cmd first to confirm "
            "what will be reverted; repeated calls roll back successive commands."
        )
    }
}

_PREVIOUS_DESCRIPTION = {
    "get_previous_rollbackable_cmd": {
        "default": (
            "List the kubectl commands you previously executed that can still be rolled "
            "back, in the order rollback_command would revert them. Read-only."
        )
    }
}


async def _rollback_command(ctx: ToolContext) -> str:
    """Undo the last cluster write.

    Undoing a write is itself a write: it changes the cluster, and one issued against
    the wrong state is as damaging as the change it reverses. So it carries the same
    authority and passes the same gate.
    """
    return await ctx.transport.rollback_last()


async def _get_previous_rollbackable_cmd(ctx: ToolContext) -> str:
    """List the cluster writes that can currently be undone."""
    return await ctx.transport.rollbackable_commands()


ROLLBACK_COMMAND = ToolSpec(
    name="rollback_command",
    parameters=_NO_ARGUMENTS,
    safety="cluster_write",
    dispatch=_rollback_command,
    requires=frozenset({"kubernetes.rollback"}),
    descriptions=_ROLLBACK_DESCRIPTION,
)

GET_PREVIOUS_ROLLBACKABLE_CMD = ToolSpec(
    name="get_previous_rollbackable_cmd",
    parameters=_NO_ARGUMENTS,
    safety="read_only",
    dispatch=_get_previous_rollbackable_cmd,
    requires=frozenset({"kubernetes.rollback"}),
    descriptions=_PREVIOUS_DESCRIPTION,
)

# Registered where the tools are declared, so the approval gate reasons about them the
# same way it reasons about any other write. The gate reads a process-wide policy
# registry rather than the toolset, and a tool it cannot find is treated as a write by
# whichever stage calls it, which would lose the read/write split these two have.
read_only("get_previous_rollbackable_cmd")
always_writes(
    "rollback_command",
    stages=_WRITE_ROLES,
    refusal=lambda stage: f"error: rollback is not permitted for the {stage} agent (read-only)",
)


def adapter_tools() -> tuple[AdapterTool, ...]:
    """The tools this environment contributes, with the roles allowed to see them.

    Returns:
        The rollback pair, offered to the remediating role and to the single-agent
        configuration and to nothing else.
    """
    return (
        AdapterTool(spec=ROLLBACK_COMMAND, roles=_ROLES),
        AdapterTool(spec=GET_PREVIOUS_ROLLBACKABLE_CMD, roles=_ROLES),
    )
