"""SSH to a configured worker host, for the failures that cannot be fixed through the API.

A stopped kubelet is the case this exists for: the node stops reporting, so there is
nothing to patch, and recovery means running something on the host itself. Everything
here is deliberately narrow — an allowlisted alias, no local shell, a hard timeout, and
the write authority that puts every call through the approval gate.

Two honest notes. The node allowlist was never configured on any measured run, so no
graded session ever executed a command here; every invocation returned the
"not an allowed recovery node" refusal. The tool's *definition* was nonetheless in the
Execution role's prompt throughout, which is why it is kept rather than dropped: removing
it would edit a prompt, and a prompt edit is a change to measured behaviour.

Its description also names one specific fault, which the project's own rule against
benchmark-specific text in the harness forbids. Generalising that, and the two role
prompts that name it, is post-parity work.
"""

import asyncio
import re
from collections.abc import Mapping
from typing import Any

from kira.tools.context import ToolContext, redact
from kira.tools.errors import EXEC_OUTPUT_LIMIT, tool_error
from kira.tools.execution import runs_kubectl
from kira.tools.policy import always_writes
from kira.tools.registry import Authority, tool

__all__ = ["ssh_exec"]

_EXECUTION_STAGES = frozenset({"execution"})
_NODE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_TIMEOUT_SECONDS = 45
_MAX_COMMAND_CHARS = 4096


def host_authority(arguments: Mapping[str, Any]) -> Authority:
    """Classify a host command.

    Always a write. Nothing running on a node is inspectable enough to call read, and
    this tool exists to change host state.

    Args:
        arguments: The model's arguments.

    Returns:
        Always ``write``.
    """
    return "write"


@tool(
    name="ssh_exec",
    safety="cluster_write",
    authority=host_authority,
    parameters={
        "type": "object",
        "properties": {
            "node": {
                "type": "string",
                "description": "Configured SSH host alias for the worker node.",
            },
            "command": {
                "type": "string",
                "description": "Command to run on the allowlisted worker host.",
            },
        },
        "required": ["node", "command"],
    },
)
async def ssh_exec(ctx: ToolContext, node: str, command: str) -> str:
    """Run one command on a configured worker node.

    The stage check here duplicates the allowlist on purpose. A single misconfigured
    allowlist should not be enough to give a diagnostic role host access.

    Args:
        ctx: The run's context.
        node: An allowlisted SSH alias.
        command: The command to run on that host.

    Returns:
        Its output, or a string beginning with ``error:``.
    """
    if ctx.role not in _EXECUTION_STAGES:
        return tool_error("host recovery is permitted only for the execution agent")

    allowed = ctx.settings.host_recovery_nodes
    if not _NODE.fullmatch(node) or node not in allowed:
        visible = ", ".join(sorted(allowed)) or "(none configured)"
        return tool_error(f"host '{node}' is not an allowed recovery node; configured: {visible}")

    if not command.strip():
        return tool_error("ssh_exec requires a non-empty command")
    if "\x00" in command or len(command) > _MAX_COMMAND_CHARS:
        return tool_error(
            "ssh_exec command contains an unsupported character or exceeds 4096 characters"
        )
    if runs_kubectl(command):
        return tool_error(
            "do not run kubectl through ssh_exec. Use the Kubernetes tool "
            "for cluster reads and writes; ssh_exec is for host-level commands "
            "such as systemctl, journalctl, df, or ip."
        )

    process = await asyncio.create_subprocess_exec(
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        node,
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        stdout, _ = await asyncio.wait_for(process.communicate(), timeout=_TIMEOUT_SECONDS)
    except TimeoutError:
        process.kill()
        return tool_error(f"ssh_exec timed out after {_TIMEOUT_SECONDS}s")

    rendered = redact(
        ctx.evidence, f"exit_code={process.returncode}\n{stdout.decode(errors='replace')}"
    )
    return ctx.workspace.save_tool_output(rendered, "host_recovery", EXEC_OUTPUT_LIMIT)


always_writes(
    "ssh_exec",
    stages=frozenset({"execution"}),
    refusal=lambda stage: (
        f"error: host recovery is permitted only for the execution agent, not {stage}"
    ),
)
