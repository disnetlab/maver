"""The two sandboxes: a shell and a Python interpreter, both without cluster access.

These exist to process output the model already has — filter a saved manifest, count
something across a log file, compute a difference — rather than to reach anything new.
The containment property is structural: the subprocess environment carries no cluster
credentials, so there is no configuration in which a command here can mutate a cluster.

The kubectl and ssh guards below are therefore not the safety boundary. They are about
turn economy: a model that shells out to kubectl gets an authentication failure, reads it
as a broken tool, and retries variants of the same command until its budget is gone.
Naming the right tool in the refusal is what stops that.

The shell tool is called ``bash_exec`` and must not be renamed back. A middlebox between
the cluster and a cleartext model endpoint resets any payload containing the literal
``shell_exec(``, which is a remote-execution signature. It silently killed every call of
one role for a whole run.
"""

import asyncio
import re
import sys
import tempfile
import textwrap
from pathlib import Path

from kira.tools.context import ToolContext, deliver
from kira.tools.errors import format_exit_result, tool_error
from kira.tools.policy import read_only
from kira.tools.registry import tool
from kira.tools.sandbox import SANDBOX_ERROR_PREFIX, sandboxed_command

__all__ = ["bash_exec", "code_exec"]

OUTPUT_LIMIT = 8192

# `kubectl` at command position only: at the start, or after a separator or pipe. So
# `foo && kubectl get pods` is caught and `grep kubectl notes.txt` is not.
_LEADING_KUBECTL = re.compile(r"(?:^|[\n;&|])\s*(?:sudo\s+)?kubectl\b")
_LEADING_SSH = re.compile(r"(?:^|[\n;&|])\s*(?:sudo\s+)?ssh\b")
_LEADING_SSH_TOOL = re.compile(r"(?:^|[\n;&|])\s*ssh_exec\b")

# In Python source, a bare mention is not a shell-out. Both a reference and a way to run
# it must be present, so a script that merely reads a kubectl-named file still runs.
_KUBECTL = re.compile(r"\bkubectl\b")
_SSH = re.compile(r"\bssh\b")
_SSH_TOOL = re.compile(r"\bssh_exec\b")
_SHELLOUT = re.compile(
    r"subprocess|os\.system|os\.popen|\bpopen\b|check_output|check_call|getoutput|getstatusoutput"
)

_KUBECTL_FROM_SHELL = (
    "do not run kubectl through bash_exec — the session shell has no "
    'cluster credentials. Use exec_kubectl_cmd_safely(cmd, save_as="out.json") '
    "for cluster reads, then bash_exec only to read/process the saved file at "
    "tool_outputs/ in the current directory."
)
_KUBECTL_FROM_CODE = (
    "do not run kubectl from code_exec — this sandbox has no cluster "
    "credentials, so shelling out to kubectl fails. Run cluster reads with "
    'exec_kubectl_cmd_safely(cmd, save_as="out.json"), then read '
    "tool_outputs/out.json (your current directory) here to parse/process it."
)
_SSH_FROM_SANDBOX = (
    "do not run ssh from {origin}. Host mutations must use "
    "ssh_exec(node, command), which is execution-only and HITL-gated."
)


async def _run(ctx: ToolContext, command: list[str], *, timeout: float, label: str) -> str:
    """Run a sandboxed subprocess and render its result.

    Args:
        ctx: The run's context.
        command: The command to run, before sandbox wrapping.
        timeout: Seconds to wait before killing it.
        label: Used in the saved output's filename.

    Returns:
        The rendered result, or a string beginning with ``error:``.
    """
    session = str(ctx.workspace.path)
    wrapped = sandboxed_command(command, session) if ctx.settings.fs_sandbox else command
    process = await asyncio.create_subprocess_exec(
        *wrapped,
        cwd=session,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=ctx.workspace.subprocess_env(),
    )
    try:
        stdout, _ = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except TimeoutError:
        process.kill()
        return tool_error(f"{label} timed out after {timeout:.0f}s")

    output = stdout.decode(errors="replace")
    if process.returncode == 126 and output.startswith(SANDBOX_ERROR_PREFIX):
        return tool_error(output.strip())
    rendered = format_exit_result(process.returncode or 0, output)
    return deliver(ctx, rendered, label, OUTPUT_LIMIT)


@tool(
    name="bash_exec",
    safety="local_exec",
    parameters={
        "type": "object",
        "properties": {"cmd": {"type": "string", "description": "The bash command to run."}},
        "required": ["cmd"],
    },
)
async def bash_exec(ctx: ToolContext, cmd: str) -> str:
    """Run a shell command in the session workspace.

    Args:
        ctx: The run's context.
        cmd: The command to run.

    Returns:
        Its output, or a string beginning with ``error:``.
    """
    if _LEADING_KUBECTL.search(cmd):
        return tool_error(_KUBECTL_FROM_SHELL)
    if _LEADING_SSH.search(cmd):
        return tool_error(_SSH_FROM_SANDBOX.format(origin="bash_exec"))
    if _LEADING_SSH_TOOL.search(cmd):
        return tool_error(
            "ssh_exec is a function tool, not a shell command. Call "
            "ssh_exec(node=<allowed worker alias>, command=<host command>) "
            "directly; do not type it into bash_exec."
        )
    return await _run(
        ctx,
        ["/bin/sh", "-c", cmd],
        timeout=max(10.0, float(ctx.settings.shell_exec_timeout_s)),
        label="shell",
    )


@tool(
    name="code_exec",
    safety="local_exec",
    parameters={
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Python source code to execute."},
            "language": {
                "type": "string",
                "description": "Programming language. Currently only 'python' is supported.",
            },
        },
        "required": ["code", "language"],
    },
)
async def code_exec(ctx: ToolContext, code: str, language: str) -> str:
    """Run a Python script in the session workspace.

    Args:
        ctx: The run's context.
        code: The source to run.
        language: Must be python.

    Returns:
        Its output, or a string beginning with ``error:``.
    """
    if language.lower() not in ("python", "python3"):
        return tool_error(f"unsupported language '{language}'. Only 'python' is supported.")
    if _KUBECTL.search(code) and _SHELLOUT.search(code):
        return tool_error(_KUBECTL_FROM_CODE)
    if _SSH.search(code) and _SHELLOUT.search(code):
        return tool_error(_SSH_FROM_SANDBOX.format(origin="code_exec"))
    if _SSH_TOOL.search(code):
        return tool_error(
            "ssh_exec is a function tool, not a Python or shell command. "
            "Call ssh_exec(node=<allowed worker alias>, command=<host command>) "
            "directly."
        )

    with tempfile.NamedTemporaryFile(
        suffix=".py", dir=ctx.workspace.path, mode="w", delete=False
    ) as handle:
        handle.write(textwrap.dedent(code))
        path = handle.name
    try:
        return await _run(
            ctx,
            [sys.executable, path],
            timeout=max(10.0, float(ctx.settings.code_exec_timeout_s)),
            label="code_exec",
        )
    finally:
        Path(path).unlink(missing_ok=True)


# Sandboxed and credential-free by construction, so these are reads however
# much code they run. Gating them on capability would refuse every diagnostic
# role's use of them.
read_only("bash_exec")
read_only("code_exec")
