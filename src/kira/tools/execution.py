"""The two sandboxes: a shell and a Python interpreter, both without cluster access.

These exist to process output the model already has — filter a saved manifest, count
something across a log file, compute a difference — rather than to reach anything new.
The containment property is structural: the subprocess environment carries no cluster
credentials, so there is no configuration in which a command here can mutate a cluster.

The kubectl and ssh guards below are therefore not the safety boundary *here*. They are
about turn economy: a model that shells out to kubectl gets an authentication failure,
reads it as a broken tool, and retries variants of the same command until its budget is
gone. Naming the right tool in the refusal is what stops that.

The kubectl one is shared with the host tool, where it *is* a safety boundary, which is
why it lives in one place: the two copies had drifted, and the weaker of them guarded the
tool that reaches a credentialed shell.

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
from kira.tools.errors import EXEC_OUTPUT_LIMIT, format_exit_result, timed_out, tool_error
from kira.tools.policy import read_only
from kira.tools.registry import tool
from kira.tools.sandbox import SANDBOX_ERROR_PREFIX, sandboxed_command

__all__ = ["bash_exec", "code_exec", "command_segments", "runs_kubectl"]

# What ends one command and begins the next, defined once. The guard below and the host
# tool's segmentation both read it, because two screens disagreeing about what a
# separator is is exactly how the host tool came to see neither a newline nor a subshell.
_SEPARATORS = r"\n;&|(`"

# Command position only: the start of the command, or after a separator, a pipe or the
# opening of a subshell. So `foo && kubectl get pods`, a second line, and
# `echo $(kubectl get secret)` are all caught, and `grep kubectl notes.txt` is not.
_AT_COMMAND_POSITION = rf"(?:^|[{_SEPARATORS}])\s*"
_SEGMENT = re.compile(rf"[{_SEPARATORS}]")
_KUBECTL_COMMAND = re.compile(_AT_COMMAND_POSITION + r"(?:sudo\s+)?(?:k3s\s+)?kubectl\b")
_LEADING_SSH = re.compile(_AT_COMMAND_POSITION + r"(?:sudo\s+)?ssh\b")
_LEADING_SSH_TOOL = re.compile(_AT_COMMAND_POSITION + r"ssh_exec\b")

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


def command_segments(command: str) -> list[str]:
    """Split a command line into the simple commands a shell would run in it.

    The separators are the ones the kubectl guard already recognises, so a caller that
    screens every segment sees each command the guard sees at a command position, and
    the ones it does not — a pipeline's right-hand side reached through ``xargs``, a
    wrapper's argument — as segments of their own.

    Quoting is deliberately not honoured. A separator inside a quoted argument splits
    the segment anyway, which can only produce more segments to screen and never fewer,
    so the error is a refusal rather than a command that slipped past. Honouring quotes
    would mean writing a shell parser, which is the thing this declines to do.

    Args:
        command: The command line as the model wrote it.

    Returns:
        Its non-empty segments, in order.
    """
    return [segment for segment in _SEGMENT.split(command) if segment.strip()]


def runs_kubectl(command: str) -> bool:
    """Report whether a shell command would run kubectl.

    Shared with the host tool because the two copies of this question had drifted, and
    the one that answered it for ``ssh_exec`` — the only tool that reaches a shell
    holding the node's own kubeconfig, which on a k3s host is cluster-admin — was the
    one that saw neither a newline nor a subshell.

    Args:
        command: The command as the model wrote it.

    Returns:
        True when kubectl runs at a command position within it.
    """
    return _KUBECTL_COMMAND.search(command) is not None


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
        return timed_out(label, timeout)

    output = stdout.decode(errors="replace")
    if process.returncode == 126 and output.startswith(SANDBOX_ERROR_PREFIX):
        return tool_error(output.strip())
    rendered = format_exit_result(process.returncode or 0, output)
    return deliver(ctx, rendered, label, EXEC_OUTPUT_LIMIT)


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
    if runs_kubectl(cmd):
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
