"""SSH to a configured worker host, for the failures that cannot be fixed through the API.

A stopped kubelet is the case this exists for: the node stops reporting, so there is
nothing to patch, and recovery means running something on the host itself. Everything
here is deliberately narrow — an allowlisted alias, an allowlist of host binaries, no
local shell, a hard timeout, and the write authority that puts every call through the
approval gate.

Two honest notes. The node allowlist was never configured on any measured run, so no
graded session ever executed a command here; every invocation returned the
"not an allowed recovery node" refusal. The tool's *definition* was nonetheless in the
Execution role's prompt throughout, which is why it is kept rather than dropped, and why
the resolver offers it whatever the allowlist holds: withholding it, like removing it,
edits what the model was shown, and that is a change to measured behaviour. A node the
deployment may not reach is refused here instead, in the words the archive refused it in.

Its description also names one specific fault, which the project's own rule against
benchmark-specific text in the harness forbids. Generalising that, and the two role
prompts that name it, is post-parity work.
"""

import asyncio
import contextlib
import os
import re
import signal

from maver.tools.context import ToolContext, redact
from maver.tools.errors import EXEC_OUTPUT_LIMIT, timed_out, tool_error
from maver.tools.execution import command_segments, runs_kubectl
from maver.tools.policy import always_writes
from maver.tools.registry import tool

__all__ = ["ssh_exec"]

_EXECUTION_STAGES = frozenset({"execution"})
_NODE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_TIMEOUT_SECONDS = 45
_REAP_SECONDS = 5.0
_MAX_COMMAND_CHARS = 4096

# The binaries this tool exists to run. The argument goes to the node's login shell, and
# on a k3s host that shell holds a cluster-admin kubeconfig, so every way of reaching the
# cluster from it — `sh -c 'kubectl ...'`, `xargs kubectl`, `eval`, `timeout 5 kubectl`,
# `python -c`, a script that wraps any of them — is a cluster write outside the kubectl
# tool's screen, outside the approval gate's selectors and outside the ServiceAccount's
# RBAC. No position-based screen can close a shell: the guard below caught kubectl after
# a separator and none of those. So the vocabulary is closed instead, to the host-level
# commands the docstring and the tool's own description have always claimed it is for.
#
# The contentious entries, each decided on whether it can reach the cluster's credentials
# or start something that can:
#   systemctl   the fault this tool exists to repair is a unit that is not running. A
#               unit runs what its file says, and no binary here can write one.
#   swapoff     a kubelet refuses to start while swap is on, so it is part of the repair.
#   sysctl      `-w` writes a kernel parameter. conntrack, inotify and file-descriptor
#               limits are node faults with node fixes, and none reaches the API server.
#   cat head tail grep ls stat
#               these read the node, credential files included. Reading /var/log and
#               /etc is most of what diagnosis on a host is, and the line this list
#               draws is execution: none of them runs anything, and none writes content
#               of the model's choosing, so a unit file cannot be planted for systemctl.
#
# Refused, though a host operator would reach for them:
#   crictl ctr  `crictl exec` opens a shell inside a container that may mount a service
#               account token, and no first-token screen separates it from `crictl ps`.
#   kubelet containerd
#               running the daemon directly starts a second one holding the node's
#               credentials; `systemctl status` answers the version question.
#   iptables    no first-token screen separates `-L` from `-F`, and flushing a node's
#               rules breaks every Service on it.
#   sh bash python perl xargs eval exec source .
#               each one's purpose is to run something else.
# It is a constant rather than a setting because it is the boundary itself: an operator
# who can add `bash` to it has the shell back, and nothing in a deployment needs that.
_ALLOWED_BINARIES = frozenset(
    {
        "cat",
        "df",
        "dig",
        "dmesg",
        "du",
        "free",
        "grep",
        "head",
        "ip",
        "journalctl",
        "ls",
        "lsblk",
        "mount",
        "netstat",
        "nslookup",
        "ping",
        "ps",
        "ss",
        "stat",
        "swapoff",
        "sysctl",
        "systemctl",
        "tail",
        "top",
        "uptime",
    }
)

# Wrappers that change how a program runs but not which program runs, so the screen looks
# past them to the program. Their options are not: `sudo -u`, `sudo -i` and `env -i` all
# change what runs or what it runs with, so a segment carrying one is refused rather than
# walked past. `env` is followed by its VAR=value assignments, `timeout` by its duration.
_TRANSPARENT = frozenset({"sudo", "nohup"})
_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_DURATION = re.compile(r"^\d+(?:\.\d+)?[smhd]?$")

_NOT_A_HOST_COMMAND = (
    "ssh_exec does not run {what!r}. It runs host-level commands only: {allowed}. "
    "Every command in the line must start with one of those, optionally behind sudo, "
    "nohup, env VAR=value or timeout <seconds> and with no option in between, and "
    "named plainly rather than by path. Use the Kubernetes tool for anything in the "
    "cluster, and read a node's state with the commands above."
)


def _binary(segment: str) -> str:
    """Name the program one simple command would run.

    Args:
        segment: One simple command, as the node's shell would see it.

    Returns:
        The token naming the program, past the wrappers that do not change it, or the
        empty string when the segment names none.
    """
    tokens = segment.split()
    index = 0
    while index < len(tokens):
        token = tokens[index].strip("\"'")
        if token in _TRANSPARENT:
            index += 1
        elif token == "env":
            index += 1
            while index < len(tokens) and _ASSIGNMENT.match(tokens[index]):
                index += 1
        elif (
            token == "timeout"
            and index + 1 < len(tokens)
            and _DURATION.fullmatch(tokens[index + 1])
        ):
            index += 2
        else:
            return token
    return ""


def _refuse_unknown_binary(command: str) -> str | None:
    """Screen every simple command in the line against the allowlist.

    Every segment, not the first: the shell runs all of them, and the ones after the
    first are where a cluster call hides.

    Args:
        command: The command line as the model wrote it.

    Returns:
        The refusal to return, or None when every segment runs an allowed binary.
    """
    allowed = ", ".join(sorted(_ALLOWED_BINARIES))
    segments = command_segments(command)
    if not segments:
        return tool_error(_NOT_A_HOST_COMMAND.format(what=command.strip(), allowed=allowed))
    for segment in segments:
        name = _binary(segment)
        if name not in _ALLOWED_BINARIES:
            return tool_error(
                _NOT_A_HOST_COMMAND.format(what=name or segment.strip(), allowed=allowed)
            )
    return None


@tool(
    name="ssh_exec",
    safety="cluster_write",
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
    # Kept in front of the allowlist, which would refuse kubectl anyway, because this
    # refusal names the tool to use instead and the allowlist's cannot.
    if runs_kubectl(command):
        return tool_error(
            "do not run kubectl through ssh_exec. Use the Kubernetes tool "
            "for cluster reads and writes; ssh_exec is for host-level commands "
            "such as systemctl, journalctl, df, or ip."
        )
    if refusal := _refuse_unknown_binary(command):
        return refusal

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
        # In its own process group, as the direct transport's kubectl is, so the bound
        # below can reach what ssh started as well as ssh itself.
        start_new_session=True,
    )
    try:
        stdout, _ = await asyncio.wait_for(process.communicate(), timeout=_TIMEOUT_SECONDS)
    except TimeoutError:
        # kill() reaches ssh alone, and anything it left behind keeps the inherited
        # pipes open after the model has been told the call was abandoned on time. The
        # reap is bounded for the same reason the call is.
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(process.wait(), _REAP_SECONDS)
        return timed_out("ssh_exec", _TIMEOUT_SECONDS)

    rendered = redact(
        ctx.evidence, f"exit_code={process.returncode}\n{stdout.decode(errors='replace')}"
    )
    return ctx.workspace.save_tool_output(rendered, "host_recovery", EXEC_OUTPUT_LIMIT)


# Every call is a write: nothing running on a node is inspectable enough to call a read,
# and this tool exists to change host state.
always_writes(
    "ssh_exec",
    stages=frozenset({"execution"}),
    refusal=lambda stage: (
        f"error: host recovery is permitted only for the execution agent, not {stage}"
    ),
)
