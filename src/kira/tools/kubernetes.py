"""Running kubectl, and deciding what a given kubectl command is asking for.

The verb and flag tables below encode incident-driven tuning and are carried over
unchanged. The classification they feed is what the approval gate acts on, so it fails
closed everywhere it is unsure: an unrecognised verb, an unrecognised leading flag, or an
unknown ``rollout`` subcommand is treated as a write.

Classification happens *after* global flags are walked past. A command whose verb is
hidden behind ``--context foo`` must not read as something else, and the parser returning
"no verb found" must not read as "nothing to gate".
"""

import re
import shlex
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from kira.settings.groups import ToolSettings
from kira.tools.context import ToolContext, deliver, redact
from kira.tools.errors import OUTPUT_LIMIT, tool_error
from kira.tools.policy import ActionClass, ToolPolicy, register_policy
from kira.tools.registry import Authority, tool

__all__ = [
    "KubernetesPolicy",
    "classify",
    "exec_kubectl_cmd_safely",
    "read_kubectl",
    "tokenize",
]

_WRITE_VERBS = {
    "apply", "create", "patch", "edit", "replace", "scale", "rollout",
    "annotate", "label", "set", "delete", "drain", "cordon", "uncordon",
    "taint", "expose", "autoscale",
}  # fmt: skip

_READ_VERBS = {
    "describe", "events", "get", "logs", "top", "version", "wait",
}  # fmt: skip

# Interactive, session-modifying, or broad discovery; never appropriate.
_BLOCKED_VERBS = {
    "api-resources", "api-versions", "auth", "cluster-info", "cp",
    "debug", "explain", "attach", "port-forward", "proxy",
}  # fmt: skip

_BLOCKED_FLAGS_WITH_VALUE = {
    "--as", "--as-group", "--certificate-authority", "--client-certificate",
    "--client-key", "--cluster", "--context", "--kubeconfig", "--raw",
    "--server", "--tls-server-name", "--token", "--user",
}  # fmt: skip

_GLOBAL_FLAGS_WITH_VALUE = {
    "-n", "--namespace", "--as", "--as-group", "--cache-dir",
    "--certificate-authority", "--client-certificate", "--client-key",
    "--cluster", "--context", "--kubeconfig", "--log-flush-frequency",
    "--profile", "--profile-output", "--request-timeout", "--server",
    "--tls-server-name", "--token", "--user", "--v", "--vmodule",
}  # fmt: skip

_GLOBAL_BOOL_FLAGS = {
    "-A", "-h", "--add-dir-header", "--all-namespaces", "--help",
    "--insecure-skip-tls-verify", "--match-server-version", "--skip-headers",
    "--skip-log-headers", "--warnings-as-errors",
}  # fmt: skip

_READ_ROLLOUT_SUBCOMMANDS = {"history", "status"}

# Only *bare* operator tokens count. shlex keeps an operator that appears inside quotes
# within one token, so a quoted go-template or jsonpath value is unaffected.
_SHELL_OPERATORS = {"|", "||", "&", "&&", ";", ">", ">>", "<", "<<", "|&"}

_INTERACTIVE_VERBS = {"run", "exec"}


@dataclass(frozen=True)
class KubernetesPolicy:
    """What this deployment refuses to let the model ask kubectl for.

    Distinct from authority, which is about *who* may do a thing. These are commands no
    role may issue at all, for reasons that are about scoping an experiment or hiding a
    benchmark's own machinery rather than about safety.
    """

    deny_all_namespaces: bool = True
    """Refuse cluster-wide reads.

    A cluster-wide read is normal for a real operator, so this is incident scoping and
    not a safety property. It defaults to on, which is the behaviour every published
    measurement was taken under, and an operator running Kira for real can turn it off.
    """

    denied_argument_pattern: re.Pattern[str] | None = field(
        default_factory=lambda: re.compile(r"chaos", re.IGNORECASE)
    )
    """Arguments the model may not name.

    A benchmark that injects faults with a named controller leaks the answer to anyone
    who can query that controller's resources, which turns diagnosis into a lookup. An
    integrity knob, then, not a safety rule — and it refuses reads and writes alike, so
    a real cluster whose own namespaces match it cannot be operated through this agent
    until the pattern is cleared.
    """

    @classmethod
    def from_settings(cls, tools: ToolSettings) -> "KubernetesPolicy":
        """Read the deployment's screens.

        Args:
            tools: The tool-layer settings this run carries.

        Returns:
            The policy this deployment refuses under.
        """
        return _policy(tools.kubectl_deny_all_namespaces, tools.kubectl_denied_argument_pattern)

    def denies_flag(self, arg: str) -> bool:
        """Report whether a flag is refused outright.

        Args:
            arg: One argument.

        Returns:
            True when the flag may not be used.
        """
        flag = arg.split("=", 1)[0]
        if flag in _BLOCKED_FLAGS_WITH_VALUE:
            return True
        return self.deny_all_namespaces and flag in {"-A", "--all-namespaces"}

    def denies_argument(self, arg: str) -> bool:
        """Report whether an argument names something the model may not see.

        Args:
            arg: One argument.

        Returns:
            True when the argument is refused.
        """
        if self.denied_argument_pattern is None:
            return False
        return bool(self.denied_argument_pattern.search(arg))


@lru_cache(maxsize=8)
def _policy(deny_all_namespaces: bool, pattern: str | None) -> KubernetesPolicy:
    """Build one policy per distinct configuration, since it is read on every call."""
    return KubernetesPolicy(
        deny_all_namespaces=deny_all_namespaces,
        denied_argument_pattern=re.compile(pattern, re.IGNORECASE) if pattern else None,
    )


DEFAULT_POLICY = KubernetesPolicy()
"""The shipped screens, for a caller with no settings to read. Every call site inside
the package reads the deployment's instead."""


def tokenize(cmd: str) -> list[str]:
    """Split a kubectl command into arguments, dropping a leading ``kubectl``.

    Note that the leading binary is dropped **here**. A caller that slices the result
    again eats the verb, and every command then classifies as something else.

    Args:
        cmd: The command as the model wrote it.

    Returns:
        The arguments after the binary.
    """
    try:
        parts = shlex.split(cmd)
    except ValueError:
        parts = cmd.split()
    if parts and parts[0] == "kubectl":
        parts = parts[1:]
    return parts


def leading_verb(argv: Sequence[str]) -> tuple[str | None, int | None]:
    """Find the verb, walking past global flags.

    Args:
        argv: The arguments.

    Returns:
        The verb and its index, or ``(None, None)`` when the shape is unrecognised.
        Callers treat that as a write, so an odd flag cannot slip past the gate.
    """
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--":
            i += 1
            continue
        if not arg.startswith("-"):
            return arg.lower(), i

        flag = arg.split("=", 1)[0]
        if flag in _BLOCKED_FLAGS_WITH_VALUE:
            return None, None
        if flag in _GLOBAL_FLAGS_WITH_VALUE:
            i += 1 if "=" in arg else 2
            continue
        if flag in _GLOBAL_BOOL_FLAGS:
            i += 1
            continue
        return None, None
    return None, None


def classify(argv: Sequence[str], policy: KubernetesPolicy = DEFAULT_POLICY) -> Authority:
    """Say what one kubectl invocation is asking for.

    Args:
        argv: The arguments, without the binary.
        policy: What this deployment refuses outright.

    Returns:
        ``read``, ``write``, ``exec`` or ``blocked``.
    """
    if not argv:
        return "read"
    if any(policy.denies_flag(a) or policy.denies_argument(a) for a in argv):
        return "blocked"

    verb, idx = leading_verb(argv)
    if verb is None:
        return "write"  # fail closed: an unrecognised shape must not bypass the gate
    if verb in _BLOCKED_VERBS:
        return "blocked"
    if verb == "exec":
        # Its own class, not a read. An in-container command cannot be proven
        # read-only, but the diagnostic roles used it for read-only inspection in every
        # measured run, so folding it into `write` would change what was measured.
        # Naming it lets the approval policy hold or deny it without a code change, and
        # lets the event log record that its authority was never provable.
        return "exec"
    if verb == "rollout":
        sub = argv[idx + 1].lower() if idx is not None and idx + 1 < len(argv) else ""
        return "read" if sub in _READ_ROLLOUT_SUBCOMMANDS else "write"
    if verb in _WRITE_VERBS:
        return "write"
    if verb in _READ_VERBS:
        return "read"
    return "write"  # unknown verb fails closed


def kubectl_authority(arguments: Mapping[str, Any], tools: ToolSettings | None = None) -> Authority:
    """Classify a pending call for the approval gate.

    Args:
        arguments: The model's arguments.
        tools: The deployment's tool settings, so this reads the same screens the
            dispatch will. None falls back to the shipped ones.

    Returns:
        What the call is asking for. A call carrying no readable command is classified
        as a write, not as a read: the gate is deciding whether to ask a human, and an
        argument shape the classifier cannot read is not evidence that nothing will be
        changed. The dispatch below would reject it anyway, but the audit record would
        have said a write was a read.
    """
    command = str(arguments.get("cmd") or "").strip()
    if not command:
        return "write"
    policy = KubernetesPolicy.from_settings(tools) if tools is not None else DEFAULT_POLICY
    return classify(tokenize(command), policy)


def _bare_shell_operator(argv: Sequence[str]) -> str | None:
    return next((a for a in argv if a in _SHELL_OPERATORS), None)


def _interactive_flag(argv: Sequence[str]) -> str | None:
    verb, _ = leading_verb(argv)
    if verb not in _INTERACTIVE_VERBS:
        return None
    for arg in argv:
        if arg in {"-i", "-t", "-it", "-ti"}:
            return arg
        if arg.startswith("-") and not arg.startswith("--") and set("it") & set(arg[1:]):
            return arg
    return None


def _bound_blocking_wait(argv: list[str], timeout_seconds: int) -> list[str]:
    """Give an unbounded blocking command a timeout just under the transport's.

    Left unbounded, ``wait`` and ``rollout status`` block until the transport's own
    read timeout hard-kills the call, so one wrong fix turns every retry into a full
    stall. Capped, kubectl returns its own clean "timed out waiting for the condition"
    shortly before that, which is informative and cheap to retry, while a genuinely slow
    but healthy rollout still has room to finish. An explicit timeout is respected.
    """
    verb, idx = leading_verb(argv)
    blocking = verb == "wait" or (
        verb == "rollout"
        and idx is not None
        and idx + 1 < len(argv)
        and argv[idx + 1].lower() == "status"
    )
    if not blocking or any(a == "--timeout" or a.startswith("--timeout=") for a in argv):
        return argv
    return [*argv, f"--timeout={max(5, timeout_seconds - 10)}s"]


def _append_delete_wait_false(argv: list[str]) -> list[str]:
    """Stop a delete from blocking on finalizers.

    A blocking delete on a finalizer-protected resource wedged one session's server-side
    processing for roughly three quarters of an hour, after which every subsequent
    kubectl in that session failed, reads included.
    """
    verb, _ = leading_verb(argv)
    if verb != "delete" or any(a == "--wait" or a.startswith("--wait=") for a in argv):
        return argv
    return [*argv, "--wait=false"]


_PARAMETERS = {
    "type": "object",
    "properties": {
        "cmd": {
            "type": "string",
            "description": (
                'Full kubectl command, e.g. "kubectl get pods -n shop". '
                'Must start with "kubectl". All-namespace (-A), API discovery, '
                "and credential/context override flags are blocked."
            ),
        },
        "save_as": {
            "type": "string",
            "description": (
                'Optional filename (e.g. "pods.json"). Use only when the '
                "output may be long or you need to reuse/post-process the full "
                "result. When set, the full output is written to "
                "tool_outputs/<save_as> in the session workspace and a short "
                "head + path is returned. Read that file with code_exec or "
                "bash_exec to filter/parse/compute over it — never run kubectl "
                "inside code_exec/bash_exec (they have no cluster credentials)."
            ),
        },
    },
    "required": ["cmd"],
}


def _reject(argv: Sequence[str], policy: KubernetesPolicy) -> str | None:
    """Screen a command before it is classified, returning a refusal or None."""
    operator = _bare_shell_operator(argv)
    if operator is not None:
        return tool_error(
            f"shell operator '{operator}' is not supported — kubectl runs without "
            "a shell, so pipes, redirects, and command chains are passed to kubectl as "
            "arguments. Run the kubectl command alone (e.g. with -o json/jsonpath), then "
            "post-process its saved output with bash_exec or code_exec."
        )
    interactive = _interactive_flag(argv)
    if interactive is not None:
        verb, _ = leading_verb(argv)
        return tool_error(
            f"kubectl {verb} with interactive flags ({interactive}) is not "
            "permitted. Run a bounded, non-interactive diagnostic command instead."
        )
    if classify(argv, policy) == "blocked":
        # Report what was actually refused. The prototype named the verb even when the
        # refusal came from an argument, so a blocked argument told the model its verb
        # was banned and it stopped using a verb that was never the problem.
        offending = next(
            (a for a in argv if policy.denies_flag(a) or policy.denies_argument(a)), None
        )
        if offending is not None:
            return tool_error(f"kubectl argument '{offending}' is not permitted for the agent")
        verb, _ = leading_verb(argv)
        return tool_error(
            f"kubectl '{verb or (argv[0] if argv else 'command')}' is not permitted for the agent"
        )
    return None


@tool(
    name="exec_kubectl_cmd_safely",
    safety="cluster_write",
    requires={"kubernetes"},
    authority=kubectl_authority,
    parameters=_PARAMETERS,
)
async def exec_kubectl_cmd_safely(ctx: ToolContext, cmd: str, save_as: str | None = None) -> str:
    """Run one kubectl command.

    Authority is settled by the approval gate before this is reached; what happens here
    is the screening that applies to every role equally.

    Args:
        ctx: The run's context.
        cmd: The command as the model wrote it.
        save_as: Filename to save the full output under, instead of returning it.

    Returns:
        The command's output, or a refusal beginning with ``error:``.
    """
    argv = tokenize(cmd)
    refusal = _reject(argv, KubernetesPolicy.from_settings(ctx.settings))
    if refusal is not None:
        return refusal

    argv = _append_delete_wait_false(_bound_blocking_wait(argv, ctx.transport.timeout_seconds))
    raw = await ctx.transport.kubectl(argv)
    if save_as:
        return ctx.workspace.save_named_output(redact(ctx.evidence, raw), save_as)
    return deliver(ctx, raw, "kubectl", OUTPUT_LIMIT)


async def read_kubectl(ctx: ToolContext, cmd: str) -> str:
    """Run a read-only kubectl and return its complete output.

    The internal composition path, used by tools that need to parse a full result rather
    than show it to the model, so it deliberately skips the save-and-truncate step. It
    re-classifies rather than trusting its caller, and refuses anything that is not a
    read — including ``exec``, since no internal composition needs an in-container
    command.

    Args:
        ctx: The run's context.
        cmd: The command to run.

    Returns:
        The complete filtered output, or a refusal beginning with ``error:``.
    """
    argv = tokenize(cmd)
    if not argv:
        return tool_error("unsupported kubectl command shape for read-only snapshot")
    if classify(argv, KubernetesPolicy.from_settings(ctx.settings)) != "read":
        return tool_error("read_kubectl only runs read-only kubectl")
    return redact(ctx.evidence, await ctx.transport.kubectl(argv))


_ACTION_BY_AUTHORITY = {
    "read": ActionClass.READ,
    "write": ActionClass.WRITE,
    "exec": ActionClass.EXEC,
    "blocked": ActionClass.BLOCKED,
}

_RESOURCE_ALIASES = {
    "deploy": "deployment",
    "sts": "statefulset",
    "ds": "daemonset",
    "svc": "service",
    "po": "pod",
    "cm": "configmap",
    "ns": "namespace",
}


def _selectors(arguments: Mapping[str, Any]) -> tuple[str, ...]:
    """Describe a call as facets a policy rule can match.

    This is what lets a deployment say "never touch a secret" without the settings layer
    having to know what kubectl is.

    Args:
        arguments: The model's arguments.

    Returns:
        Facets such as ``verb:patch`` or ``resource:secret``.
    """
    argv = tokenize(str(arguments.get("cmd", "")))
    verb, idx = leading_verb(argv)
    found: list[str] = []
    if verb:
        found.append(f"verb:{verb}")
    namespace = _namespace_of(argv)
    if namespace:
        found.append(f"namespace:{namespace}")
    if idx is not None:
        rest = _positional(argv[idx + 1 :])
        if rest:
            head = rest[0].split("/", 1)
            resource = _RESOURCE_ALIASES.get(head[0].rstrip("s"), head[0].rstrip("s"))
            found.append(f"resource:{resource}")
            name = head[1] if len(head) > 1 else (rest[1] if len(rest) > 1 else "")
            if name:
                found.append(f"name:{name}")
    return tuple(found)


def _positional(argv: Sequence[str]) -> list[str]:
    """Drop flags and the values they consume.

    Without this a namespace given as ``-n shop`` reads as the resource being acted on,
    so a rule written against a resource class matches the wrong calls.
    """
    found: list[str] = []
    skip = False
    for arg in argv:
        if arg == "--":
            break  # everything after this is the container's command, not a target
        if skip:
            skip = False
            continue
        if arg.startswith("-"):
            skip = "=" not in arg and arg in _GLOBAL_FLAGS_WITH_VALUE
            continue
        found.append(arg)
    return found


def _namespace_of(argv: Sequence[str]) -> str:
    for i, arg in enumerate(argv):
        if arg in ("-n", "--namespace") and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--namespace="):
            return arg.split("=", 1)[1]
    return ""


def _targets(arguments: Mapping[str, Any]) -> tuple[str, ...]:
    """Name what a call acts on, for an operator reviewing it.

    Args:
        arguments: The model's arguments.

    Returns:
        References such as ``deployment/web`` and ``namespace/shop``.
    """
    facets = dict(part.split(":", 1) for part in _selectors(arguments) if ":" in part)
    named: list[str] = []
    if "resource" in facets:
        resource = facets["resource"]
        named.append(f"{resource}/{facets['name']}" if "name" in facets else resource)
    if "namespace" in facets:
        named.append(f"namespace/{facets['namespace']}")
    return tuple(named)


register_policy(
    "exec_kubectl_cmd_safely",
    ToolPolicy(
        classify=lambda args, tools: _ACTION_BY_AUTHORITY[kubectl_authority(args, tools)],
        write_stages=frozenset({"execution", "single.mitigation"}),
        # The whole command, not a preview. An operator who cannot see the patch body
        # is not reviewing anything.
        summarise=lambda args: str(args.get("cmd", "")),
        selectors=_selectors,
        targets=_targets,
        stage_refusal=lambda stage: (
            f"error: kubectl writes are not permitted for the {stage} agent (read-only)"
        ),
    ),
)
