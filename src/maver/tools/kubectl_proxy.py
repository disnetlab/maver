"""kubectl against a credential-free HTTP proxy, and the shim that puts it on PATH.

The route is called ``proxy_exec``, which is the name the archived deployment gave it and
the name its refusals say out loud, so the text a model reads here is the text it read
then.

Two callers, one policy. A sandboxed shell or code call resolves ``kubectl`` through
PATH and lands on :func:`main`, which classifies the command, refuses what the route
cannot carry, and hands the rest to the real binary. The transport that serves a role's
own cluster reads call the same preparation through :func:`run_via_proxy`. Both write the
same generated kubeconfig: no credentials, no context, one cluster whose server is the
proxy. Nothing here ever reads an ambient ``KUBECONFIG``, and the environment builder
below removes it, because a sandbox that inherited an operator's kubeconfig would reach
the cluster around the approval gate rather than through it.

The two do not share a process-spawning call, and that is deliberate: the shim inherits
its parent's streams so that ``kubectl apply -f -`` still reads the pipe it was given,
while the transport captures output to return it. What they share — the argument
bounding, the kubeconfig, the environment — is where a divergence would be silent.
"""

import asyncio
import contextlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

from maver.tools.errors import tool_error
from maver.tools.kubernetes import (
    DEFAULT_POLICY,
    _append_delete_wait_false,
    _bound_blocking_wait,
    classify,
    leading_verb,
)
from maver.tools.processes import stop_process_group

__all__ = [
    "PROXY_ROUTE",
    "ProxyRoute",
    "apply_proxy_env",
    "check_proxy",
    "main",
    "run_via_proxy",
    "write_proxy_kubeconfig",
]

PROXY_ROUTE = "proxy_exec"
"""What this route is called wherever a refusal names it.

Carried from the prototype rather than renamed. Every refusal below says "in proxy_exec
mode", and those strings are the interface a model was measured against.
"""

_MODULE = "maver.tools.kubectl_proxy"
"""This module's importable name, for the shim script that re-enters it.

Written down rather than read from ``__name__``, which is ``__main__`` in the process
that runs the shim and would generate a script naming a module that does not exist.
"""

_WRITE_STAGES = frozenset({"execution", "single.mitigation"})
"""The stages that may mutate a cluster.

The same two the kubectl tool's own policy names. A third place that decides this is a
third place to disagree, so if one of them moves the others have to move with it.
"""

_DEFAULT_REAL_KUBECTL = "/usr/local/bin/kubectl.real"
"""Where an image that renamed its kubectl out of the shim's way keeps the real one."""

_DEFAULT_TIMEOUT = 120.0
_SHIM_DIR = ".kubectl-proxy"
_KUBECONFIG_NAME = "config"
_REFUSED = 1
_TIMED_OUT = 124
_REAP_SECONDS = 5.0
"""How long a killed kubectl is waited on before the caller gives up on reaping it."""


@dataclass(frozen=True, slots=True)
class ProxyRoute:
    """Everything a deployment needs to reach its cluster through the proxy.

    One object rather than six arguments, because it travels together: a transport holds
    it, and the sandbox tools read it to build the environment a shell's kubectl runs
    under. The defaults are the configuration the archived campaign ran.
    """

    url: str = "http://127.0.0.1:16444"
    """The credential-free HTTP endpoint a local kubectl reaches the cluster through."""

    required: bool = True
    """Whether the shim waits for the proxy to answer before running anything."""

    check_timeout_s: float = 5.0
    """How long one readiness probe may take."""

    ready_timeout_s: float = 60.0
    """How long probing continues before the shim gives up and refuses."""

    binary: str = "kubectl"
    """The real kubectl, looked up on PATH unless given as a path."""

    exec_timeout_s: float = 120.0
    """How long one kubectl may run."""


def write_proxy_kubeconfig(proxy_url: str, path: Path) -> Path:
    """Write the credential-free kubeconfig kubectl reaches the proxy with.

    No user credentials, no certificate and no context to inherit: the proxy holds the
    cluster's credentials and decides what a request may do, so anything here would only
    be a second answer to a question already settled elsewhere.

    Args:
        proxy_url: The proxy's base URL, which becomes the cluster's server.
        path: Where to write it. Parent directories are created.

    Returns:
        The path written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "apiVersion": "v1",
        "kind": "Config",
        "current-context": "sregym-proxy",
        "clusters": [
            {
                "name": "sregym-proxy",
                "cluster": {
                    "server": proxy_url,
                    "insecure-skip-tls-verify": True,
                },
            }
        ],
        "contexts": [
            {
                "name": "sregym-proxy",
                "context": {"cluster": "sregym-proxy", "user": "sregym-agent"},
            }
        ],
        "users": [{"name": "sregym-agent", "user": {}}],
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    # Best effort: a filesystem that refuses the mode change is not a reason to refuse
    # the command, and the file holds no secret to protect in the first place.
    with contextlib.suppress(OSError):
        path.chmod(0o600)
    return path


def check_proxy(proxy_url: str, *, check_timeout_s: float, ready_timeout_s: float) -> str | None:
    """Wait for the proxy to answer, and say why it did not.

    The wait exists because the proxy is a port-forward into the cluster that is
    restarted between problems, and a kubectl that raced it would report a connection
    refused the model has no way to act on.

    Args:
        proxy_url: The proxy's base URL.
        check_timeout_s: How long one probe may take.
        ready_timeout_s: How long probing continues before giving up.

    Returns:
        None once the proxy answers, or the failure as a string beginning with
        ``error:``, ready to be shown or printed.
    """
    deadline = time.monotonic() + ready_timeout_s
    while True:
        try:
            with urlopen(f"{proxy_url}/version", timeout=check_timeout_s) as response:
                if response.status < 400:
                    return None
                return tool_error(f"KUBECTL_PROXY_URL returned HTTP {response.status}: {proxy_url}")
        except OSError as exc:
            if time.monotonic() >= deadline:
                return tool_error(
                    f"KUBECTL_PROXY_URL is unavailable after {ready_timeout_s:.0f}s: "
                    f"{proxy_url} ({exc})"
                )
            time.sleep(0.5)


async def run_via_proxy(
    argv: Sequence[str],
    *,
    proxy_url: str,
    kubeconfig_dir: Path,
    binary: str,
    timeout_s: float,
) -> tuple[int, str]:
    """Run one kubectl against the proxy and capture what it said.

    Nothing is classified here. Authority is settled before this is reached, and a second
    parse is a second chance to disagree about what the command says. What is added is
    the bounding a blocking command needs, which is a property of the transport rather
    than of the command.

    Args:
        argv: The arguments, without a leading ``kubectl``.
        proxy_url: The proxy's base URL.
        kubeconfig_dir: Directory the generated kubeconfig is written in.
        binary: The real kubectl.
        timeout_s: How long the command may run.

    Returns:
        The exit status and the text that goes with it: a successful command's own
        stdout, byte for byte, and otherwise the reason it failed, unprefixed. Marking a
        failure for the model is the caller's, because a caller that returns raw stdout
        on success has to decide where the prefix goes anyway.
    """
    try:
        command, env = _prepared(
            argv,
            proxy_url=proxy_url,
            kubeconfig=Path(kubeconfig_dir) / _KUBECONFIG_NAME,
            binary=binary,
            timeout_seconds=int(timeout_s),
        )
    except OSError as exc:
        return _REFUSED, f"could not write proxy kubeconfig: {exc}"

    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            # In its own process group, so the timeout below can reach what kubectl
            # started as well as kubectl itself.
            start_new_session=True,
        )
    except OSError as exc:
        return _REFUSED, f"kubectl could not be executed: {exc}"

    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout_s)
    except asyncio.CancelledError:
        await stop_process_group(process)
        raise
    except TimeoutError:
        await stop_process_group(process)
        return _TIMED_OUT, f"kubectl timed out after {timeout_s:.0f}s"

    code = process.returncode or 0
    if code == 0:
        return 0, stdout.decode("utf-8", "replace")
    return code, stderr.decode("utf-8", "replace").strip() or "kubectl failed"


def apply_proxy_env(
    env: dict[str, str],
    *,
    session_dir: Path,
    stage: str,
    proxy_url: str,
    proxy_required: bool,
    check_timeout_s: float,
    ready_timeout_s: float,
    kubectl_binary: str,
    exec_timeout_s: float,
    authority_mode: str,
    kubectl_exec_auto_approved: bool = False,
) -> dict[str, str]:
    """Wire a sandbox subprocess so that its ``kubectl`` is this module's shim.

    A shim on PATH rather than a rule about what may be typed: a shell pipeline, a
    Python ``subprocess`` call and a script the model wrote all resolve the binary the
    same way, so all three land on the same policy without anything having to recognise
    them as kubectl calls first.

    Args:
        env: The subprocess environment, modified in place.
        session_dir: The incident's directory, which holds the shim and its kubeconfig.
        stage: The calling role, which decides whether a write is allowed.
        proxy_url: The proxy's base URL.
        proxy_required: Whether the shim waits for the proxy before running anything.
        check_timeout_s: How long one readiness probe may take.
        ready_timeout_s: How long probing continues before the shim refuses.
        kubectl_binary: The real kubectl, as a path or a name to resolve.
        exec_timeout_s: How long one kubectl may run.
        authority_mode: The deployment's approval mode. Only ``auto`` permits a write
            here, since a command inside a sandbox has no gate in front of it.
        kubectl_exec_auto_approved: Explicit unattended permission for pod execution.
            Otherwise use the structured tool, which can pause for human approval.

    Returns:
        The same mapping, wired.
    """
    values = {
        "KUBECTL_MODE": PROXY_ROUTE,
        "KUBECTL_PROXY_URL": proxy_url.rstrip("/"),
        "KUBECTL_PROXY_REQUIRED": "true" if proxy_required else "false",
        "KUBECTL_PROXY_KUBECONFIG": str(Path(session_dir) / _SHIM_DIR / _KUBECONFIG_NAME),
        "KUBECTL_PROXY_CHECK_TIMEOUT_SECONDS": _seconds(check_timeout_s),
        "KUBECTL_PROXY_READY_TIMEOUT_SECONDS": _seconds(ready_timeout_s),
        "KUBECTL_EXEC_TIMEOUT_SECONDS": _seconds(exec_timeout_s),
        # Resolved here, where PATH still means what it meant before the shim was
        # prepended to it. Left as a bare name it would resolve back to the shim, which
        # would re-enter this module for as long as the process limit allowed.
        "KUBECTL_REAL_BINARY": _resolved_binary(kubectl_binary, env.get("PATH", "")),
        "AGENT_STAGE": stage,
        "MAVER_AUTHORITY_MODE": authority_mode,
        "MAVER_KUBECTL_EXEC_AUTO_APPROVED": "true" if kubectl_exec_auto_approved else "false",
    }
    env.update(values)
    shim_dir = _ensure_shim(session_dir, defaults=values)
    env["PATH"] = f"{shim_dir}{os.pathsep}{env.get('PATH', '')}"
    # An inherited operator kubeconfig would reach the cluster around the proxy and
    # around the gate. The shim writes the credential-free one at execution time.
    env.pop("KUBECONFIG", None)
    return env


def main(argv: list[str] | None = None) -> int:
    """Run one kubectl for a sandboxed shell or code call.

    The entry point of the shim :func:`apply_proxy_env` installs, and runnable on its own
    as ``python -m maver.tools.kubectl_proxy``.

    Args:
        argv: The arguments, without a leading ``kubectl``. Defaults to the process's.

    Returns:
        The command's own exit status, 1 for a refusal or a proxy that never answered,
        and 124 for a command that outlived its bound.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    mode = os.environ.get("KUBECTL_MODE", "").strip().lower().replace("-", "_")
    if mode != PROXY_ROUTE and not _env("KUBECTL_PROXY_URL"):
        return _err("kubectl is only available through shell/code when KUBECTL_MODE=proxy_exec")

    for refusal in (
        _authorisation_refusal(_env("AGENT_STAGE").lower(), args),
        _watch_refusal(args),
        _interactive_refusal(args),
    ):
        if refusal is not None:
            return _err(refusal)

    proxy_url = _env("KUBECTL_PROXY_URL").rstrip("/")
    if not proxy_url:
        return _err("KUBECTL_PROXY_URL is required in KUBECTL_MODE=proxy_exec")
    if _bool_env("KUBECTL_PROXY_REQUIRED", True):
        check_timeout = _float_env("KUBECTL_PROXY_CHECK_TIMEOUT_SECONDS", 2.0, minimum=0.1)
        unavailable = check_proxy(
            proxy_url,
            check_timeout_s=check_timeout,
            ready_timeout_s=_float_env(
                "KUBECTL_PROXY_READY_TIMEOUT_SECONDS", 10.0, minimum=check_timeout
            ),
        )
        if unavailable is not None:
            print(unavailable, file=sys.stderr)
            return _REFUSED

    real = _real_kubectl_path()
    if not real:
        return _err(
            "real kubectl binary not found; set KUBECTL_REAL_BINARY or install kubectl.real"
        )

    timeout = _exec_timeout()
    command, env = _prepared(
        args,
        proxy_url=proxy_url,
        kubeconfig=_kubeconfig_path(),
        binary=real,
        timeout_seconds=int(timeout),
    )
    try:
        # The parent's own streams, so that a command reading a piped manifest still
        # reads it and a long listing reaches the shell as it is produced.
        process = subprocess.Popen(
            command, stdin=sys.stdin, stdout=sys.stdout, stderr=sys.stderr, env=env
        )
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=_REAP_SECONDS)
            print(tool_error(f"kubectl command timed out after {timeout:.0f}s"), file=sys.stderr)
            return _TIMED_OUT
    except OSError as exc:
        return _err(f"failed to execute real kubectl: {exc}")


def _prepared(
    argv: Sequence[str],
    *,
    proxy_url: str,
    kubeconfig: Path,
    binary: str,
    timeout_seconds: int,
) -> tuple[list[str], dict[str, str]]:
    """Build the command line and environment one kubectl runs under.

    The single place the two execution paths agree, so a bound that is applied on one of
    them cannot go missing on the other.
    """
    bounded = _append_delete_wait_false(_bound_blocking_wait(list(argv), timeout_seconds))
    write_proxy_kubeconfig(proxy_url, kubeconfig)
    return [binary, *bounded], {**os.environ, "KUBECONFIG": str(kubeconfig)}


def _authorisation_refusal(stage: str, argv: Sequence[str]) -> str | None:
    """Refuse what this route or this stage may not run.

    The classification is the deployment's shipped one rather than its configured one: a
    shim is a subprocess holding an environment, and settings do not reach it. A
    deployment that has widened or narrowed what kubectl may be asked for still has that
    honoured by the tool the model calls; this path enforces the shipped screens.
    """
    kind = classify(argv, DEFAULT_POLICY)
    if kind == "blocked":
        verb, _ = leading_verb(argv)
        target = verb or (argv[0] if argv else "command")
        for arg in argv:
            flag = arg.split("=", 1)[0]
            if flag in ("-A", "--all-namespaces"):
                return (
                    f"kubectl {target}: {flag} is not permitted in proxy_exec mode — "
                    f"the verb itself is fine; scope the query to one namespace, "
                    f"e.g. kubectl {target} -n <namespace> ..."
                )
            if DEFAULT_POLICY.denies_argument(arg):
                return (
                    f"kubectl {target}: argument '{flag}' is not permitted in "
                    "proxy_exec mode (credential/context override); rerun without it"
                )
        return f"kubectl '{target}' is not permitted in proxy_exec mode"
    if kind == "exec":
        # A subprocess cannot checkpoint an approval request. Never silently turn a
        # required human decision into an unattended call through the shell route.
        if (
            os.environ.get("MAVER_KUBECTL_EXEC_AUTO_APPROVED", "").strip().lower() == "true"
            and os.environ.get("MAVER_AUTHORITY_MODE", "").strip().lower() == "auto"
        ):
            return None
        return (
            "kubectl exec through the shell requires explicit unattended permission; "
            "use exec_kubectl_cmd_safely so the approval gate can authorize the command"
        )
    if kind != "write":
        return None
    if stage not in _WRITE_STAGES:
        return f"kubectl writes are not permitted for the {stage or 'unknown'} agent in proxy_exec mode"
    # The one screen the prototype did not have. Every other write in this system passes
    # the approval gate, and a command inside a sandbox has nothing in front of it, so a
    # deployment that asks a human about its writes must not be written around from here.
    mode = os.environ.get("MAVER_AUTHORITY_MODE", "").strip().lower()
    if mode != "auto":
        return (
            "kubectl writes through the shell are not approval-gated; this deployment "
            f"requires approval (authority.mode={mode or 'unknown'}), so writes are "
            "refused here"
        )
    return None


def _watch_refusal(argv: Sequence[str]) -> str | None:
    """Refuse the verbs that leave kubectl hanging on the proxy, with the advice given."""
    verb, idx = leading_verb(argv)
    if (
        verb == "rollout"
        and idx is not None
        and idx + 1 < len(argv)
        and argv[idx + 1].lower() == "status"
    ):
        return (
            "kubectl rollout status is not permitted in proxy_exec mode because "
            "it can leave kubectl hanging against the SREGym proxy; use kubectl "
            "get with -o json/jsonpath to check deployment generation, Ready "
            "replicas, pods, endpoints, events, logs, or the incident SLI"
        )
    if verb == "wait":
        return (
            "kubectl wait is not permitted in proxy_exec mode because it uses a "
            "watch request that can wedge the SREGym proxy; use kubectl get "
            "with -o json/jsonpath to check readiness or rollout fields"
        )
    return None


def _interactive_refusal(argv: Sequence[str]) -> str | None:
    """Refuse an interactive run or exec, which hangs against a non-interactive proxy."""
    verb, _ = leading_verb(argv)
    if verb not in ("run", "exec"):
        return None
    for arg in argv:
        combined = arg.startswith("-") and not arg.startswith("--") and set("it") & set(arg[1:])
        if arg in ("-i", "-t", "-it", "-ti") or combined:
            return (
                f"kubectl {verb} with interactive flags ({arg}) is not permitted "
                "in proxy_exec mode because it hangs the non-interactive proxy. "
                "Run non-interactive commands without -i or -t instead."
            )
    return None


def _real_kubectl_path() -> str | None:
    """Find the kubectl the shim forwards to, never the shim itself."""
    configured = os.environ.get("KUBECTL_REAL_BINARY", "").strip()
    if configured:
        return configured if os.sep in configured else shutil.which(configured)
    if Path(_DEFAULT_REAL_KUBECTL).exists():
        return _DEFAULT_REAL_KUBECTL
    return shutil.which("kubectl.real")


def _kubeconfig_path() -> Path:
    """Where the shim writes the kubeconfig it hands to the real kubectl."""
    configured = os.environ.get("KUBECTL_PROXY_KUBECONFIG", "").strip()
    if configured:
        return Path(configured)
    session_dir = os.environ.get("SESSION_DIR", "").strip()
    base = Path(session_dir) if session_dir else Path(tempfile.gettempdir())
    return base / _SHIM_DIR / _KUBECONFIG_NAME


def _exec_timeout() -> float:
    """How long the shim lets one kubectl run.

    The cluster bound when there is one, and the shell's otherwise, so a kubectl typed
    into ``bash_exec`` cannot outlive the call that is waiting for it.
    """
    if _env("KUBECTL_EXEC_TIMEOUT_SECONDS"):
        return _float_env("KUBECTL_EXEC_TIMEOUT_SECONDS", _DEFAULT_TIMEOUT, minimum=1.0)
    return _float_env("SHELL_EXEC_TIMEOUT_SECONDS", _DEFAULT_TIMEOUT, minimum=1.0)


def _ensure_shim(session_dir: Path, *, defaults: Mapping[str, str]) -> Path:
    """Write the per-session ``kubectl`` script, and return the directory holding it.

    The script re-exports its own defaults before running, so a command that clears the
    environment — a ``sudo``, an ``env -i``, a script sourcing a clean profile — still
    reaches the proxy with the stage it was issued under rather than silently losing the
    route and falling back to a kubectl with no credentials at all.
    """
    base = Path(session_dir) / _SHIM_DIR / "bin"
    base.mkdir(parents=True, exist_ok=True)
    shim = base / "kubectl"
    script = (
        "#!/bin/sh\n"
        f"{_default_exports(defaults)}"
        f'exec {shlex.quote(sys.executable)} -m {_MODULE} "$@"\n'
    )
    if not shim.exists() or shim.read_text(encoding="utf-8") != script:
        shim.write_text(script, encoding="utf-8")
        shim.chmod(0o755)
    return base


def _default_exports(defaults: Mapping[str, str]) -> str:
    """Render the shell lines that restore a value the caller's environment dropped."""
    lines: list[str] = []
    for key, value in defaults.items():
        if not value:
            continue
        lines.extend(
            [
                f'if [ -z "${{{key}:-}}" ]; then',
                f"  {key}={shlex.quote(value)}",
                "fi",
                f"export {key}",
            ]
        )
    return "\n".join(lines) + "\n" if lines else ""


def _resolved_binary(name: str, path: str) -> str:
    """Resolve a bare kubectl name against a PATH the shim is not yet on."""
    if os.sep in name:
        return name
    return shutil.which(name, path=path or os.defpath) or name


def _seconds(value: float) -> str:
    """Render a timeout for an environment variable without a trailing ``.0``."""
    return f"{value:g}"


def _err(message: str, code: int = _REFUSED) -> int:
    """Print a refusal where a shell command's caller will read it."""
    print(tool_error(message), file=sys.stderr)
    return code


def _env(name: str) -> str:
    return os.environ.get(name, "").strip()


def _bool_env(name: str, default: bool) -> bool:
    value = _env(name).lower()
    return default if not value else value not in ("false", "0", "no", "off", "disabled")


def _float_env(name: str, default: float, *, minimum: float) -> float:
    try:
        return max(minimum, float(_env(name)))
    except ValueError:
        return default


if __name__ == "__main__":
    raise SystemExit(main())
