"""Command-line kubectl shim used by KUBECTL_MODE=proxy_exec.

The shim is intentionally a normal executable target: bash_exec pipelines and
code_exec subprocess calls both resolve `kubectl` through PATH and hit the same
policy. It never uses ambient kubeconfig; every forwarded kubectl command gets a
generated no-credential kubeconfig pointing at the SREGym filtering proxy.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from agent.env import bool_env, float_env
from agent.tools.kubectl import (
    _bound_blocking_wait,
    _classify,
    _is_sensitive_arg,
    _leading_verb,
    _tokenize,
)

_WRITE_AGENTS = frozenset({"execution", "single.mitigation"})
_READ_ONLY_AGENTS = frozenset({"rca", "planning", "learning", "verify"})
_DEFAULT_REAL_KUBECTL = "/usr/local/bin/kubectl.real"
_DEFAULT_TIMEOUT = 120.0


def _err(message: str, code: int = 1) -> int:
    print(f"error: {message}", file=sys.stderr)
    return code


def _proxy_url() -> str:
    return os.environ.get("KUBECTL_PROXY_URL", "").strip().rstrip("/")


def _real_kubectl_path() -> str | None:
    configured = os.environ.get("KUBECTL_REAL_BINARY", "").strip()
    if configured:
        return configured
    if Path(_DEFAULT_REAL_KUBECTL).exists():
        return _DEFAULT_REAL_KUBECTL
    found = shutil.which("kubectl.real")
    if found:
        return found
    return None


def _kubeconfig_path() -> Path:
    configured = os.environ.get("KUBECTL_PROXY_KUBECONFIG", "").strip()
    if configured:
        return Path(configured)
    session_dir = os.environ.get("SESSION_DIR", "").strip()
    base = Path(session_dir) if session_dir else Path(tempfile.gettempdir())
    return base / ".kubectl-proxy" / "config"


def _write_proxy_kubeconfig(proxy_url: str) -> Path:
    path = _kubeconfig_path()
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
    try:
        path.chmod(0o600)
    except OSError:
        pass
    return path


def _check_proxy(proxy_url: str) -> int | None:
    if not bool_env("KUBECTL_PROXY_REQUIRED", True):
        return None
    timeout = float_env(
        "KUBECTL_PROXY_CHECK_TIMEOUT_SECONDS", 2.0, minimum=0.1
    )
    ready_timeout = float_env(
        "KUBECTL_PROXY_READY_TIMEOUT_SECONDS", 10.0, minimum=timeout
    )
    deadline = time.monotonic() + ready_timeout
    last_exc: object | None = None
    while True:
        try:
            with urlopen(f"{proxy_url}/version", timeout=timeout) as resp:
                if resp.status < 400:
                    return None
                return _err(f"KUBECTL_PROXY_URL returned HTTP {resp.status}: {proxy_url}")
        except (OSError, URLError) as exc:
            last_exc = exc
            if time.monotonic() >= deadline:
                return _err(f"KUBECTL_PROXY_URL is unavailable after {ready_timeout:.0f}s: {proxy_url} ({last_exc})")
            time.sleep(0.5)


def _has_wait_flag(args: list[str]) -> bool:
    return any(arg == "--wait" or arg.startswith("--wait=") for arg in args)


def _is_rollout_status(args: list[str]) -> bool:
    verb, idx = _leading_verb(args)
    return (
        verb == "rollout"
        and idx is not None
        and idx + 1 < len(args)
        and args[idx + 1].lower() == "status"
    )


def _is_wait(args: list[str]) -> bool:
    verb, _ = _leading_verb(args)
    return verb == "wait"


def _reject_proxy_watch(args: list[str]) -> int | None:
    if _is_rollout_status(args):
        return _err(
            "kubectl rollout status is not permitted in proxy_exec mode because "
            "it can leave kubectl hanging against the SREGym proxy; use kubectl "
            "get with -o json/jsonpath to check deployment generation, Ready "
            "replicas, pods, endpoints, events, logs, or the incident SLI"
        )
    if _is_wait(args):
        return _err(
            "kubectl wait is not permitted in proxy_exec mode because it uses a "
            "watch request that can wedge the SREGym proxy; use kubectl get "
            "with -o json/jsonpath to check readiness or rollout fields"
        )
    return None


def _reject_interactive_flags(args: list[str]) -> int | None:
    verb, _ = _leading_verb(args)
    if verb in ("run", "exec"):
        for arg in args:
            if arg in ("-i", "-t", "-it", "-ti"):
                return _err(
                    f"kubectl {verb} with interactive flags ({arg}) is not permitted "
                    "in proxy_exec mode because it hangs the non-interactive proxy. "
                    "Run non-interactive commands without -i or -t instead."
                )
            if arg.startswith("-") and not arg.startswith("--"):
                # Check for combined short flags like -it or -itd
                if "i" in arg[1:] or "t" in arg[1:]:
                    return _err(
                        f"kubectl {verb} with interactive flags ({arg}) is not permitted "
                        "in proxy_exec mode because it hangs the non-interactive proxy. "
                        "Run non-interactive commands without -i or -t instead."
                    )
    return None


def _append_delete_wait_false(command: str) -> str:
    args = _tokenize(command)
    verb, _ = _leading_verb(args)
    if verb == "delete" and not _has_wait_flag(args):
        return f"{command} --wait=false"
    return command


def _prepared_command(argv: list[str]) -> tuple[str, list[str], str]:
    command = "kubectl" if not argv else f"kubectl {shlex.join(argv)}"
    args = _tokenize(command)
    kind = _classify(args)
    return command, args, kind


def _authorize(stage: str, args: list[str], kind: str) -> int | None:
    if kind == "blocked":
        verb, _ = _leading_verb(args)
        target = verb or (args[0] if args else "command")
        for arg in args:
            flag = arg.split("=", 1)[0]
            if flag in ("-A", "--all-namespaces"):
                return _err(
                    f"kubectl {target}: {flag} is not permitted in proxy_exec mode — "
                    f"the verb itself is fine; scope the query to one namespace, "
                    f"e.g. kubectl {target} -n <namespace> ..."
                )
            if _is_sensitive_arg(arg):
                return _err(
                    f"kubectl {target}: argument '{flag}' is not permitted in "
                    "proxy_exec mode (credential/context override); rerun without it"
                )
        return _err(f"kubectl '{target}' is not permitted in proxy_exec mode")
    if kind != "write":
        return None
    if stage in _WRITE_AGENTS:
        return None
    shown_stage = stage or "unknown"
    return _err(f"kubectl writes are not permitted for the {shown_stage} agent in proxy_exec mode")


def _exec_timeout() -> float:
    explicit = os.environ.get("KUBECTL_EXEC_TIMEOUT_SECONDS", "").strip()
    if explicit:
        return float_env(
            "KUBECTL_EXEC_TIMEOUT_SECONDS", _DEFAULT_TIMEOUT, minimum=1.0
        )
    return float_env("SHELL_EXEC_TIMEOUT_SECONDS", _DEFAULT_TIMEOUT, minimum=1.0)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    mode = os.environ.get("KUBECTL_MODE", "").strip().lower().replace("-", "_")
    if mode != "proxy_exec" and not os.environ.get("KUBECTL_PROXY_URL", "").strip():
        return _err("kubectl is only available through shell/code when KUBECTL_MODE=proxy_exec")

    command, args, kind = _prepared_command(argv)
    stage = os.environ.get("AGENT_STAGE", "").strip().lower()
    auth_error = _authorize(stage, args, kind)
    if auth_error is not None:
        return auth_error
    watch_error = _reject_proxy_watch(args)
    if watch_error is not None:
        return watch_error

    interactive_error = _reject_interactive_flags(args)
    if interactive_error is not None:
        return interactive_error

    proxy_url = _proxy_url()
    if not proxy_url:
        return _err("KUBECTL_PROXY_URL is required in KUBECTL_MODE=proxy_exec")
    proxy_error = _check_proxy(proxy_url)
    if proxy_error is not None:
        return proxy_error

    real = _real_kubectl_path()
    if not real:
        return _err("real kubectl binary not found; set KUBECTL_REAL_BINARY or install kubectl.real")

    kubeconfig = _write_proxy_kubeconfig(proxy_url)
    bounded = _bound_blocking_wait(_append_delete_wait_false(command))
    real_args = _tokenize(bounded)
    env = dict(os.environ)
    env["KUBECONFIG"] = str(kubeconfig)

    timeout = _exec_timeout()
    try:
        proc = subprocess.Popen(
            [real, *real_args],
            stdin=sys.stdin,
            stdout=sys.stdout,
            stderr=sys.stderr,
            env=env,
        )
        try:
            return proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            print(f"error: kubectl command timed out after {timeout:.0f}s", file=sys.stderr)
            return 124
    except OSError as exc:
        return _err(f"failed to execute real kubectl: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
