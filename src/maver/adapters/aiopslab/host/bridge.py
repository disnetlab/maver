"""The loopback bridge: the only cluster authority the agent container is given.

The agent runs with no kubeconfig and no shell on the benchmark host. Everything it
does to the cluster arrives here as one ``POST /v1/action``, and every containment
property the deployment claims is a check in this file rather than a promise about the
model: a bearer token compared in constant time, a namespace known from
``init_problem`` and never from the request, a flag denylist, a refusal to run
``kubectl exec``, and ``argv`` executed without a shell.

One check is new in MAVER. ``allow_writes`` is derived from the task type before the
episode starts, so a read-only graded task cannot write whatever the request claims its
stage to be. The prototype gated the two extra actions on a client-supplied ``stage``
alone, which re-reads a claim rather than deriving a fact; that check stays as a second,
weaker one.

So is the access log. The prototype's handler discarded it, so the process holding the
kubeconfig kept no record of what it was asked to do and the agent's own audit log was
the only account of every cluster action. :class:`AccessLog` records receipt before
dispatch and completion even after a client disconnect, with complete host-private
request and response artifacts linked to the agent invocation.
"""

import hmac
import json
import logging
import os
import re
import shlex
import socket
import subprocess
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast
from urllib.error import HTTPError
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import Request, urlopen
from uuid import uuid4

from .mapping import _bounded_int

__all__ = ["AccessLog", "BridgeBackend", "BridgeHTTPServer", "serve_bridge"]

_log = logging.getLogger(__name__)

_SESSION_HEADER = "X-MAVER-Session"
"""What the agent's transport sends so its audit events join these lines.

Spelled out rather than imported: the agent half that sends it is not importable here.
Without the header the two logs share only a wall clock, and a timestamp join mis-mapped
five sessions of six in one measured case.
"""

_BLOCKED_KUBECTL_FLAGS = {
    "-A",
    "--all-namespaces",
    "--as",
    "--as-group",
    "--certificate-authority",
    "--client-certificate",
    "--client-key",
    "--cluster",
    "--context",
    "--kubeconfig",
    "--raw",
    "--server",
    "--tls-server-name",
    "--token",
    "--user",
}
_WRITE_VERBS = {
    "annotate",
    "apply",
    "autoscale",
    "cordon",
    "create",
    "delete",
    "drain",
    "edit",
    "expose",
    "label",
    "patch",
    "replace",
    "rollout",
    "run",
    "scale",
    "set",
    "taint",
    "uncordon",
}
_MONGODB_RECOVERY_SCRIPTS = {
    ("mongodb-rate", "privileges"): "/scripts/revoke-mitigate-admin-rate-mongo.sh",
    ("mongodb-geo", "privileges"): "/scripts/revoke-mitigate-admin-geo-mongo.sh",
    ("mongodb-rate", "user"): "/scripts/remove-mitigate-admin-rate-mongo.sh",
    ("mongodb-geo", "user"): "/scripts/remove-mitigate-admin-geo-mongo.sh",
}
_SERVICE_NAME = re.compile(r"[a-z0-9](?:[-a-z0-9]{0,61}[a-z0-9])?$")


def _namespace_args(argv: list[str]) -> list[str]:
    values: list[str] = []
    for index, arg in enumerate(argv):
        if arg in {"-n", "--namespace"} and index + 1 < len(argv):
            values.append(argv[index + 1])
        elif arg.startswith("--namespace="):
            values.append(arg.split("=", 1)[1])
    return values


def _kubectl_verb(argv: list[str]) -> str:
    skip_value = {"-n", "--namespace", "--request-timeout"}
    index = 1
    while index < len(argv):
        arg = argv[index]
        if arg in skip_value:
            index += 2
            continue
        if arg.startswith("--namespace=") or arg.startswith("--request-timeout="):
            index += 1
            continue
        if arg.startswith("-"):
            index += 1
            continue
        return arg.lower()
    return ""


class BridgeBackend:
    """Defence-in-depth dispatcher for one active AIOpsLab problem."""

    def __init__(
        self,
        namespace: str,
        *,
        allow_writes: bool = False,
        prometheus_url: str = "http://127.0.0.1:32000",
    ):
        """Bind the bridge to one problem.

        Args:
            namespace: The namespace ``init_problem`` deployed into.
            allow_writes: Whether this task is graded on remediation. Defaults to
                refusing, so a task type nobody thought about is read-only.
            prometheus_url: AIOpsLab's Prometheus, on its NodePort.
        """
        self.namespace = namespace
        self.allow_writes = allow_writes
        self.prometheus_url = prometheus_url.rstrip("/")
        self._jaeger_process: subprocess.Popen[str] | None = None
        self._jaeger_base_url: str | None = None
        self._jaeger_lock = threading.Lock()

    def dispatch(self, payload: dict[str, Any]) -> str:
        """Route one bridge request to the single action it names.

        Args:
            payload: The decoded request body.

        Returns:
            The tool's output, or a string beginning ``error:`` for a failure the agent
            is meant to see and reason about.

        Raises:
            ValueError: For a protocol error or a refused request, which the handler
                returns as a 400.
        """
        requested_namespace = str(payload.get("namespace") or "")
        if requested_namespace != self.namespace:
            raise ValueError(
                f"incident namespace {requested_namespace!r} does not match active "
                f"AIOpsLab namespace {self.namespace!r}"
            )
        server = str(payload.get("server") or "")
        tool = str(payload.get("tool") or "")
        arguments = payload.get("arguments")
        if not isinstance(arguments, dict):
            raise ValueError("arguments must be an object")

        if server == "kubectl" and tool == "exec_kubectl_cmd_safely":
            return self._exec_kubectl(str(arguments.get("cmd") or ""))
        if server == "prometheus" and tool == "get_metrics":
            return self._prometheus("/api/v1/query", {"query": str(arguments.get("query") or "")})
        if server == "prometheus" and tool == "get_alerts":
            return self._prometheus("/api/v1/alerts")
        if server == "jaeger" and tool == "get_services":
            return self._jaeger_query("/api/services", {}, data_only=True)
        if server == "jaeger" and tool == "get_operations":
            service = str(arguments.get("service") or "").strip()
            if not service:
                raise ValueError("Jaeger service must not be empty")
            return self._jaeger_query(
                f"/api/services/{quote(service, safe='')}/operations",
                {},
                data_only=True,
            )
        if server == "jaeger" and tool == "get_traces":
            service = str(arguments.get("service") or "").strip()
            if not service:
                raise ValueError("Jaeger service must not be empty")
            minutes = _bounded_int(arguments.get("last_n_minutes"), default=15, low=1, high=120)
            return self._jaeger_query(
                "/api/traces",
                {"service": service, "lookback": f"{minutes * 60}s", "limit": "20"},
                data_only=True,
            )
        if server == "jaeger" and tool == "get_dependency_graph":
            minutes = _bounded_int(arguments.get("last_n_minutes"), default=30, low=1, high=120)
            return self._jaeger_query(
                "/api/dependencies",
                {
                    "endTs": str(int(time.time() * 1000)),
                    "lookback": str(minutes * 60 * 1000),
                },
                data_only=False,
            )
        if server == "aiopslab" and tool == "restore_mongodb_access":
            self._require_writes("MongoDB recovery")
            if str(payload.get("stage") or "") != "execution":
                raise ValueError("MongoDB recovery is permitted only during the execution stage")
            return self._restore_mongodb_access(
                service=str(arguments.get("service") or ""),
                access=str(arguments.get("access") or ""),
            )
        if server == "aiopslab" and tool == "probe_service":
            self._require_writes("Service probing")
            if str(payload.get("stage") or "") != "verify":
                raise ValueError("Service probing is permitted only during the verify stage")
            return self._probe_service(
                service=str(arguments.get("service") or ""),
                path=str(arguments.get("path") or ""),
                port=arguments.get("port"),
            )
        raise ValueError(f"unsupported bridge tool: {server}.{tool}")

    def _require_writes(self, what: str) -> None:
        """Refuse anything beyond reading on a task graded read-only.

        Derived from the task type before the episode began, so unlike the ``stage``
        check below it does not depend on what the requesting process says about itself.
        """
        if not self.allow_writes:
            raise ValueError(f"{what} is not permitted: this AIOpsLab task is graded read-only")

    def _exec_kubectl(self, command: str) -> str:
        try:
            argv = shlex.split(command)
        except ValueError as exc:
            raise ValueError(f"invalid kubectl quoting: {exc}") from exc
        if not argv or argv[0] != "kubectl":
            raise ValueError("bridge accepts only commands beginning with kubectl")
        for arg in argv[1:]:
            flag = arg.split("=", 1)[0]
            if flag in _BLOCKED_KUBECTL_FLAGS:
                raise ValueError(f"kubectl flag {flag!r} is not permitted")

        namespaces = _namespace_args(argv)
        if len(set(namespaces)) > 1 or any(ns != self.namespace for ns in namespaces):
            raise ValueError("kubectl command escapes the active problem namespace")
        verb = _kubectl_verb(argv)
        if verb == "exec":
            raise ValueError(
                "kubectl exec is not permitted by the AIOpsLab bridge because "
                "an in-container command cannot be proven read-only"
            )
        if verb in _WRITE_VERBS:
            self._require_writes(f"kubectl {verb}")
            if not namespaces:
                raise ValueError("kubectl writes must explicitly name the active namespace")
        if not namespaces:
            # Prevent a namespaced read from silently using the kubeconfig's
            # default namespace. Cluster-scoped resources ignore this flag.
            argv.extend(["--namespace", self.namespace])

        timeout = 120
        try:
            proc = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return f"error: kubectl timed out after {timeout}s"
        output = (proc.stdout or "") + (proc.stderr or "")
        output = output.strip()
        if proc.returncode:
            return f"error: {output or f'kubectl exited {proc.returncode}'}"
        return output or "(kubectl completed with no output)"

    def _prometheus(self, path: str, params: dict[str, str] | None = None) -> str:
        url = f"{self.prometheus_url}{path}"
        if params:
            url += "?" + urlencode(params)
        try:
            with urlopen(url, timeout=30) as response:  # fixed loopback base URL
                body = json.loads(response.read().decode("utf-8", "replace"))
        except Exception as exc:  # returned to the agent as a tool error
            return f"error: Prometheus query failed: {type(exc).__name__}: {exc}"
        if not isinstance(body, dict) or body.get("status") != "success":
            return f"error: Prometheus returned {body!r}"
        return json.dumps(body.get("data"), separators=(",", ":"), default=str)

    def _probe_service(self, *, service: str, path: str, port: Any) -> str:
        """Make one bounded HTTP request through an active-namespace Service.

        This deliberately does not expose a generic command runner.  The only
        network destination is the selected Kubernetes Service, reached through
        a short-lived localhost-only ``kubectl port-forward``.  The optional
        requested port must appear in that Service's own spec.
        """
        if not _SERVICE_NAME.fullmatch(service):
            raise ValueError("service must be a DNS-1123 Service name")
        parsed = urlsplit(path)
        if (
            not path.startswith("/")
            or path.startswith("//")
            or len(path) > 256
            or parsed.scheme
            or parsed.netloc
            or parsed.fragment
            or "\r" in path
            or "\n" in path
        ):
            raise ValueError("path must be a bounded absolute HTTP path")

        raw_service = self._run(
            ["kubectl", "get", "service", service, "-n", self.namespace, "-o", "json"],
            timeout=60,
        )
        if raw_service.startswith("error:"):
            return raw_service
        try:
            service_spec = json.loads(raw_service).get("spec", {})
            service_ports = service_spec.get("ports", [])
            if not isinstance(service_ports, list):
                raise ValueError("Service spec.ports is not a list")
            declared = [
                item
                for item in service_ports
                if isinstance(item, dict)
                and isinstance(item.get("port"), int)
                and 1 <= item["port"] <= 65535
            ]
        except (AttributeError, TypeError, ValueError, json.JSONDecodeError) as exc:
            return f"error: could not parse Service {service!r}: {exc}"
        if not declared:
            return f"error: Service {service!r} has no declared TCP port"

        if port is None:
            # Prefer named/app-protocol HTTP ports; fall back deterministically to
            # the first declared port so the model does not control destinations.
            selected = next(
                (
                    item
                    for item in declared
                    if str(item.get("appProtocol") or "").lower() in {"http", "https"}
                    or "http" in str(item.get("name") or "").lower()
                ),
                declared[0],
            )
        else:
            if isinstance(port, bool) or not isinstance(port, int):
                raise ValueError("port must be an integer")
            found = next((item for item in declared if item["port"] == port), None)
            if found is None:
                raise ValueError("port must be declared by the selected Service")
            selected = found
        service_port = selected["port"]

        local_port = self._allocate_loopback_port()

        process = subprocess.Popen(
            [
                "kubectl",
                "port-forward",
                f"service/{service}",
                f"{local_port}:{service_port}",
                "-n",
                self.namespace,
                "--address",
                "127.0.0.1",
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        try:
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    return "error: Service port-forward exited before becoming ready"
                try:
                    with socket.create_connection(("127.0.0.1", local_port), timeout=0.2):
                        break
                except OSError:
                    time.sleep(0.1)
            else:
                return "error: Service port-forward did not become ready within 15s"

            endpoint = f"http://127.0.0.1:{local_port}{path}"
            request = Request(endpoint, headers={"Accept": "*/*"}, method="GET")
            try:
                with urlopen(request, timeout=20) as response:  # fixed loopback URL
                    status = int(response.status)
            except HTTPError as exc:
                status = int(exc.code)
            except Exception as exc:  # tool failures remain observable
                return (
                    f"error: HTTP probe via service/{service}:{service_port} returned no HTTP "
                    f"response ({type(exc).__name__}: {exc}). This may be an internal "
                    "non-HTTP protocol, not the user-facing entry point. Choose an "
                    "Ingress-, gateway-, or frontend-facing HTTP Service from live "
                    "Kubernetes evidence; do not claim recovery without an HTTP status."
                )
            return f"HTTP {status} via service/{service}:{service_port}{path}"
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)

    @staticmethod
    def _allocate_loopback_port() -> int:
        """Reserve a fresh loopback port briefly before starting port-forward."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])

    def _restore_mongodb_access(self, *, service: str, access: str) -> str:
        script = _MONGODB_RECOVERY_SCRIPTS.get((service, access))
        if script is None:
            raise ValueError("unsupported MongoDB recovery selection")

        inventory = self._run(
            ["kubectl", "get", "pods", "-n", self.namespace, "-o", "json"],
            timeout=60,
        )
        if inventory.startswith("error:"):
            return inventory
        try:
            body = json.loads(inventory)
            pod_names = sorted(
                str(item.get("metadata", {}).get("name") or "")
                for item in body.get("items", [])
                if isinstance(item, dict)
                and str(item.get("metadata", {}).get("name") or "").startswith(f"{service}-")
            )
        except (AttributeError, TypeError, ValueError, json.JSONDecodeError) as exc:
            return f"error: could not parse MongoDB pod inventory: {exc}"
        if not pod_names:
            return f"error: no pod for {service!r} exists in namespace {self.namespace!r}"

        repaired = self._run(
            [
                "kubectl",
                "exec",
                pod_names[0],
                "-n",
                self.namespace,
                "--",
                "/bin/bash",
                script,
            ],
            timeout=120,
        )
        if repaired.startswith("error:"):
            return repaired
        application = service.removeprefix("mongodb-")
        restarted = self._run(
            [
                "kubectl",
                "rollout",
                "restart",
                f"deployment/{application}",
                "-n",
                self.namespace,
            ],
            timeout=120,
        )
        if restarted.startswith("error:"):
            return f"{repaired}\n{restarted}"
        return f"{repaired}\n{restarted}".strip()

    def _run(self, argv: list[str], *, timeout: int) -> str:
        try:
            proc = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return f"error: command timed out after {timeout}s"
        except OSError as exc:
            return f"error: command failed to start: {exc}"
        output = ((proc.stdout or "") + (proc.stderr or "")).strip()
        if proc.returncode:
            return f"error: {output or f'command exited {proc.returncode}'}"
        return output or "(command completed with no output)"

    def _jaeger_query(
        self,
        path: str,
        params: dict[str, str],
        *,
        data_only: bool,
    ) -> str:
        try:
            base_url = self._ensure_jaeger_port_forward()
        except Exception as exc:  # returned to the agent as a tool error
            return f"error: Jaeger port-forward failed: {type(exc).__name__}: {exc}"
        url = f"{base_url}{path}"
        if params:
            url += "?" + urlencode(params)
        try:
            with urlopen(url, timeout=30) as response:  # loopback URL
                body = json.loads(response.read().decode("utf-8", "replace"))
        except Exception as exc:  # returned to the agent as a tool error
            return f"error: Jaeger query failed: {type(exc).__name__}: {exc}"
        if not isinstance(body, dict):
            return f"error: Jaeger returned {body!r}"
        result = body.get("data") if data_only else body
        return json.dumps(result, separators=(",", ":"), default=str)

    def _ensure_jaeger_port_forward(self) -> str:
        with self._jaeger_lock:
            if (
                self._jaeger_base_url
                and self._jaeger_process is not None
                and self._jaeger_process.poll() is None
            ):
                return self._jaeger_base_url

            self.close()
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(("127.0.0.1", 0))
                port = int(sock.getsockname()[1])

            target = "service/jaeger"
            path_prefix = ""
            if self.namespace == "astronomy-shop":
                pod = self._run(
                    [
                        "kubectl",
                        "get",
                        "pods",
                        "-n",
                        self.namespace,
                        "-l",
                        "app.kubernetes.io/name=jaeger",
                        "-o",
                        "jsonpath={.items[0].metadata.name}",
                    ],
                    timeout=60,
                )
                if pod.startswith("error:") or not pod.strip():
                    raise RuntimeError(pod)
                target = f"pod/{pod.strip()}"
                path_prefix = "/jaeger/ui"

            self._jaeger_process = subprocess.Popen(
                [
                    "kubectl",
                    "port-forward",
                    target,
                    f"{port}:16686",
                    "-n",
                    self.namespace,
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                if self._jaeger_process.poll() is not None:
                    raise RuntimeError("kubectl port-forward exited before becoming ready")
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                        self._jaeger_base_url = f"http://127.0.0.1:{port}{path_prefix}"
                        return self._jaeger_base_url
                except OSError:
                    time.sleep(0.1)
            raise TimeoutError("kubectl port-forward did not become ready within 15s")

    def close(self) -> None:
        """Tear down the Jaeger port-forward this backend opened, if any."""
        process = self._jaeger_process
        self._jaeger_process = None
        self._jaeger_base_url = None
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class AccessLog:
    """Append-only request lifecycle records and complete private payload artifacts.

    Files belong to the host process, never the agent workspace. Only operation
    payloads are captured; bearer tokens and HTTP headers are excluded.
    """

    def __init__(self, path: Path | str):
        """Create a private log and artifact directory beside it."""
        self._path = Path(path)
        self._lock = threading.Lock()
        self._artifacts = self._path.with_name(self._path.name + ".artifacts")
        self._artifacts.mkdir(mode=0o700, exist_ok=True)
        if self._artifacts.is_symlink():
            raise ValueError("bridge artifact directory must not be a symbolic link")
        self._artifacts.chmod(0o700)
        descriptor = os.open(
            self._path, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600
        )
        try:
            os.fchmod(descriptor, 0o600)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _sync_directory(self._path.parent)

    def capture(self, receipt_id: str, label: str, content: str) -> dict[str, Any]:
        """Save an entire payload, addressed only by a server-generated identity."""
        path = self._artifacts / f"{receipt_id}.{label}"
        encoded = content.encode("utf-8")
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        _sync_directory(self._artifacts)
        return {
            "path": str(path.relative_to(self._path.parent)),
            "bytes": len(encoded),
            "sha256": sha256(encoded).hexdigest(),
        }

    def record(self, **fields: Any) -> None:
        """Append one lifecycle event, closing the file before returning."""
        line = json.dumps(fields, separators=(",", ":"), default=str)
        with self._lock:
            descriptor = os.open(self._path, os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW)
            with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
                handle.write(f"{line}\n")
                handle.flush()
                os.fsync(handle.fileno())
            _sync_directory(self._path.parent)


class _BridgeHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        """Answer a liveness probe, so a dead bridge is told apart from a refused action."""
        if self.path != "/healthz":
            self.send_error(404)
            return
        self._json(200, {"ok": True, "namespace": self._server().backend.namespace})

    def do_POST(self) -> None:
        """Authenticate, record, dispatch and finish independently of the client."""
        if self.path != "/v1/action":
            self.send_error(404)
            return
        started = time.monotonic()
        receipt_id = uuid4().hex
        payload: dict[str, Any] = {}
        request: dict[str, Any] | None = None
        receipt_attempted = False
        audit_errors: list[str] = []
        result, decision, status = "bridge action did not finish", "failed", 500
        access_log = self._server().access_log

        def audit_failure(phase: str, exc: Exception) -> None:
            detail = f"{phase}: {type(exc).__name__}: {exc}"
            audit_errors.append(detail)
            _log.error("bridge audit recording failed: %s", detail)

        def record_received() -> bool:
            nonlocal request, receipt_attempted
            receipt_attempted = True
            try:
                if access_log is not None:
                    request = access_log.capture(
                        receipt_id, "request.json", json.dumps(payload, sort_keys=True)
                    )
                self._record(
                    started, payload, receipt_id, "received", None, "received", request=request
                )
                return True
            except Exception as exc:
                audit_failure("receipt", exc)
                return False

        try:
            supplied = self.headers.get("Authorization", "")
            expected = f"Bearer {self._server().token}"
            if not hmac.compare_digest(supplied, expected):
                result, decision, status = "unauthorized", "unauthorized", 401
            else:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 1_000_000:
                    raise ValueError("invalid request size")
                decoded = json.loads(self.rfile.read(length))
                if not isinstance(decoded, dict):
                    raise ValueError("request body must be an object")
                # Store only the action contract, never authentication or arbitrary
                # extra body fields a caller may have attached to it.
                payload = {
                    key: decoded[key]
                    for key in ("server", "tool", "namespace", "stage", "arguments")
                    if key in decoded
                }
                if record_received():
                    result = self._server().backend.dispatch(payload)
                    decision = "failed" if result.startswith("error:") else "served"
                    status = 200
                else:
                    result = f"bridge audit receipt failed: {audit_errors[-1]}"
                    decision, status = "failed", 500
        except Exception as exc:
            result, decision, status = str(exc), "refused", 400
        finally:
            # Never retry an exclusive artifact name after a partially successful
            # write: preserve the original failure and the actual dispatch outcome.
            if not receipt_attempted:
                record_received()
            result_artifact = None
            if access_log is not None:
                try:
                    result_artifact = access_log.capture(receipt_id, "response.txt", result)
                except Exception as exc:
                    audit_failure("result", exc)
            delivery_error = None
            try:
                self._json(status, {"ok": decision == "served", "result": result})
            except OSError as exc:
                # A disconnected client changes delivery, not whether dispatch ran.
                delivery_error = f"{type(exc).__name__}: {exc}"
            finally:
                try:
                    self._record(
                        started,
                        payload,
                        receipt_id,
                        "finished",
                        status,
                        decision,
                        request=request,
                        result=result_artifact,
                        result_preview=result[:2048],
                        result_chars=len(result),
                        result_preview_truncated=len(result) > 2048,
                        error=result[:2048] if decision != "served" else None,
                        delivery_error=delivery_error,
                        audit_errors=audit_errors,
                    )
                except Exception as exc:
                    audit_failure("completion", exc)

    def log_message(self, _format: str, *_args: Any) -> None:
        """Keep the harness's console for the episode, not for every tool call."""
        return

    def _record(
        self,
        started: float,
        payload: dict[str, Any],
        receipt_id: str,
        event: str,
        status: int | None,
        decision: str,
        **evidence: Any,
    ) -> None:
        """Join the host's evidence to one invocation and backend request."""
        access_log = self._server().access_log
        if access_log is None:
            return
        server = str(payload.get("server") or "")
        tool = str(payload.get("tool") or "")
        arguments = json.dumps(payload.get("arguments") or {}, sort_keys=True)
        access_log.record(
            schema_version=2,
            event=event,
            receipt_id=receipt_id,
            timestamp=datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            session=self.headers.get(_SESSION_HEADER, ""),
            request_id=self.headers.get("X-MAVER-Request-ID", "") or receipt_id,
            invocation_id=self.headers.get("X-MAVER-Invocation-ID", ""),
            call_id=self.headers.get("X-MAVER-Call-ID", ""),
            run_id=self.headers.get("X-MAVER-Run-ID", ""),
            stage=str(payload.get("stage") or ""),
            port=int(self._server().server_address[1]),
            op=f"{server}.{tool}" if server or tool else "",
            namespace=str(payload.get("namespace") or ""),
            arguments_preview=arguments[:2048],
            arguments_preview_truncated=len(arguments) > 2048,
            decision=decision,
            status=status,
            duration_ms=round((time.monotonic() - started) * 1000, 1),
            **evidence,
        )

    def _server(self) -> "BridgeHTTPServer":
        return cast("BridgeHTTPServer", self.server)

    def _json(self, status: int, body: dict[str, Any]) -> None:
        encoded = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


class BridgeHTTPServer(ThreadingHTTPServer):
    """The bridge's server, carrying the one backend and token its handlers read."""

    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        backend: BridgeBackend,
        token: str,
        access_log: AccessLog | None = None,
    ):
        """Bind the server.

        Args:
            address: Host and port to bind. Loopback in every deployment.
            backend: The dispatcher for the active problem.
            token: The bearer token every action must present.
            access_log: Where to record each action. None keeps the prototype's
                behaviour of recording nothing, which is what the unit tests want.
        """
        super().__init__(address, _BridgeHandler)
        self.backend = backend
        self.token = token
        self.access_log = access_log


@contextmanager
def serve_bridge(
    backend: BridgeBackend,
    token: str,
    *,
    host: str = "127.0.0.1",
    port: int = 1819,
    access_log: Path | str | None = None,
    poll_interval: float = 0.5,
) -> Iterator[BridgeHTTPServer]:
    """Serve the bridge for the lifetime of one problem.

    Args:
        backend: The dispatcher for the active problem.
        token: The bearer token every action must present.
        host: Address to bind.
        port: Port to bind. Zero takes an ephemeral one, which tests use.
        access_log: Lifecycle log with private payload files beside it. Its directory must exist.
        poll_interval: How often the serving thread checks for shutdown, in seconds.
            The default is the standard library's; shutdown waits up to one interval,
            so tests that open a bridge per case lower it.

    Yields:
        The running server.

    Raises:
        ValueError: If no token was supplied, which would leave the cluster reachable
            by anything that can open a loopback socket.
    """
    if not token:
        raise ValueError("bridge bearer token must not be empty")
    server = BridgeHTTPServer(
        (host, port), backend, token, AccessLog(access_log) if access_log else None
    )
    thread = threading.Thread(
        target=server.serve_forever,
        kwargs={"poll_interval": poll_interval},
        name="aiopslab-bridge",
        daemon=True,
    )
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        backend.close()
