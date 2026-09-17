"""The bridge is the containment claim, so its refusals are the tests.

Everything here runs without a cluster: the two places that would touch one,
`subprocess.run` and `subprocess.Popen`, are replaced, and the one test that starts a
real server binds an ephemeral loopback port and talks to itself.
"""

import io
import json
import subprocess
from http.client import RemoteDisconnected
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from maver.adapters.aiopslab.host import bridge as bridge_module
from maver.adapters.aiopslab.host.bridge import BridgeBackend, serve_bridge

NAMESPACE = "test-hotel-reservation"


class FakeRun:
    """Stands in for `subprocess.run`, recording what would have been executed."""

    def __init__(self, *results: subprocess.CompletedProcess[str]) -> None:
        self._results = list(results)
        self.calls: list[tuple[Any, dict[str, Any]]] = []

    def __call__(self, argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        self.calls.append((argv, kwargs))
        return self._results.pop(0)


def completed(stdout: str = "", returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr="")


def kubectl(cmd: str, namespace: str = NAMESPACE) -> dict[str, Any]:
    return {
        "namespace": namespace,
        "server": "kubectl",
        "tool": "exec_kubectl_cmd_safely",
        "arguments": {"cmd": cmd},
    }


@pytest.fixture
def backend() -> BridgeBackend:
    return BridgeBackend(NAMESPACE, allow_writes=True)


@pytest.mark.parity
def test_a_request_for_another_namespace_is_refused(backend: BridgeBackend) -> None:
    with pytest.raises(ValueError, match="does not match"):
        backend.dispatch(kubectl("kubectl get pods", namespace="other"))


@pytest.mark.parity
def test_an_explicit_namespace_escape_is_refused(backend: BridgeBackend) -> None:
    with pytest.raises(ValueError, match="escapes"):
        backend.dispatch(kubectl("kubectl get pods -n kube-system"))


@pytest.mark.parity
def test_a_second_namespace_is_refused(backend: BridgeBackend) -> None:
    with pytest.raises(ValueError, match="escapes"):
        backend.dispatch(kubectl(f"kubectl get pods -n {NAMESPACE} -n kube-system"))


@pytest.mark.parity
def test_a_blocked_flag_is_refused(backend: BridgeBackend) -> None:
    with pytest.raises(ValueError, match="is not permitted"):
        backend.dispatch(kubectl("kubectl get pods --all-namespaces"))
    with pytest.raises(ValueError, match="is not permitted"):
        backend.dispatch(kubectl("kubectl get pods --kubeconfig=/etc/admin.conf"))


@pytest.mark.parity
def test_exec_is_refused_because_the_inner_command_cannot_be_classified(
    backend: BridgeBackend,
) -> None:
    with pytest.raises(ValueError, match="cannot be proven read-only"):
        backend.dispatch(kubectl(f"kubectl exec pod/x -n {NAMESPACE} -- touch /tmp/x"))


@pytest.mark.parity
def test_a_write_without_an_explicit_namespace_is_refused(backend: BridgeBackend) -> None:
    with pytest.raises(ValueError, match="writes must explicitly"):
        backend.dispatch(kubectl("kubectl scale deployment/x --replicas=1"))


@pytest.mark.parity
def test_a_namespace_less_read_is_scoped_to_the_active_namespace(
    backend: BridgeBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = FakeRun(completed("pod/x\n"))
    monkeypatch.setattr(bridge_module.subprocess, "run", run)

    backend.dispatch(kubectl("kubectl get pods"))

    assert run.calls[0][0] == ["kubectl", "get", "pods", "--namespace", NAMESPACE]


@pytest.mark.parity
def test_the_command_is_executed_without_a_shell(
    backend: BridgeBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = FakeRun(completed("pod/x\n"))
    monkeypatch.setattr(bridge_module.subprocess, "run", run)

    result = backend.dispatch(kubectl(f"kubectl get pods -n {NAMESPACE}"))

    argv, kwargs = run.calls[0]
    assert result == "pod/x"
    assert argv == ["kubectl", "get", "pods", "-n", NAMESPACE]
    assert "shell" not in kwargs


@pytest.mark.parametrize("verb", ["delete", "patch", "scale", "apply", "rollout", "cordon"])
def test_a_read_only_task_refuses_every_write_verb(verb: str) -> None:
    """Derived from the task type before the episode began, so it holds whatever the
    requesting process claims its stage to be."""
    read_only = BridgeBackend(NAMESPACE, allow_writes=False)

    with pytest.raises(ValueError, match="graded read-only"):
        read_only.dispatch(kubectl(f"kubectl {verb} deployment/x -n {NAMESPACE}"))


@pytest.mark.parametrize(
    ("tool", "stage", "arguments"),
    [
        ("restore_mongodb_access", "execution", {"service": "mongodb-geo", "access": "privileges"}),
        ("probe_service", "verify", {"service": "frontend", "path": "/"}),
    ],
)
def test_a_read_only_task_refuses_both_extra_actions(
    tool: str, stage: str, arguments: dict[str, Any]
) -> None:
    read_only = BridgeBackend(NAMESPACE, allow_writes=False)

    with pytest.raises(ValueError, match="graded read-only"):
        read_only.dispatch(
            {
                "namespace": NAMESPACE,
                "stage": stage,
                "server": "aiopslab",
                "tool": tool,
                "arguments": arguments,
            }
        )


@pytest.mark.parity
def test_a_read_only_task_still_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    read_only = BridgeBackend(NAMESPACE, allow_writes=False)
    run = FakeRun(completed("pod/x\n"))
    monkeypatch.setattr(bridge_module.subprocess, "run", run)

    assert read_only.dispatch(kubectl("kubectl get pods")) == "pod/x"


@pytest.mark.parity
def test_a_jaeger_service_query_is_dispatched(
    backend: BridgeBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, dict[str, str], bool]] = []

    def fake_query(path: str, params: dict[str, str], *, data_only: bool) -> str:
        calls.append((path, params, data_only))
        return json.dumps(["frontend", "geo"])

    monkeypatch.setattr(backend, "_jaeger_query", fake_query)

    result = backend.dispatch(
        {"namespace": NAMESPACE, "server": "jaeger", "tool": "get_services", "arguments": {}}
    )

    assert json.loads(result) == ["frontend", "geo"]
    assert calls == [("/api/services", {}, True)]


@pytest.mark.parity
def test_recovery_is_refused_outside_the_execution_stage(backend: BridgeBackend) -> None:
    with pytest.raises(ValueError, match="execution stage"):
        backend.dispatch(
            {
                "namespace": NAMESPACE,
                "stage": "rca",
                "server": "aiopslab",
                "tool": "restore_mongodb_access",
                "arguments": {"service": "mongodb-geo", "access": "privileges"},
            }
        )


@pytest.mark.parity
def test_recovery_runs_only_the_allowlisted_script_and_restart(
    backend: BridgeBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    pods = {
        "items": [
            {"metadata": {"name": "mongodb-geo-7f6dbf8f7b-abcd1"}},
            {"metadata": {"name": "geo-6dfb568c66-efgh2"}},
        ]
    }
    run = FakeRun(
        completed(json.dumps(pods)),
        completed("Privilege restored successfully\n"),
        completed("deployment.apps/geo restarted\n"),
    )
    monkeypatch.setattr(bridge_module.subprocess, "run", run)

    result = backend.dispatch(
        {
            "namespace": NAMESPACE,
            "stage": "execution",
            "server": "aiopslab",
            "tool": "restore_mongodb_access",
            "arguments": {"service": "mongodb-geo", "access": "privileges"},
        }
    )

    assert "Privilege restored successfully" in result
    assert run.calls[1][0] == [
        "kubectl",
        "exec",
        "mongodb-geo-7f6dbf8f7b-abcd1",
        "-n",
        NAMESPACE,
        "--",
        "/bin/bash",
        "/scripts/revoke-mitigate-admin-geo-mongo.sh",
    ]
    assert run.calls[2][0] == [
        "kubectl",
        "rollout",
        "restart",
        "deployment/geo",
        "-n",
        NAMESPACE,
    ]


@pytest.mark.parity
def test_recovery_refuses_an_unknown_selection(backend: BridgeBackend) -> None:
    with pytest.raises(ValueError, match="unsupported MongoDB recovery"):
        backend.dispatch(
            {
                "namespace": NAMESPACE,
                "stage": "execution",
                "server": "aiopslab",
                "tool": "restore_mongodb_access",
                "arguments": {"service": "mongodb-geo", "access": "shell"},
            }
        )


@pytest.mark.parity
def test_probing_is_refused_outside_the_verify_stage(backend: BridgeBackend) -> None:
    with pytest.raises(ValueError, match="verify stage"):
        backend.dispatch(
            {
                "namespace": NAMESPACE,
                "stage": "rca",
                "server": "aiopslab",
                "tool": "probe_service",
                "arguments": {"service": "frontend", "path": "/"},
            }
        )


@pytest.mark.parity
def test_probing_refuses_a_port_the_service_does_not_declare(
    backend: BridgeBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = {"spec": {"ports": [{"name": "http", "port": 8080}]}}
    monkeypatch.setattr(backend, "_run", lambda argv, *, timeout: json.dumps(service))
    monkeypatch.setattr(
        bridge_module.subprocess,
        "Popen",
        lambda *a, **k: pytest.fail("no port-forward may start for a refused port"),
    )

    with pytest.raises(ValueError, match="declared by the selected Service"):
        backend._probe_service(service="frontend", path="/", port=80)


class FakeProcess:
    """A port-forward that stays up and records how it was torn down."""

    def __init__(self) -> None:
        self.terminated = 0

    def poll(self) -> None:
        return None

    def terminate(self) -> None:
        self.terminated += 1

    def wait(self, timeout: float | None = None) -> int:
        return 0


@pytest.fixture
def port_forward(monkeypatch: pytest.MonkeyPatch) -> tuple[FakeProcess, list[list[str]]]:
    process = FakeProcess()
    argvs: list[list[str]] = []

    def fake_popen(argv: list[str], **_kwargs: Any) -> FakeProcess:
        argvs.append(argv)
        return process

    class FakeSocket:
        def __enter__(self) -> "FakeSocket":
            return self

        def __exit__(self, *_exc: object) -> bool:
            return False

    monkeypatch.setattr(bridge_module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(bridge_module.socket, "create_connection", lambda *a, **k: FakeSocket())
    return process, argvs


@pytest.mark.parity
def test_probing_forwards_only_to_a_declared_port(
    backend: BridgeBackend,
    monkeypatch: pytest.MonkeyPatch,
    port_forward: tuple[FakeProcess, list[list[str]]],
) -> None:
    process, argvs = port_forward
    service = {"spec": {"ports": [{"name": "http", "port": 8080}]}}
    monkeypatch.setattr(backend, "_run", lambda argv, *, timeout: json.dumps(service))
    monkeypatch.setattr(backend, "_allocate_loopback_port", lambda: 45678)

    class FakeResponse:
        status = 200

        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *_exc: object) -> bool:
            return False

    monkeypatch.setattr(bridge_module, "urlopen", lambda *a, **k: FakeResponse())

    result = backend._probe_service(service="frontend", path="/health", port=8080)

    assert result == "HTTP 200 via service/frontend:8080/health"
    assert argvs[0][:4] == ["kubectl", "port-forward", "service/frontend", "45678:8080"]
    assert argvs[0][4:] == ["-n", NAMESPACE, "--address", "127.0.0.1"]
    assert process.terminated == 1


@pytest.mark.parity
def test_a_non_http_answer_is_reported_as_evidence_rather_than_recovery(
    backend: BridgeBackend,
    monkeypatch: pytest.MonkeyPatch,
    port_forward: tuple[FakeProcess, list[list[str]]],
) -> None:
    service = {"spec": {"ports": [{"port": 9090}]}}
    monkeypatch.setattr(backend, "_run", lambda argv, *, timeout: json.dumps(service))
    monkeypatch.setattr(backend, "_allocate_loopback_port", lambda: 45678)

    def refuse(*_args: object, **_kwargs: object) -> None:
        raise RemoteDisconnected("closed")

    monkeypatch.setattr(bridge_module, "urlopen", refuse)

    result = backend._probe_service(service="internal-rpc", path="/", port=9090)

    assert "service/internal-rpc:9090" in result
    assert "non-HTTP protocol" in result
    assert "do not claim recovery without an HTTP status" in result


def post_action(port: int, token: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    request = Request(
        f"http://127.0.0.1:{port}/v1/action",
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=5) as response:
            return int(response.status), json.loads(response.read())
    except HTTPError as exc:
        return int(exc.code), json.loads(exc.read())


@pytest.mark.parity
def test_the_bridge_answers_only_a_matching_bearer_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bridge_module.subprocess, "run", FakeRun(completed("pod/x\n")))

    with serve_bridge(BridgeBackend(NAMESPACE), "s3cret-token-value", port=0) as server:
        port = server.server_address[1]
        assert isinstance(port, int)
        unauthorized = post_action(port, "wrong-token", kubectl("kubectl get pods"))
        authorized = post_action(port, "s3cret-token-value", kubectl("kubectl get pods"))
        with urlopen(f"http://127.0.0.1:{port}/healthz", timeout=5) as health:
            healthz = json.loads(health.read())

    assert unauthorized == (401, {"ok": False, "result": "unauthorized"})
    assert authorized == (200, {"ok": True, "result": "pod/x"})
    assert healthz == {"ok": True, "namespace": NAMESPACE}


def test_a_refused_action_is_a_structured_400_rather_than_a_crash() -> None:
    with serve_bridge(BridgeBackend(NAMESPACE), "s3cret-token-value", port=0) as server:
        port = server.server_address[1]
        assert isinstance(port, int)
        status, body = post_action(port, "s3cret-token-value", kubectl("kubectl get pods", "other"))

    assert status == 400
    assert body["ok"] is False
    assert "does not match" in body["result"]


def test_an_empty_token_is_refused_before_the_server_binds() -> None:
    """A bridge serving no token is a cluster reachable by anything on the host."""
    with (
        pytest.raises(ValueError, match="must not be empty"),
        serve_bridge(BridgeBackend(NAMESPACE), ""),
    ):
        pass  # pragma: no cover - serve_bridge raises before the body runs


def test_dispatch_is_recorded_before_execution_and_after_a_client_disconnect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = bridge_module.AccessLog(tmp_path / "bridge.log")
    payload = kubectl("kubectl get pods")
    dispatched = []

    def dispatch(arguments):
        rows = [json.loads(line) for line in (tmp_path / "bridge.log").read_text().splitlines()]
        assert [row["event"] for row in rows] == ["received"]
        assert json.loads((tmp_path / rows[0]["request"]["path"]).read_text()) == payload
        dispatched.append(arguments)
        return "pod/x"

    def disconnected(*args):
        raise BrokenPipeError("client left")

    handler = object.__new__(bridge_module._BridgeHandler)
    handler.path = "/v1/action"
    encoded = json.dumps(payload).encode()
    handler.headers = {"Authorization": "Bearer test-token", "Content-Length": str(len(encoded))}
    handler.rfile = io.BytesIO(encoded)
    handler.server = SimpleNamespace(
        token="test-token",
        backend=SimpleNamespace(dispatch=dispatch),
        access_log=log,
        server_address=("127.0.0.1", 0),
    )
    monkeypatch.setattr(handler, "_json", disconnected)

    handler.do_POST()

    assert dispatched == [payload]
    rows = [json.loads(line) for line in (tmp_path / "bridge.log").read_text().splitlines()]
    assert [row["event"] for row in rows] == ["received", "finished"]
    assert rows[-1]["decision"] == "served"
    assert rows[-1]["status"] == 200
    assert "BrokenPipeError" in rows[-1]["delivery_error"]
    assert (tmp_path / rows[-1]["result"]["path"]).read_text() == "pod/x"


@pytest.mark.parametrize("failed_label", ["request.json", "response.txt"])
def test_audit_capture_failure_is_reported_without_repeating_or_hiding_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed_label: str
) -> None:
    log = bridge_module.AccessLog(tmp_path / "bridge.log")
    original_capture = log.capture
    captures = []
    dispatched = []
    delivered = []

    def capture(receipt_id, label, content):
        captures.append(label)
        if label == failed_label:
            raise OSError("disk full")
        return original_capture(receipt_id, label, content)

    def dispatch(payload):
        dispatched.append(payload)
        return "pod/x"

    monkeypatch.setattr(log, "capture", capture)
    handler = object.__new__(bridge_module._BridgeHandler)
    handler.path = "/v1/action"
    encoded = json.dumps(kubectl("kubectl get pods")).encode()
    handler.headers = {"Authorization": "Bearer test-token", "Content-Length": str(len(encoded))}
    handler.rfile = io.BytesIO(encoded)
    handler.server = SimpleNamespace(
        token="test-token",
        backend=SimpleNamespace(dispatch=dispatch),
        access_log=log,
        server_address=("127.0.0.1", 0),
    )
    monkeypatch.setattr(handler, "_json", lambda status, body: delivered.append((status, body)))

    handler.do_POST()

    assert captures == ["request.json", "response.txt"]
    rows = [json.loads(line) for line in (tmp_path / "bridge.log").read_text().splitlines()]
    assert rows[-1]["event"] == "finished"
    assert "disk full" in rows[-1]["audit_errors"][0]
    if failed_label == "request.json":
        assert dispatched == []
        assert delivered[0][0] == 500
        assert "disk full" in delivered[0][1]["result"]
        assert rows[-1]["decision"] == "failed"
    else:
        assert len(dispatched) == 1
        assert delivered == [(200, {"ok": True, "result": "pod/x"})]
        assert rows[-1]["decision"] == "served"
        assert rows[-1]["result"] is None
