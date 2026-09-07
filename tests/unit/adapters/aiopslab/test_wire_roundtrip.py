"""The two halves of the AIOpsLab wire, run against each other.

The bridge and the transport were written from one written description without either
author seeing the other's code, and every other test of them stubs the half it is not
testing. A stub agrees with the mistake its author made: the transport's own suite
answers ``get_alerts`` with a bare JSON list, which is not what the host code in this
repository sends, and no test either side could see that.

So nothing here is stubbed at the wire. The real ``serve_bridge`` runs on an ephemeral
loopback port and the real ``AiopsLabTransport`` talks to it. What is replaced is only
what the bridge shells out to or fetches from — a ``kubectl`` on PATH that echoes its
own argv, a Prometheus, and the Jaeger the bridge port-forwards to — so that the
argument names, the encodings and the refusal wording are the two implementations' own.

Where a role parses what a transport returns, the assertion is the role's parser rather
than a comparison this file invents: a payload that is byte-perfect and unreadable by
``_parse_alerts`` is worth nothing to the model.
"""

import ast
import json
import os
import stat
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from urllib.request import urlopen

import pytest

from kira.adapters.aiopslab.host.bridge import BridgeBackend, serve_bridge
from kira.adapters.aiopslab.transport import AiopsLabTransport
from kira.tools.errors import call_failed
from kira.tools.metrics import _parse_alerts, partition_incident_alerts
from kira.tools.snapshot import _load_items
from kira.tools.traces import _filter_traces_to_candidates, _parse_structured

NAMESPACE = "test-hotel-reservation"
TOKEN = "s3cret-token-value"
SESSION = "incident-42"
CANDIDATES = frozenset({"frontend", "geo", "mongodb-geo"})

PODS = {
    "apiVersion": "v1",
    "kind": "PodList",
    "items": [
        {
            "metadata": {"name": "geo-6dfb568c66-efgh2", "labels": {"app": "geo"}},
            "status": {"phase": "Running"},
        },
        {
            "metadata": {"name": "mongodb-geo-7f6dbf8f7b-abcd1", "labels": {"app": "mongodb-geo"}},
            "status": {"phase": "Running"},
        },
    ],
}

SERVICE = {"spec": {"ports": [{"name": "http", "port": 8080}]}}

JAEGER = {
    "/api/services": {"data": ["frontend", "geo", "otherapp"]},
    "/api/services/frontend/operations": {"data": ["GET /", "POST /reserve"]},
    "/api/traces": {
        "data": [
            {
                "traceID": "abc",
                "spans": [
                    {"spanID": "1", "processID": "p1", "operationName": "GET /"},
                    {"spanID": "2", "processID": "p2", "operationName": "GET /ads"},
                ],
                "processes": {
                    "p1": {"serviceName": "frontend"},
                    "p2": {"serviceName": "otherapp"},
                },
            }
        ]
    },
    "/api/dependencies": {
        "data": [
            {"parent": "frontend", "child": "geo", "callCount": 7},
            {"parent": "otherapp", "child": "geo", "callCount": 1},
        ]
    },
}

ALERTS = [
    {
        "labels": {"alertname": "PodCrashLooping", "namespace": NAMESPACE},
        "annotations": {"summary": "geo is restarting"},
        "state": "firing",
        "value": None,
    },
    {
        "labels": {"alertname": "HighLatency", "namespace": "otherapp"},
        "annotations": {"summary": "p99 above budget"},
        "state": "firing",
        "silenced": True,
    },
]

METRIC_DATA = {
    "resultType": "vector",
    "result": [{"metric": {"__name__": "up", "namespace": NAMESPACE}, "value": [1.0, "1"]}],
}

# Longer than the transport's bound, so the marker and the cut are observed rather than
# assumed. `get events` is the shape that produced them in the archived runs.
OVERSIZED = "E" * 12_000

_KUBECTL_STUB = '''#!/usr/bin/env python3
"""A kubectl that answers a few reads from fixtures and echoes everything else.

Every invocation is appended to calls.jsonl, so a multi-step bridge action (probing a
Service, repairing MongoDB) can be checked against the commands it actually ran rather
than only against the sentence it returned.
"""

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

HERE = Path(__file__).resolve().parent
FIXTURES = json.loads((HERE / "fixtures.json").read_text())
CALLS = HERE / "calls.jsonl"


def record(entry):
    with CALLS.open("a") as log:
        log.write(json.dumps(entry) + "\\n")


class Handler(BaseHTTPRequestHandler):
    """The forwarded-to backend: Jaeger for its own paths, 200 for anything else."""

    def do_GET(self):
        record({"kind": "http", "path": self.path})
        body = FIXTURES["jaeger"].get(urlsplit(self.path).path, {"status": "ok"})
        encoded = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, *_args):
        return


def main():
    args = sys.argv[1:]
    record({"kind": "kubectl", "argv0": os.path.basename(sys.argv[0]), "args": args})
    if args[:1] == ["port-forward"]:
        mapping = next(a for a in args if ":" in a and a.split(":", 1)[0].isdigit())
        ThreadingHTTPServer(("127.0.0.1", int(mapping.split(":", 1)[0])), Handler).serve_forever()
    elif args[:2] == ["get", "service"]:
        print(json.dumps(FIXTURES["service"]))
    elif args[:2] == ["get", "pods"] and "json" in args:
        print(json.dumps(FIXTURES["pods"]))
    elif args[:2] == ["get", "events"]:
        print(FIXTURES["oversized"])
    elif "pod/does-not-exist" in args:
        print('Error from server (NotFound): pods "does-not-exist" not found', file=sys.stderr)
        sys.exit(1)
    elif args[:1] == ["exec"]:
        print("Privilege restored successfully")
    elif args[:2] == ["rollout", "restart"]:
        print("deployment.apps/geo restarted")
    else:
        print(json.dumps({"argv0": os.path.basename(sys.argv[0]), "args": args}))


main()
'''


class _PrometheusHandler(BaseHTTPRequestHandler):
    """AIOpsLab's Prometheus, answering the two paths the bridge asks for."""

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path == "/api/v1/query":
            body: Any = {"status": "success", "data": METRIC_DATA}
        elif path == "/api/v1/alerts":
            body = {"status": "success", "data": {"alerts": ALERTS}}
        else:
            self.send_error(404)
            return
        encoded = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, *_args: Any) -> None:
        return


@pytest.fixture
def prometheus() -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _PrometheusHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture
def kubectl_stub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Put a kubectl on PATH that the bridge will find and this test can read back."""
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir()
    (stub_dir / "fixtures.json").write_text(
        json.dumps({"jaeger": JAEGER, "pods": PODS, "service": SERVICE, "oversized": OVERSIZED})
    )
    binary = stub_dir / "kubectl"
    binary.write_text(_KUBECTL_STUB)
    binary.chmod(binary.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    monkeypatch.setenv("PATH", f"{stub_dir}{os.pathsep}{os.environ['PATH']}")
    return stub_dir


def calls(stub_dir: Path, kind: str) -> list[dict[str, Any]]:
    log = stub_dir / "calls.jsonl"
    if not log.exists():
        return []
    entries = [json.loads(line) for line in log.read_text().splitlines() if line]
    return [entry for entry in entries if entry["kind"] == kind]


Wire = Callable[..., Any]


@pytest.fixture
def wire(prometheus: str, kubectl_stub: Path) -> Wire:
    """Open the real bridge and hand back the real transport bound to it."""

    @contextmanager
    def open_wire(
        *,
        allow_writes: bool = True,
        namespace: str = NAMESPACE,
        token: str = TOKEN,
        stage: str = "rca",
    ) -> Iterator[AiopsLabTransport]:
        backend = BridgeBackend(NAMESPACE, allow_writes=allow_writes, prometheus_url=prometheus)
        with serve_bridge(backend, TOKEN, port=0) as server:
            transport = AiopsLabTransport(
                f"http://127.0.0.1:{server.server_address[1]}",
                token,
                timeout_s=30.0,
                session_id=SESSION,
            )
            transport.namespace = namespace
            transport.stage = stage
            yield transport

    return open_wire


async def test_a_namespace_less_read_reaches_kubectl_scoped_to_the_active_namespace(
    wire: Wire, kubectl_stub: Path
) -> None:
    """The transport re-joins the binary the tool stripped, and the bridge scopes.

    Both halves are visible in one answer: the stub reports the argv it was executed
    with, so a transport that forgot to prepend ``kubectl`` (the bridge would refuse
    the command outright) and one that prepended it twice (the verb would be
    ``kubectl``, and every classification downstream would read the command as
    something else) are each a different failure here.
    """
    with wire() as transport:
        answer = await transport.kubectl(["get", "deployments"])

    echoed = json.loads(answer)
    assert echoed == {
        "argv0": "kubectl",
        "args": ["get", "deployments", "--namespace", NAMESPACE],
    }
    assert calls(kubectl_stub, "kubectl")[0]["args"][0] == "get"


async def test_a_pod_inventory_survives_the_round_trip_for_the_parser_that_reads_it(
    wire: Wire,
) -> None:
    """Two consumers parse this as JSON, and both report a failure to as an empty
    namespace rather than as an error, so a re-encoding here is invisible downstream."""
    with wire() as transport:
        answer = await transport.kubectl(["get", "pods", "-n", NAMESPACE, "-o", "json"])

    assert answer == json.dumps(PODS)
    items = _load_items(answer)
    assert items is not None
    assert [item["metadata"]["name"] for item in items] == [
        "geo-6dfb568c66-efgh2",
        "mongodb-geo-7f6dbf8f7b-abcd1",
    ]


async def test_an_oversized_answer_is_cut_with_the_marker_the_prompts_were_written_against(
    wire: Wire,
) -> None:
    with wire() as transport:
        answer = await transport.kubectl(["get", "events"])

    assert answer == "E" * 10_000 + "… [truncated]"


async def test_an_incident_namespace_the_bridge_does_not_serve_is_refused_in_its_words(
    wire: Wire,
) -> None:
    """A refusal, not an exception: the loop counts a failed tool call by its prefix."""
    with wire(namespace="another-problem") as transport:
        answer = await transport.kubectl(["get", "pods"])

    assert answer == (
        "error: incident namespace 'another-problem' does not match active "
        f"AIOpsLab namespace {NAMESPACE!r}"
    )
    assert call_failed(answer)


async def test_a_rejected_token_reaches_the_loop_as_a_failed_call_that_names_the_failure(
    wire: Wire,
) -> None:
    with wire(token="not-the-token") as transport:
        answer = await transport.kubectl(["get", "pods"])

    assert answer == "error: unauthorized"
    assert call_failed(answer)


@pytest.mark.parametrize("stage", ["rca", "planning", "execution", "verify"])
async def test_a_write_is_refused_on_a_read_only_task_whatever_stage_claims_it(
    wire: Wire, stage: str
) -> None:
    """``allow_writes`` is derived from the task type before the episode starts, so it
    holds against a request that claims any stage at all."""
    with wire(allow_writes=False, stage=stage) as transport:
        answer = await transport.kubectl(
            ["scale", "deployment/geo", "--replicas=3", "-n", NAMESPACE]
        )

    assert answer == (
        "error: kubectl scale is not permitted: this AIOpsLab task is graded read-only"
    )
    assert call_failed(answer)


async def test_a_write_on_a_remediation_task_reaches_kubectl_unaltered(wire: Wire) -> None:
    with wire(allow_writes=True, stage="execution") as transport:
        answer = await transport.kubectl(
            ["scale", "deployment/geo", "--replicas=3", "-n", NAMESPACE]
        )

    assert json.loads(answer) == {
        "argv0": "kubectl",
        "args": ["scale", "deployment/geo", "--replicas=3", "-n", NAMESPACE],
    }


async def test_an_in_container_command_is_refused_because_it_cannot_be_proven_read_only(
    wire: Wire, kubectl_stub: Path
) -> None:
    with wire() as transport:
        answer = await transport.kubectl(["exec", "pod/geo", "-n", NAMESPACE, "--", "ls"])

    assert answer.startswith("error: kubectl exec is not permitted by the AIOpsLab bridge")
    assert calls(kubectl_stub, "kubectl") == []


@pytest.mark.parametrize(
    ("argv", "flag"),
    [
        (["get", "pods", "--all-namespaces"], "--all-namespaces"),
        (["get", "pods", "--kubeconfig=/etc/admin.conf"], "--kubeconfig"),
        (["get", "pods", "--context", "admin@prod"], "--context"),
        (["get", "--raw", "/api/v1/nodes"], "--raw"),
    ],
)
async def test_a_credential_or_scope_flag_is_refused_by_name(
    wire: Wire, kubectl_stub: Path, argv: list[str], flag: str
) -> None:
    """Naming the flag is the whole error channel: a role that is told which one was
    refused repairs the command, and one told only "failed" reissues it."""
    with wire() as transport:
        answer = await transport.kubectl(argv)

    assert answer == f"error: kubectl flag {flag!r} is not permitted"
    assert calls(kubectl_stub, "kubectl") == []


async def test_a_metric_query_comes_back_as_the_literal_the_delivery_layer_carries(
    wire: Wire,
) -> None:
    """The window is in the protocol and not on this wire: the bridge runs an instant
    query and takes no range, exactly as the harness's own metrics server does, so a
    caller passing one must not turn the query into an action the bridge refuses."""
    with wire() as transport:
        answer = await transport.query_metrics('up{job="geo"}', window_minutes=60)

    assert ast.literal_eval(answer) == METRIC_DATA


async def test_firing_alerts_come_back_in_the_shape_the_partitioning_reads(wire: Wire) -> None:
    """The partitioning parses with ``ast.literal_eval`` alone and, when it cannot,
    returns the payload untouched — so an unreadable shape reaches the model unscoped
    and unlabelled, with nothing anywhere to say that it happened.

    The bridge sends Prometheus's ``data`` object and the alerts are one key inside it,
    which is not what either half's own suite stubs: the transport's answers this action
    with a bare list, and the bridge's replaces the fetch entirely."""
    with wire() as transport:
        answer = await transport.query_alerts()

    assert _parse_alerts(answer) == ALERTS
    partitioned = partition_incident_alerts(NAMESPACE, answer)
    assert partitioned.startswith(f"INCIDENT-SCOPED firing alerts for namespace {NAMESPACE!r}:")
    assert "OUT-OF-SCOPE firing alerts from other namespaces" in partitioned


async def test_the_service_catalogue_is_the_list_of_names_the_scoping_intersects(
    wire: Wire,
) -> None:
    with wire() as transport:
        answer = await transport.query_traces("services")

    services = _parse_structured(answer)
    assert services == ["frontend", "geo", "otherapp"]
    assert sorted(s for s in services if s in CANDIDATES) == ["frontend", "geo"]


async def test_an_operations_query_carries_the_service_name_the_bridge_reads(
    wire: Wire, kubectl_stub: Path
) -> None:
    with wire() as transport:
        answer = await transport.query_traces("operations", service="frontend")

    assert _parse_structured(answer) == ["GET /", "POST /reserve"]
    assert any(
        call["path"] == "/api/services/frontend/operations" for call in calls(kubectl_stub, "http")
    )


async def test_a_trace_query_maps_the_transports_arguments_onto_jaegers_window(
    wire: Wire, kubectl_stub: Path
) -> None:
    """``last_n_minutes`` is the transport's word and ``lookback`` is Jaeger's, and the
    bridge is where one becomes the other. A rename on either side would leave the
    window silently at Jaeger's own default."""
    with wire() as transport:
        answer = await transport.query_traces("traces", service="frontend", last_n_minutes=5)

    assert any(
        call["path"] == "/api/traces?service=frontend&lookback=300s&limit=20"
        for call in calls(kubectl_stub, "http")
    )
    filtered = _filter_traces_to_candidates(answer, CANDIDATES)
    assert filtered is not None
    traces = ast.literal_eval(filtered)
    assert [span["spanID"] for span in traces[0]["spans"]] == ["1"]
    assert traces[0]["processes"] == {"p1": {"serviceName": "frontend"}}


async def test_the_dependency_graph_keeps_the_envelope_the_edge_filter_reads(
    wire: Wire, kubectl_stub: Path
) -> None:
    """This one query is not unwrapped on the host side, because the consumer above
    filters ``data`` in place and writes the whole object back."""
    with wire() as transport:
        answer = await transport.query_traces("dependencies", last_n_minutes=30)

    assert any(
        call["path"].startswith("/api/dependencies?endTs=") and "lookback=1800000" in call["path"]
        for call in calls(kubectl_stub, "http")
    )
    graph = _parse_structured(answer)
    assert isinstance(graph, dict)
    graph["data"] = [
        edge
        for edge in graph["data"]
        if edge["parent"] in CANDIDATES and edge["child"] in CANDIDATES
    ]
    assert graph["data"] == [{"parent": "frontend", "child": "geo", "callCount": 7}]


async def test_probing_a_service_reaches_the_bridge_with_the_arguments_it_reads(
    wire: Wire, kubectl_stub: Path
) -> None:
    """The extra actions have no place in the transport protocol, so they cross as a
    generic action and the argument names are all that binds the two halves."""
    with wire(allow_writes=True) as transport:
        answer = await transport.action(
            "aiopslab",
            "probe_service",
            {"service": "frontend", "path": "/health", "port": 8080},
            stage="verify",
        )

    assert answer == "HTTP 200 via service/frontend:8080/health"
    forwarded = [
        call for call in calls(kubectl_stub, "kubectl") if call["args"][0] == "port-forward"
    ]
    assert forwarded[0]["args"][1] == "service/frontend"
    assert forwarded[0]["args"][2].endswith(":8080")


async def test_probing_is_refused_from_any_stage_but_verify(wire: Wire) -> None:
    with wire(allow_writes=True) as transport:
        answer = await transport.action(
            "aiopslab", "probe_service", {"service": "frontend", "path": "/"}, stage="rca"
        )

    assert answer == "error: Service probing is permitted only during the verify stage"
    assert call_failed(answer)


async def test_mongodb_recovery_reaches_the_bridge_with_the_arguments_it_reads(
    wire: Wire, kubectl_stub: Path
) -> None:
    with wire(allow_writes=True) as transport:
        answer = await transport.action(
            "aiopslab",
            "restore_mongodb_access",
            {"service": "mongodb-geo", "access": "privileges"},
            stage="execution",
        )

    assert answer == "Privilege restored successfully\ndeployment.apps/geo restarted"
    ran = [call["args"] for call in calls(kubectl_stub, "kubectl")]
    assert ran[1] == [
        "exec",
        "mongodb-geo-7f6dbf8f7b-abcd1",
        "-n",
        NAMESPACE,
        "--",
        "/bin/bash",
        "/scripts/revoke-mitigate-admin-geo-mongo.sh",
    ]
    assert ran[2] == ["rollout", "restart", "deployment/geo", "-n", NAMESPACE]


async def test_both_extra_actions_are_refused_on_a_read_only_task(wire: Wire) -> None:
    with wire(allow_writes=False) as transport:
        probe = await transport.action(
            "aiopslab", "probe_service", {"service": "frontend", "path": "/"}, stage="verify"
        )
        recovery = await transport.action(
            "aiopslab",
            "restore_mongodb_access",
            {"service": "mongodb-geo", "access": "privileges"},
            stage="execution",
        )

    assert (
        probe == "error: Service probing is not permitted: this AIOpsLab task is graded read-only"
    )
    assert recovery == (
        "error: MongoDB recovery is not permitted: this AIOpsLab task is graded read-only"
    )


async def test_a_command_that_ran_and_failed_reaches_the_loop_as_a_failed_call(
    wire: Wire,
) -> None:
    """The two halves picked the same failure vocabulary independently, and the loop
    reads it: the bridge writes its own ``error:`` for a command that exited non-zero,
    and the transport stamps a refused action with one of its own. That doubles the
    prefix, which is what the prototype did and what the prompts were written against,
    so it is pinned rather than tidied — and the recogniser reads it as one failure
    either way, so a refused bridge action counts against the repeat breaker exactly
    like every other error.
    """
    with wire() as transport:
        answer = await transport.kubectl(["describe", "pod/does-not-exist", "-n", NAMESPACE])

    assert answer == ('error: error: Error from server (NotFound): pods "does-not-exist" not found')
    assert call_failed(answer)


async def test_an_action_the_bridge_does_not_serve_is_named_rather_than_dropped(
    wire: Wire,
) -> None:
    with wire() as transport:
        answer = await transport.action("loki", "query_range", {"query": '{app="geo"}'})

    assert answer == "error: unsupported bridge tool: loki.query_range"
    assert call_failed(answer)


@pytest.mark.parametrize(
    ("call", "expected"),
    [
        ("log_labels", "logs are not available on this transport"),
        ("query_logs", "logs are not available on this transport"),
        ("rollbackable_commands", "rollback is not available on this transport"),
        ("rollback_last", "rollback is not available on this transport"),
    ],
)
async def test_the_capabilities_this_environment_lacks_answer_without_a_bridge_call(
    wire: Wire, kubectl_stub: Path, call: str, expected: str
) -> None:
    """Dropped from every role's toolset before the run, so these are reached only by a
    caller ignoring the capability set — which must still read as a failure."""
    with wire() as transport:
        answer = await getattr(transport, call)(**({"query": "{}"} if call == "query_logs" else {}))
        labelled = await transport.log_label_values("app")

    assert answer.startswith(f"error: {expected}")
    assert call_failed(answer) and call_failed(labelled)
    assert calls(kubectl_stub, "kubectl") == []


def test_the_transport_declares_what_this_bridge_serves() -> None:
    transport = AiopsLabTransport("http://127.0.0.1:1", TOKEN, timeout_s=0.4)

    assert transport.name == "aiopslab"
    assert transport.capabilities == frozenset({"kubernetes", "metrics", "traces"})
    assert transport.timeout_seconds == 1


def test_the_liveness_probe_reports_the_problem_the_bridge_is_bound_to(
    prometheus: str,
) -> None:
    """The namespace is the containment claim, and it is the one thing a deploy can
    check from outside without holding the token."""
    backend = BridgeBackend(NAMESPACE, allow_writes=False, prometheus_url=prometheus)
    with (
        serve_bridge(backend, TOKEN, port=0) as server,
        urlopen(f"http://127.0.0.1:{server.server_address[1]}/healthz", timeout=5) as health,
    ):
        body = json.loads(health.read())

    assert body == {"ok": True, "namespace": NAMESPACE}
