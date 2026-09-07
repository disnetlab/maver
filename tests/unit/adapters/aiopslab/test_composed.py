"""One process, configured the way an AIOpsLab host configures it, answering incidents.

The unit tests above check each piece against its own seam. This one checks that the
pieces are actually wired to each other: settings select the adapter, the adapter builds
the bridge transport, the runner asks it for an overlay at every role entry, and what
comes out is the prompt, the toolset, the budget and the calls the prototype's AIOpsLab
mode produced. Nothing here is stubbed except the model and the bridge's far side — the
bridge is a real HTTP server on an ephemeral loopback port, so the transport, the
approval gate and the tool dispatch all run for real.
"""

import asyncio
import contextlib
import dataclasses
import json
import pathlib
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import httpx
import pytest

from kira.api.app import build_deployment, create_app
from kira.api.deps import Service
from kira.audit.journal import ListSink
from kira.llm.types import ModelRequest, ModelResponse, ToolCall, Usage
from kira.orchestration.runner import Deployment
from kira.roles.policy import control
from kira.settings.root import load_settings

PROMPTS = pathlib.Path(__file__).resolve().parents[4] / "src/kira/adapters/aiopslab/prompts"

TOKEN = "a-bridge-token-long-enough"
NAMESPACE = "test-hotel-reservation"

DETECTION_ALERT: dict[str, Any] = {
    "alertname": "ServiceHealthDegraded",
    "benchmark": "aiopslab",
    "benchmark_task": "detection",
    "namespace": NAMESPACE,
}
MITIGATION_ALERT = {**DETECTION_ALERT, "benchmark_task": "mitigation"}

ROOT_CAUSE: dict[str, Any] = {
    "fault_type": "authentication_issue",
    "affected_resource": f"{NAMESPACE}/Deployment/mongodb-rate",
    "confidence": 0.8,
    "evidence": ["rate service logs an authentication failure against mongodb-rate"],
    "summary": "the admin user's privileges on mongodb-rate were revoked",
}
PLAN: dict[str, Any] = {
    "steps": [
        {
            "action": "restore_mongodb_access",
            "target": "mongodb-rate",
            "params": {"access": "privileges"},
            "justification": "the only supported repair for a revoked admin role",
        }
    ],
    "expected_outcome": "the rate service authenticates again",
    "confidence": 0.8,
}
EXECUTION: dict[str, Any] = {
    "succeeded": True,
    "steps": [
        {
            "step_index": 0,
            "succeeded": True,
            "observation": "before: auth failures; ran the recovery; after: rollout complete",
        }
    ],
    "final_state_summary": "mongodb-rate accepts the application's credentials",
    "confidence": 0.8,
}
VERIFY: dict[str, Any] = {
    "recovered": True,
    "verdict": "recovered",
    "checks": ["probe_service frontend / -> HTTP 200 OK"],
    "summary": "the user-facing entry point answers again",
    "entry_point_status": "HTTP 200 OK from service/frontend",
}

_ANSWERS = {
    "probe_service": "HTTP 200 OK",
    "restore_mongodb_access": "recovery script applied; deployment restarted",
}


class _Handler(BaseHTTPRequestHandler):
    """Answers the one route the transport posts to, and records what it was asked."""

    def do_POST(self) -> None:
        body = self.rfile.read(int(self.headers.get("Content-Length", "0")) or 0)
        payload = json.loads(body or b"{}")
        server: Any = self.server
        server.received.append((self.headers.get("Authorization"), payload))
        answer = json.dumps(
            {"ok": True, "result": _ANSWERS.get(payload.get("tool", ""), "no resources found")}
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(answer)))
        self.end_headers()
        self.wfile.write(answer)

    def log_message(self, *args: Any) -> None:
        """Keep the test output the test's."""


@contextlib.contextmanager
def bridge() -> Iterator[Any]:
    """Serve the action bridge on an ephemeral loopback port."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.received = []  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


class Scripted:
    """Answers each role with a fixed sequence of turns, and records every request."""

    def __init__(self, script: dict[str, list[tuple[str, dict[str, Any]]]]) -> None:
        self.requests: list[tuple[str, ModelRequest]] = []
        self._script = script

    async def complete(self, request: ModelRequest, *, caller: str) -> ModelResponse:
        self.requests.append((caller, request))
        turn = sum(1 for role, _ in self.requests if role == caller)
        turns = self._script[caller]
        name, arguments = turns[min(turn, len(turns)) - 1]
        if turn > len(turns):
            # Past the script, keep making distinct reads: the repeat breaker would
            # otherwise stop the run before the turn budget does.
            name, arguments = "exec_kubectl_cmd_safely", {"cmd": f"kubectl get pods -l t={turn}"}
        return ModelResponse(
            text=None,
            tool_calls=(
                ToolCall(
                    id=f"c{len(self.requests)}",
                    name=name,
                    arguments=arguments,
                    raw_arguments=json.dumps(arguments),
                ),
            ),
            finish_reason="stop",
            reasoning=None,
            usage=Usage(),
            duration_ms=1,
            attempts=1,
        )


def deployment(
    tmp_path: pathlib.Path, port: int, model: Scripted, monkeypatch: pytest.MonkeyPatch
) -> tuple[Deployment, ListSink]:
    """Build the process the way an AIOpsLab host configures it: from the environment."""
    token = tmp_path / "bridge.token"
    token.write_text(TOKEN, encoding="utf-8")
    for name, value in {
        "KIRA_SERVER__WORKDIR": str(tmp_path),
        "KIRA_INTAKE__ADAPTERS": "aiopslab",
        "KIRA_ADAPTERS__AIOPSLAB__ENABLED": "true",
        "KIRA_ADAPTERS__AIOPSLAB__BRIDGE_URL": f"http://127.0.0.1:{port}",
        "KIRA_ADAPTERS__AIOPSLAB__BRIDGE_TOKEN_FILE": str(token),
        "KIRA_AUTHORITY__MODE": "auto",
        "KIRA_VERIFY__ENABLED": "true",
        "KIRA_MEMORY__ENABLED": "false",
        "KIRA_MEMORY__LEARNING_ENABLED": "false",
        # Off so that what reaches the bridge is what a role asked for. The auto-injected
        # snapshot is the diagnosing role's own behaviour and is tested where it lives.
        "KIRA_PIPELINE__AUTO_SNAPSHOT": "false",
        "KIRA_PIPELINE__MAX_TURNS": "25",
    }.items():
        monkeypatch.setenv(name, value)
    sink = ListSink()
    built = build_deployment(load_settings())
    return dataclasses.replace(built, completion=model, sink=sink), sink


def client(app: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://kira")


async def settle(service: Service) -> None:
    while pending := {task for task in service.runs.tasks() if not task.done()}:
        await asyncio.gather(*pending, return_exceptions=True)


async def run(app: Any, alert: dict[str, Any], stage: str) -> str:
    async with client(app) as http:
        response = await http.post(
            "/api/v1/incidents", json={"alert": alert, "target_stage": stage}
        )
        session_id = str(response.json()["session_id"])
        await settle(app.state.service)
    return session_id


def first(model: Scripted, role: str) -> ModelRequest:
    return next(request for caller, request in model.requests if caller == role)


def body(request: ModelRequest) -> str:
    """The system prompt without the timestamp, which moves between two runs."""
    return str(request.system).split("\n\n", 1)[1]


def offered(request: ModelRequest) -> set[str]:
    return {tool.name for tool in request.tools}


async def test_a_detection_incident_runs_the_bounded_yes_or_no_task(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Everything the detection overlay does, observed from outside the adapter.

    The model here never submits, so the run ends on its budget — which is what makes
    the budget observable at all, and the budget is the one field of the overlay that
    nothing downstream records.
    """
    model = Scripted({"rca": [("exec_kubectl_cmd_safely", {"cmd": "kubectl get pods"})]})
    with bridge() as server:
        deploy, _ = deployment(tmp_path, server.server_port, model, monkeypatch)
        app = create_app(deployment=deploy, schedule_resume=lambda request_id: None)
        session_id = await run(app, DETECTION_ALERT, "rca")

        # The bridge is told which incident it is serving and which role is calling,
        # because it refuses any action whose namespace is not the active problem's.
        authorization, action = server.received[0]
        assert authorization == f"Bearer {TOKEN}"
        assert action["namespace"] == NAMESPACE
        assert action["stage"] == "rca"
        assert action["server"] == "kubectl"

    opening = first(model, "rca")
    detection = (PROMPTS / "detection.txt").read_text(encoding="utf-8").strip()
    # The composed role prompt is gone; the harness plumbing around it is not, exactly
    # as the prototype's own `agent_system_prompt` wrapped this same text.
    assert body(opening) == f"{detection}\n\n{control('tool_failure_guidance')}"

    assert offered(opening).isdisjoint({"bash_exec", "code_exec", "get_operations"})
    withheld = next(
        run_.metadata["tools_withheld"]
        for run_ in deploy.storage.runs.runs_for(session_id)
        if run_.agent_name == "rca"
    )
    # Recorded with the reason, not merely absent: what a past run offered the model is
    # not reconstructible from the arm alone. The other drops on this row belong to the
    # transport and the configuration, and the environment claims none of them.
    assert {name for name, why in withheld.items() if why == "withheld by the environment"} == {
        "bash_exec",
        "code_exec",
        "get_operations",
    }

    # Ten, not the twenty-five this deployment configured.
    normal = [
        request
        for caller, request in model.requests
        if caller == "rca" and "exec_kubectl_cmd_safely" in offered(request)
    ]
    assert len(normal) == 10
    deploy.storage.close()


async def test_a_mitigation_incident_gets_the_bridges_repair_and_its_own_probe(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The remediation half, end to end: what planning and remediation are told about
    the transport, which role is offered which of the two adapter tools, that the write
    passes the approval gate, and what the checking role opens on."""
    model = Scripted(
        {
            "rca": [("submit_root_cause", ROOT_CAUSE)],
            "planning": [("submit_plan", PLAN)],
            "execution": [
                # Execution's opening call is forced to be a read.
                ("exec_kubectl_cmd_safely", {"cmd": f"kubectl get pods -n {NAMESPACE}"}),
                (
                    "restore_mongodb_access",
                    {"service": "mongodb-rate", "access": "privileges"},
                ),
                ("submit_execution_result", EXECUTION),
            ],
            "verify": [
                ("probe_service", {"service": "frontend", "path": "/"}),
                ("submit_verify_result", VERIFY),
            ],
        }
    )
    with bridge() as server:
        deploy, sink = deployment(tmp_path, server.server_port, model, monkeypatch)
        app = create_app(deployment=deploy, schedule_resume=lambda request_id: None)
        await run(app, MITIGATION_ALERT, "full")
        actions = [action for _, action in server.received]

    remediation = (PROMPTS / "remediation.txt").read_text(encoding="utf-8").strip()
    guidance = control("tool_failure_guidance")
    # After the role's own prompt and before the tool-call protocol every role is sent,
    # because the suffix is a statement about the role and the protocol is plumbing.
    assert body(first(model, "planning")).endswith(f"\n\n{remediation}\n\n{guidance}")
    # The role that writes is told the same thing, and then which namespace it is
    # confined to and that nobody is waiting to approve its mutation.
    execution_prompt = body(first(model, "execution"))
    assert f"{remediation}\n\n## AIOpsLab incident scope" in execution_prompt
    assert f"The active incident namespace is `{NAMESPACE}`" in execution_prompt
    assert "This run is non-interactive" in execution_prompt

    # Each adapter tool reaches exactly the role it was contributed for.
    assert "restore_mongodb_access" in offered(first(model, "execution"))
    assert "probe_service" not in offered(first(model, "execution"))
    assert "probe_service" in offered(first(model, "verify"))
    assert "restore_mongodb_access" not in offered(first(model, "verify"))

    # The checking role opens on the entry point it is meant to probe first.
    contract = str(first(model, "verify").messages[0]["content"])
    assert contract.startswith("## Benchmark-supplied application connectivity contract")
    assert json.loads(contract.split("```json\n", 1)[1].split("\n```", 1)[0]) == {
        "application": "Hotel Reservation",
        "namespace": NAMESPACE,
        "http_entry_point": {"service": "frontend", "port": 5000, "path": "/"},
    }

    assert ("aiopslab", "restore_mongodb_access", "execution") in [
        (action["server"], action["tool"], action["stage"]) for action in actions
    ]
    assert ("aiopslab", "probe_service", "verify") in [
        (action["server"], action["tool"], action["stage"]) for action in actions
    ]
    # The repair is a cluster write and went through the gate like any other, which on
    # this deployment decides it without a human and records that it did.
    decisions = [
        (event.payload.tool, event.payload.policy, event.payload.decision)
        for event in sink.events
        if event.payload.kind == "policy_decision"
    ]
    assert ("restore_mongodb_access", "auto", "approved") in decisions
    deploy.storage.close()
