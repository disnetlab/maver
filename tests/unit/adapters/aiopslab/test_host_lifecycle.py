"""What the driver does around the incident: blind it, wait for it, and submit once.

The agent is a real HTTP server here, on an ephemeral loopback port, because the poller
is the part that has to survive a run that is neither finished nor progressing, and a
mocked client cannot show that. AIOpsLab itself is faked at the one seam the driver
touches: an orchestrator with a registry, a problem and `start_problem`.
"""

import argparse
import json
import sys
import types
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from typing import Any

import pytest

from kira.adapters.aiopslab.host import lifecycle as lifecycle_module
from kira.adapters.aiopslab.host import main as main_module
from kira.adapters.aiopslab.host.lifecycle import (
    AIOpsLabAgentAdapter,
    instrument_observation_window,
    kubernetes_namespace,
    resolve_bridge_token,
    task_type,
)

NAMESPACE = "test-hotel-reservation"
ANOMALY = {
    "root_cause": {"halt": False, "fault_type": "Misconfiguration", "affected_resource": "x"}
}


class _AgentHandler(BaseHTTPRequestHandler):
    server: "FakeAgent"

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        self.server.posted.append(json.loads(self.rfile.read(length)))
        self._json({"session_id": "s-1", "status": "accepted"})

    def do_GET(self) -> None:
        if self.path == "/healthz":
            self._json({"status": "ok"})
            return
        self.server.reads += 1
        index = min(self.server.reads - 1, len(self.server.records) - 1)
        self._json(self.server.records[index])

    def log_message(self, _format: str, *_args: Any) -> None:
        return

    def _json(self, body: dict[str, Any]) -> None:
        encoded = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


class FakeAgent(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, records: list[dict[str, Any]]) -> None:
        super().__init__(("127.0.0.1", 0), _AgentHandler)
        self.records = records
        self.posted: list[dict[str, Any]] = []
        self.reads = 0


@contextmanager
def fake_agent(records: list[dict[str, Any]]) -> Iterator[FakeAgent]:
    server = FakeAgent(records)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def agent_url(server: FakeAgent) -> str:
    return f"http://127.0.0.1:{server.server_address[1]}"


@pytest.mark.parity
def test_the_injection_timestamp_is_recorded_and_nothing_else_is() -> None:
    events: list[str] = []

    class Problem:
        def inject_fault(self) -> None:
            events.append("inject")

    problem = Problem()
    calls: list[str] = []

    class Registry:
        def get_problem_instance(self, problem_id: str) -> Problem:
            calls.append(problem_id)
            return problem

    registry = Registry()
    marker: dict[str, str] = {}

    instrument_observation_window(registry, marker)
    returned = registry.get_problem_instance("hidden-id")
    assert marker == {}
    returned.inject_fault()

    assert returned is problem
    assert events == ["inject"]
    assert calls == ["hidden-id"]
    assert list(marker) == ["started_at"]
    assert marker["started_at"].endswith("Z")


@pytest.mark.parity
@pytest.mark.parametrize(
    ("class_name", "expected"),
    [
        ("DetectionTask", "detection"),
        ("LocalizationTask", "localization"),
        ("AnalysisTask", "analysis"),
        ("MitigationTask", "mitigation"),
    ],
)
def test_the_task_type_is_read_from_the_class_hierarchy(class_name: str, expected: str) -> None:
    base = type(class_name, (), {})
    problem = type("K8sProblem", (base,), {})()

    assert task_type(problem) == expected


@pytest.mark.parity
def test_an_unrecognised_problem_fails_loudly() -> None:
    problem = type("SomethingElse", (), {})()

    with pytest.raises(ValueError, match="cannot determine AIOpsLab task type"):
        task_type(problem)


@pytest.mark.parity
def test_a_docker_backed_problem_is_refused() -> None:
    with pytest.raises(ValueError, match="outside this Kubernetes agent"):
        kubernetes_namespace(types.SimpleNamespace(namespace="docker"))

    assert kubernetes_namespace(types.SimpleNamespace(namespace=NAMESPACE)) == NAMESPACE


def test_a_short_token_file_fails_with_its_path(tmp_path: Any) -> None:
    """Silently disabling every cluster tool is the failure this replaces."""
    path = tmp_path / "bridge-token"
    path.write_text("too-short\n", encoding="utf-8")

    with pytest.raises(ValueError, match=str(path)) as refused:
        resolve_bridge_token("", str(path))

    assert "9 characters" in str(refused.value)


def test_an_unreadable_token_file_fails_with_its_path(tmp_path: Any) -> None:
    path = tmp_path / "absent"

    with pytest.raises(ValueError, match=f"cannot read bridge token file {path}"):
        resolve_bridge_token("", str(path))


def test_a_token_file_wins_over_the_environment_value(tmp_path: Any) -> None:
    path = tmp_path / "bridge-token"
    path.write_text("0123456789abcdef0123\n", encoding="utf-8")

    assert resolve_bridge_token("from-environment", str(path)) == "0123456789abcdef0123"
    assert resolve_bridge_token("from-environment", None) == "from-environment"


def test_no_token_at_all_is_refused() -> None:
    with pytest.raises(ValueError, match="a bridge token is required"):
        resolve_bridge_token("", None)


async def test_the_poller_waits_through_an_approval_and_stops_at_the_terminal_record() -> None:
    records = [
        {"run_status": "awaiting_human", "status": "pending", "awaiting_request_id": "r-1"},
        {"run_status": "running", "status": "pending"},
        {"run_status": "completed", "status": "tentative_success", **ANOMALY},
    ]
    with fake_agent(records) as server:
        adapter = AIOpsLabAgentAdapter(
            task_type="detection",
            namespace=NAMESPACE,
            agent_url=agent_url(server),
            timeout_seconds=30,
            observation_window_start="2026-08-26T16:23:20Z",
            poll_seconds=0.01,
        )
        adapter.init_context("Hotel app", "instructions", {"submit": "submit()"})
        action = await adapter.get_action("")

        assert server.reads == 3
        assert adapter.incident_id == "s-1"
        assert action == '```\nsubmit("Yes")\n```'
        assert server.posted[0]["alert"]["observation_window_start"] == "2026-08-26T16:23:20Z"


async def test_a_lifecycle_terminal_record_ends_the_poll() -> None:
    with fake_agent([{"run_status": "running", "status": "invalidated"}]) as server:
        adapter = AIOpsLabAgentAdapter(
            task_type="localization",
            namespace=NAMESPACE,
            agent_url=agent_url(server),
            timeout_seconds=30,
            poll_seconds=0.01,
        )

        assert await adapter.get_action("") == "```\nsubmit([])\n```"
        assert server.reads == 1


async def test_the_action_is_produced_once_and_reused() -> None:
    with fake_agent([{"run_status": "completed", "status": "tentative_success", **ANOMALY}]) as srv:
        adapter = AIOpsLabAgentAdapter(
            task_type="detection",
            namespace=NAMESPACE,
            agent_url=agent_url(srv),
            timeout_seconds=30,
            poll_seconds=0.01,
        )
        first = await adapter.get_action("")
        second = await adapter.get_action("")

        assert first == second
        assert len(srv.posted) == 1


async def test_an_incident_that_never_finishes_times_out_with_its_id() -> None:
    with fake_agent([{"run_status": "running", "status": "pending"}]) as server:
        adapter = AIOpsLabAgentAdapter(
            task_type="detection",
            namespace=NAMESPACE,
            agent_url=agent_url(server),
            timeout_seconds=0.05,
            poll_seconds=0.01,
        )

        with pytest.raises(TimeoutError, match="s-1 exceeded"):
            await adapter.get_action("")


def fake_aiopslab(monkeypatch: pytest.MonkeyPatch, class_name: str) -> dict[str, Any]:
    """Install a fake `aiopslab.orchestrator` and report what the driver did with it."""
    seen: dict[str, Any] = {}
    problem = type("K8sProblem", (type(class_name, (), {}),), {})()
    problem.namespace = NAMESPACE  # type: ignore[attr-defined]

    class Registry:
        def get_problem_instance(self, problem_id: str) -> Any:
            seen["problem_id"] = problem_id
            return problem

    class Orchestrator:
        def __init__(self, results_dir: Any) -> None:
            self.probs = Registry()
            self.session: Any = None
            self.agent: Any = None
            seen["results_dir"] = str(results_dir)

        def register_agent(self, agent: Any, name: str) -> None:
            seen["agent_name"] = name

        def init_problem(self, problem_id: str) -> tuple[str, str, dict[str, str]]:
            instance = self.probs.get_problem_instance(problem_id)
            instance.inject_fault()
            self.session = types.SimpleNamespace(
                problem=instance, to_dict=lambda: {"problem": "recorded"}
            )
            return "Hotel app", "instructions", {"submit": "submit()"}

        async def start_problem(self, max_steps: int) -> dict[str, Any]:
            seen["max_steps"] = max_steps
            seen["action"] = await self.agent.get_action("")
            return {"graded": True}

    problem.inject_fault = lambda: None  # type: ignore[attr-defined]
    package = types.ModuleType("aiopslab")
    orchestrator_module = types.ModuleType("aiopslab.orchestrator")
    orchestrator_module.Orchestrator = Orchestrator  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "aiopslab", package)
    monkeypatch.setitem(sys.modules, "aiopslab.orchestrator", orchestrator_module)
    return seen


def driver_args(server: FakeAgent) -> argparse.Namespace:
    """Parse a real command line, so the defaults the host runs with are the ones here."""
    return main_module._parser().parse_args(
        [
            "run",
            "--problem",
            "misconfig-detection-1",
            "--agent-url",
            agent_url(server),
            "--bridge-token",
            "0123456789abcdef",
            "--bridge-port",
            "0",
            "--poll-seconds",
            "0.01",
        ]
    )


@pytest.mark.parametrize(
    ("class_name", "expect_writes", "expect_stage"),
    [("DetectionTask", False, "diagnose"), ("MitigationTask", True, "full")],
)
async def test_a_read_only_task_serves_a_bridge_that_cannot_write(
    monkeypatch: pytest.MonkeyPatch, class_name: str, expect_writes: bool, expect_stage: str
) -> None:
    """The bridge's write authority comes from the graded task, not from the request.

    Wired here rather than only in the bridge's own tests because the derivation is the
    part that would survive a refactor as `allow_writes=True`.
    """
    seen = fake_aiopslab(monkeypatch, class_name)
    backends: list[dict[str, Any]] = []
    real_backend = lifecycle_module.BridgeBackend

    def record(namespace: str, **kwargs: Any) -> Any:
        backends.append({"namespace": namespace, **kwargs})
        return real_backend(namespace, **kwargs)

    monkeypatch.setattr(lifecycle_module, "BridgeBackend", record)

    with fake_agent([{"run_status": "completed", "status": "tentative_success", **ANOMALY}]) as srv:
        result = await lifecycle_module.run(driver_args(srv))
        posted = srv.posted[0]

    assert backends[0]["allow_writes"] is expect_writes
    assert backends[0]["namespace"] == NAMESPACE
    assert posted["target_stage"] == expect_stage
    assert posted["alert"]["benchmark"] == "aiopslab"
    assert "misconfig-detection-1" not in json.dumps(posted)
    assert seen["max_steps"] == 1
    assert seen["action"] == result["submitted_action"]
    assert result["agent_incident_id"] == "s-1"
    assert result["aiopslab_session"] == {"problem": "recorded"}


async def test_the_analysis_provenance_is_recorded_next_to_the_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reporting an analysis score without it reports the adapter's rules as the agent's."""
    fake_aiopslab(monkeypatch, "AnalysisTask")
    record = {
        "run_status": "completed",
        "status": "tentative_success",
        "root_cause": {
            "affected_resource": "deployment/user-service",
            "summary": "the deployment replicas were set to 0",
        },
    }

    with fake_agent([record]) as server:
        result = await lifecycle_module.run(driver_args(server))

    assert result["analysis_answer"] == {
        "system_level": "Virtualization",
        "fault_type": "Operation Error",
        "source": "normalized",
        "rule": "operator_mutation",
    }
    assert "source" not in result["submitted_action"]
