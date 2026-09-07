"""The operations half: the sweep, and the two records every problem leaves behind.

Nothing here needs AIOpsLab, a cluster or an agent. AIOpsLab is faked at the one seam the
driver touches, as in `test_host_lifecycle.py`, but scripted per attempt so the retry
rule can be exercised; the agent is a real HTTP server on an ephemeral loopback port; and
`lifecycle.run` is the real one, so the records asserted here are written by the code
that will write them on a host.
"""

import argparse
import fcntl
import json
import subprocess
import sys
import types
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Any
from urllib.request import Request, urlopen

import pytest

from kira.adapters.aiopslab.host import bridge as bridge_module
from kira.adapters.aiopslab.host import lifecycle as lifecycle_module
from kira.adapters.aiopslab.host import main as main_module
from kira.adapters.aiopslab.host import sweep as sweep_module
from kira.adapters.aiopslab.host.bridge import BridgeBackend, serve_bridge

NAMESPACE = "test-hotel-reservation"
TOKEN = "0123456789abcdef"
ROOT_CAUSE = {
    "root_cause": {
        "halt": False,
        "fault_type": "Misconfiguration",
        "affected_resource": "deployment/user-service",
        "summary": "the deployment replicas were set to 0",
    }
}
TERMINAL_RECORD = {"run_status": "completed", "status": "tentative_success", **ROOT_CAUSE}


# --------------------------------------------------------------------------- the agent


class _AgentHandler(BaseHTTPRequestHandler):
    server: "FakeAgent"

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        self.server.posted.append(json.loads(self.rfile.read(length)))
        self._json({"session_id": f"s-{len(self.server.posted)}", "status": "accepted"})

    def do_GET(self) -> None:
        if self.path == "/healthz":
            self._json({"status": "ok", "version": "0.1.0", "revision": "abc123"})
            return
        self._json(TERMINAL_RECORD)

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

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _AgentHandler)
        self.posted: list[dict[str, Any]] = []


@contextmanager
def fake_agent() -> Iterator[FakeAgent]:
    server = FakeAgent()
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


# ------------------------------------------------------------------------- the harness


class FakeHarness:
    """AIOpsLab at the seam the driver touches, scripted per attempt.

    An outcome of `pass` or `fail` writes the result file the harness would write;
    `crash` writes nothing and raises; `pass+crash` does both, which is the case the
    retry rule exists for; `boom` raises something the driver does not expect. The last
    outcome repeats once the script runs out.
    """

    def __init__(self, outcomes: dict[str, list[str]] | None = None) -> None:
        self.outcomes = {problem: list(script) for problem, script in (outcomes or {}).items()}
        self.calls: list[str] = []

    def next_outcome(self, problem: str) -> str:
        script = self.outcomes.setdefault(problem, ["pass"])
        return script.pop(0) if len(script) > 1 else script[0]


def install_fake_aiopslab(monkeypatch: pytest.MonkeyPatch, harness: FakeHarness) -> None:
    """Install a fake `aiopslab.orchestrator` that grades what the driver submits."""

    def problem_instance(problem_id: str) -> Any:
        task = problem_id.rsplit("-", 1)[0].rsplit("-", 1)[-1].capitalize()
        instance = type("K8sProblem", (type(f"{task}Task", (), {}),), {})()
        instance.namespace = NAMESPACE
        instance.inject_fault = lambda: None
        return instance

    class Registry:
        def get_problem_instance(self, problem_id: str) -> Any:
            return problem_instance(problem_id)

    class Orchestrator:
        def __init__(self, results_dir: Any) -> None:
            self.results_dir = Path(results_dir)
            self.probs = Registry()
            self.session: Any = None
            self.agent: Any = None
            self.problem_id = ""

        def register_agent(self, agent: Any, name: str) -> None:
            return None

        def init_problem(self, problem_id: str) -> tuple[str, str, dict[str, str]]:
            self.problem_id = problem_id
            instance = self.probs.get_problem_instance(problem_id)
            instance.inject_fault()
            self.session = types.SimpleNamespace(
                problem=instance, to_dict=lambda: {"problem": problem_id}
            )
            return "The hotel app is returning errors", "instructions", {"submit": "submit()"}

        async def start_problem(self, max_steps: int) -> dict[str, Any]:
            action = await self.agent.get_action("")
            harness.calls.append(self.problem_id)
            outcome = harness.next_outcome(self.problem_id)
            if outcome.startswith(("pass", "fail")):
                graded = outcome.startswith("pass")
                results = (
                    {"Detection Accuracy": "Correct" if graded else "Incorrect"}
                    if "detection" in self.problem_id
                    else {"success": graded}
                )
                path = self.results_dir / f"{self.problem_id}-{len(harness.calls)}.json"
                path.write_text(json.dumps({"results": results}), encoding="utf-8")
            if outcome.endswith("crash"):
                raise RuntimeError("AIOpsLab teardown failed")
            if outcome == "boom":
                raise KeyError("something nobody anticipated")
            return {"action": action}

    package = types.ModuleType("aiopslab")
    orchestrator_module = types.ModuleType("aiopslab.orchestrator")
    orchestrator_module.Orchestrator = Orchestrator  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "aiopslab", package)
    monkeypatch.setitem(sys.modules, "aiopslab.orchestrator", orchestrator_module)


def run_args(server: FakeAgent, results_dir: Path, problem: str) -> argparse.Namespace:
    """Parse a real `run` command line, so the host's own defaults are the ones tested."""
    return main_module._parser().parse_args(
        [
            "run",
            "--problem",
            problem,
            "--agent-url",
            agent_url(server),
            "--bridge-token",
            TOKEN,
            "--bridge-port",
            "0",
            "--poll-seconds",
            "0.01",
            "--results-dir",
            str(results_dir),
        ]
    )


# --------------------------------------------------------------------------- the records


def post_action(port: int, payload: dict[str, Any], token: str = TOKEN) -> None:
    request = Request(
        f"http://127.0.0.1:{port}/v1/action",
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "X-Kira-Session": "sess-1",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=5) as response:
            response.read()
    except OSError:
        return


def test_the_bridge_records_every_action_it_was_asked_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    """The credential-holding process keeps its own account, and still not on the console.

    Its value is that it is independent of the agent's audit log: a cluster action with no
    line here was not taken, whatever the agent recorded about it.
    """
    monkeypatch.setattr(
        bridge_module.subprocess,
        "run",
        lambda *_a, **_k: subprocess.CompletedProcess([], 0, "pod/x\n", ""),
    )
    access_log = tmp_path / "bridge.log"

    def action(namespace: str = NAMESPACE) -> dict[str, Any]:
        return {
            "namespace": namespace,
            "server": "kubectl",
            "tool": "exec_kubectl_cmd_safely",
            "arguments": {"cmd": "kubectl get pods"},
        }

    with serve_bridge(BridgeBackend(NAMESPACE), TOKEN, port=0, access_log=access_log) as server:
        port = int(server.server_address[1])
        post_action(port, action())
        post_action(port, action(namespace="kube-system"))
        post_action(port, action(), token="wrong-token")

    lines = [json.loads(line) for line in access_log.read_text(encoding="utf-8").splitlines()]

    assert [line["decision"] for line in lines] == ["served", "refused", "unauthorized"]
    assert [line["status"] for line in lines] == [200, 400, 401]
    assert lines[0]["op"] == "kubectl.exec_kubectl_cmd_safely"
    assert lines[0]["namespace"] == NAMESPACE
    assert lines[0]["session"] == "sess-1"
    assert lines[0]["port"] == port
    assert lines[0]["timestamp"].endswith("Z")
    assert isinstance(lines[0]["duration_ms"], float)
    # Not the arguments: they are large, and the agent's audit log already holds them.
    assert "kubectl get pods" not in access_log.read_text(encoding="utf-8")
    assert "GET /" not in capfd.readouterr().err


async def test_one_problem_leaves_both_records_beside_the_harness_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A single `run` writes what the sweep writes: nothing here is sweep-only."""
    install_fake_aiopslab(monkeypatch, FakeHarness())

    with fake_agent() as server:
        await lifecycle_module.run(run_args(server, tmp_path, "misconfig-analysis-1"))

    record = json.loads((tmp_path / "kira-submission.json").read_text(encoding="utf-8"))

    assert (tmp_path / "bridge.log").exists()
    assert record["problem"] == "misconfig-analysis-1"
    assert record["task_type"] == "analysis"
    assert record["namespace"] == NAMESPACE
    assert record["agent_incident_id"] == "s-1"
    assert record["alert"] == server.posted[0]["alert"]
    assert record["incident_record"] == TERMINAL_RECORD
    assert "submit(" in record["submitted_action"]
    # The graded answer, and who decided it: this one is the adapter's rule, not the
    # model's classification, and an analysis score that cannot say so is not the
    # agent's score.
    assert record["analysis_answer"] == {
        "system_level": "Virtualization",
        "fault_type": "Operation Error",
        "source": "normalized",
        "rule": "operator_mutation",
    }
    assert record["started_utc"].endswith("Z")
    assert record["ended_utc"] >= record["started_utc"]


# ----------------------------------------------------------------------------- the sweep


def sweep_args(server: FakeAgent, sweep_dir: Path, **overrides: str) -> argparse.Namespace:
    """Parse a real `sweep` command line, defaults included."""
    argv = [
        "sweep",
        "--sweep-dir",
        str(sweep_dir),
        "--agent-url",
        agent_url(server),
        "--bridge-token",
        TOKEN,
        "--bridge-port",
        "0",
        "--poll-seconds",
        "0.01",
        "--retry-delay",
        "0",
    ]
    for name, value in overrides.items():
        flag = f"--{name.replace('_', '-')}"
        argv.extend([flag] if value == "" else [flag, value])
    return main_module._parser().parse_args(argv)


def problems_file(tmp_path: Path, *problems: str) -> Path:
    path = tmp_path / "problems.txt"
    path.write_text("# a comment\n\n" + "".join(f"{p}\n" for p in problems), encoding="utf-8")
    return path


def status_rows(sweep_dir: Path) -> list[dict[str, str]]:
    lines = (sweep_dir / "status.tsv").read_text(encoding="utf-8").splitlines()
    header = lines[0].split("\t")
    return [dict(zip(header, line.split("\t"), strict=True)) for line in lines[1:]]


def install_fake_registry(monkeypatch: pytest.MonkeyPatch, deployments: dict[str, str]) -> None:
    """Install a fake `ProblemRegistry` reporting each problem's deployment kind."""

    class ProblemRegistry:
        def get_problem_ids(self) -> list[str]:
            return list(deployments)

        def get_problem_deployment(self, problem_id: str) -> str:
            return deployments[problem_id]

    module = types.ModuleType("aiopslab.orchestrator.problems.registry")
    module.ProblemRegistry = ProblemRegistry  # type: ignore[attr-defined]
    for name in ("aiopslab.orchestrator.problems", "aiopslab.orchestrator", "aiopslab"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "aiopslab.orchestrator.problems.registry", module)


def test_a_problem_list_and_a_task_filter_choose_what_runs(tmp_path: Path) -> None:
    """Selection needs no AIOpsLab: a sweep can be planned before a host has one."""
    listed = problems_file(
        tmp_path,
        "noop_detection_hotel_reservation-1",
        "misconfig_app_hotel_res-mitigation-1",
        "scale_pod-localization-2",
    )

    every, excluded = sweep_module.select_problems(sweep_module.TASK_TYPES, listed)
    mitigation, _ = sweep_module.select_problems(["mitigation"], listed)

    assert every == [
        sweep_module.Selected("noop_detection_hotel_reservation-1", "detection"),
        sweep_module.Selected("misconfig_app_hotel_res-mitigation-1", "mitigation"),
        sweep_module.Selected("scale_pod-localization-2", "localization"),
    ]
    assert excluded == []
    assert [item.problem for item in mitigation] == ["misconfig_app_hotel_res-mitigation-1"]


def test_the_registry_supplies_the_problems_and_names_the_docker_backed_ones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Excluded, not dropped: a denominator that omits them reports a smaller benchmark."""
    install_fake_registry(
        monkeypatch,
        {
            "misconfig-detection-1": "k8s",
            "flower_client-detection-1": "docker",
            "scale_pod-mitigation-1": "k8s",
        },
    )

    selected, excluded = sweep_module.select_problems(sweep_module.TASK_TYPES)

    assert [item.problem for item in selected] == [
        "misconfig-detection-1",
        "scale_pod-mitigation-1",
    ]
    assert excluded == ["flower_client-detection-1"]


def test_a_problem_naming_no_task_fails_before_anything_is_deployed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot infer the AIOpsLab task type"):
        sweep_module.select_problems(
            sweep_module.TASK_TYPES, problems_file(tmp_path, "misconfig-detection-1", "puzzling-4")
        )


@pytest.mark.parity
async def test_a_graded_result_is_never_run_again_and_only_a_non_grade_is_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The rule the archived sweep's numbers depend on.

    Retrying a graded failure would replace one cell with a second sample of the model,
    which is a better score manufactured out of variance rather than a measurement.
    """
    caplog.set_level("INFO")
    harness = FakeHarness(
        {
            "a-mitigation-1": ["crash", "pass"],
            "b-mitigation-1": ["fail"],
            "c-mitigation-1": ["pass+crash"],
        }
    )
    install_fake_aiopslab(monkeypatch, harness)
    listed = problems_file(tmp_path, "a-mitigation-1", "b-mitigation-1", "c-mitigation-1")

    with fake_agent() as server:
        code = await sweep_module.run_sweep(
            sweep_args(server, tmp_path / "sweep", problems=str(listed), max_attempts="2")
        )

    rows = {row["problem"]: row for row in status_rows(tmp_path / "sweep")}

    assert code == 0
    assert harness.calls.count("a-mitigation-1") == 2
    assert rows["a-mitigation-1"]["attempts"] == "2"
    assert rows["a-mitigation-1"]["grade"] == "pass"
    # A graded failure is the outcome, and a second attempt would be a second sample.
    assert harness.calls.count("b-mitigation-1") == 1
    assert rows["b-mitigation-1"] == {**rows["b-mitigation-1"], "attempts": "1", "grade": "fail"}
    # And so is a grade AIOpsLab persisted before its own cleanup failed.
    assert harness.calls.count("c-mitigation-1") == 1
    assert rows["c-mitigation-1"]["grade"] == "pass"
    assert rows["c-mitigation-1"]["exit_code"] == "1"
    assert "keeping the externally graded pass result" in caplog.text


async def test_every_problem_gets_a_row_a_record_and_a_line_in_the_sweep_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """What a run leaves behind is the experiment: the layout is part of the interface."""
    caplog.set_level("INFO")
    install_fake_aiopslab(monkeypatch, FakeHarness())
    sweep_dir = tmp_path / "sweep"
    listed = problems_file(tmp_path, "misconfig-detection-1", "scale_pod-mitigation-1")

    with fake_agent() as server:
        await sweep_module.run_sweep(sweep_args(server, sweep_dir, problems=str(listed)))

    rows = status_rows(sweep_dir)
    problem_dir = sweep_dir / "results" / "misconfig-detection-1"
    metadata = dict(
        line.split("=", 1)
        for line in (sweep_dir / "metadata.env").read_text(encoding="utf-8").splitlines()
    )

    assert (sweep_dir / "status.tsv").read_text(encoding="utf-8").splitlines()[0].split(
        "\t"
    ) == list(sweep_module.STATUS_COLUMNS)
    assert [row["problem"] for row in rows] == ["misconfig-detection-1", "scale_pod-mitigation-1"]
    assert rows[0]["index"] == "1"
    assert rows[0]["task_type"] == "detection"
    assert rows[0]["grade"] == "pass"
    assert rows[0]["agent_incident_id"] == "s-1"
    assert rows[0]["started_utc"].endswith("Z")
    assert rows[0]["ended_utc"].endswith("Z")
    assert float(rows[0]["duration_seconds"]) >= 0
    assert Path(rows[0]["result_file"]).parent == problem_dir
    assert (problem_dir / "kira-submission.json").exists()
    assert (problem_dir / "bridge.log").exists()
    assert metadata["problem_count"] == "2"
    assert metadata["agent_revision"] == "abc123"
    assert metadata["completed_utc"].endswith("Z")
    assert "finished misconfig-detection-1" in (sweep_dir / "sweep.log").read_text(encoding="utf-8")


async def test_a_graded_problem_is_skipped_on_a_resume_unless_a_rerun_is_asked_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reusing the sweep directory continues the sweep rather than repeating it."""
    harness = FakeHarness()
    install_fake_aiopslab(monkeypatch, harness)
    sweep_dir = tmp_path / "sweep"
    listed = problems_file(tmp_path, "misconfig-detection-1")

    with fake_agent() as server:
        args = sweep_args(server, sweep_dir, problems=str(listed))
        await sweep_module.run_sweep(args)
        await sweep_module.run_sweep(args)
        resumed = list(harness.calls)
        await sweep_module.run_sweep(sweep_args(server, sweep_dir, problems=str(listed), rerun=""))

    assert resumed == ["misconfig-detection-1"]
    assert harness.calls == ["misconfig-detection-1", "misconfig-detection-1"]
    assert len(status_rows(sweep_dir)) == 2


async def test_a_second_sweep_on_the_same_host_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AIOpsLab deploys into shared cluster state: two sweeps make two spoiled arms."""
    install_fake_aiopslab(monkeypatch, FakeHarness())
    listed = problems_file(tmp_path, "misconfig-detection-1")
    lock = tmp_path / ".sweep.lock"
    holder = lock.open("a", encoding="utf-8")
    fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)

    try:
        with fake_agent() as server, pytest.raises(RuntimeError, match="another AIOpsLab sweep"):
            await sweep_module.run_sweep(
                sweep_args(server, tmp_path / "sweep", problems=str(listed))
            )
    finally:
        holder.close()


async def test_one_problem_raising_does_not_end_the_sweep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """An eight-hour sweep must not be lost to one problem nobody anticipated."""
    caplog.set_level("INFO")
    harness = FakeHarness({"a-mitigation-1": ["boom"]})
    install_fake_aiopslab(monkeypatch, harness)
    sweep_dir = tmp_path / "sweep"
    listed = problems_file(tmp_path, "a-mitigation-1", "b-mitigation-1")

    with fake_agent() as server:
        code = await sweep_module.run_sweep(
            sweep_args(server, sweep_dir, problems=str(listed), max_attempts="1")
        )

    rows = {row["problem"]: row for row in status_rows(sweep_dir)}

    assert code == 0
    assert rows["a-mitigation-1"]["grade"] == "error"
    assert rows["a-mitigation-1"]["exit_code"] == "1"
    assert rows["b-mitigation-1"]["grade"] == "pass"
    assert "something nobody anticipated" in caplog.text
