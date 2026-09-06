"""The adapter seam: the episode driver's orderings, and the boundary that keeps it one.

Every assertion here stands for something that cost a run. The submission ordering, the
deferred run status, the release in a finally and the refusal to release a parked
episode are the four decisions the prototype's episode driver was rewritten around, and
the import boundary is the property the whole rebuild claims.
"""

import ast
import importlib
import json
import pathlib
import subprocess
import sys
from typing import Any

import pytest

from kira.adapters.base import (
    EnvironmentAdapter,
    EpisodeContext,
    ExternalFeedback,
    SubmissionRecord,
)
from kira.adapters.registry import adapter, available_adapters, clear_adapters, get_adapter
from kira.audit.journal import ListSink
from kira.llm.types import ModelRequest, ModelResponse, ToolCall, Usage
from kira.memory.factory import build_storage
from kira.middleware.authority import ApprovalPending
from kira.orchestration import episode as episode_module
from kira.orchestration.episode import open_episode, run_episode
from kira.orchestration.runner import Deployment
from kira.settings.groups import MemorySettings, ServerSettings, VerifySettings
from kira.settings.profiles import load_profile
from kira.settings.root import Settings

PROFILE = load_profile("qwen3.6-35b-a3b", pathlib.Path("configs/models"))

ROOT_CAUSE: dict[str, Any] = {
    "fault_type": "misconfiguration",
    "affected_resource": "shop/Service/user",
    "confidence": 0.8,
    "evidence": ["endpoints empty for svc/user"],
    "summary": "the user Service targets a port nothing listens on",
}
PLAN: dict[str, Any] = {
    "steps": [
        {
            "action": "patch",
            "target": "svc/user",
            "params": {"targetPort": 8080},
            "justification": "repoint the service",
        }
    ],
    "expected_outcome": "svc/user has endpoints",
    "confidence": 0.8,
}
EXECUTION: dict[str, Any] = {
    "succeeded": True,
    "steps": [{"step_index": 0, "succeeded": True, "observation": "patched"}],
    "final_state_summary": "svc/user has endpoints",
    "confidence": 0.8,
}


class FakeTransport:
    name = "fake"
    capabilities = frozenset({"kubernetes", "metrics", "logs", "traces"})
    timeout_seconds = 30

    async def kubectl(self, argv: list[str]) -> str:
        return "ok"


class SubmittingModel:
    """Answers each role by calling its submission tool and nothing else."""

    def __init__(self, *, halt: bool = False) -> None:
        self._halt = halt
        self.calls = 0

    async def complete(self, request: ModelRequest, *, caller: str) -> ModelResponse:
        self.calls += 1
        halted = {**ROOT_CAUSE, "halt": True, "halt_reason": "no cluster"}
        diagnosis = halted if self._halt else ROOT_CAUSE
        name, arguments = {
            "rca": ("submit_root_cause", diagnosis),
            "planning": ("submit_plan", PLAN),
            "execution": ("submit_execution_result", EXECUTION),
        }[caller]
        return ModelResponse(
            text=None,
            tool_calls=(
                ToolCall(
                    id=f"c{self.calls}",
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


class StubAdapter:
    """One episode against a harness that records rather than grades."""

    name = "stub"

    def __init__(self, *, ready: frozenset[str] = frozenset({"diagnosis", "mitigation"})) -> None:
        self._ready = ready
        self.calls: list[str] = []
        self.submitted: list[tuple[str, Any]] = []
        self.session_id = ""
        self.run_status_at_release: str | None = None
        self.store: Any = None
        self.releases = 0

    async def acquire(self) -> EpisodeContext:
        self.calls.append("acquire")
        return EpisodeContext(
            namespace="shop",
            app_name="shop",
            submission_kinds=("diagnosis", "mitigation"),
            external_key="k8s_target_port-misconfig",
        )

    def bind_session(self, session_id: str) -> None:
        self.session_id = session_id

    def build_signal(self, ctx: EpisodeContext) -> dict[str, Any]:
        return {"alertname": "ServiceHealthDegraded", "namespace": ctx.namespace}

    async def await_ready(self, kind: str) -> bool:
        self.calls.append(f"ready:{kind}")
        return kind in self._ready

    def encode(self, kind: str, state: Any, ctx: EpisodeContext) -> Any:
        if kind != "diagnosis":
            return ""
        diagnosis = state.get("root_cause")
        return getattr(diagnosis, "summary", "") or "nothing was concluded"

    async def submit(self, kind: str, payload: Any) -> SubmissionRecord:
        self.calls.append(f"submit:{kind}")
        self.submitted.append((kind, payload))
        return SubmissionRecord(kind=kind, payload=payload)

    def holds_environment_for_grading(self) -> bool:
        return True

    async def fetch_feedback(self) -> ExternalFeedback | None:
        self.calls.append("feedback")
        return None

    async def release(self, *, state: Any) -> None:
        self.releases += 1
        self.calls.append("release")
        record = self.store.get_session(self.session_id)
        self.run_status_at_release = record.run_status if record else None

    def tool_transport(self) -> Any:
        return None

    def evidence_policy(self) -> Any:
        return None

    def extra_tools(self) -> tuple[Any, ...]:
        return ()

    def routes(self) -> Any:
        return None


def deployment(tmp_path: pathlib.Path, *, model: Any, learning: bool = False) -> Deployment:
    settings = Settings(
        server=ServerSettings(workdir=tmp_path),
        memory=MemorySettings(
            enabled=learning, learning_enabled=learning, learning_timeout_s=0, backend="embedded"
        ),
        verify=VerifySettings(enabled=False),
    )
    storage = build_storage(settings)
    return Deployment(
        settings=settings,
        storage=storage,
        completion=model,
        profile=PROFILE,
        transport_for=lambda session_id: FakeTransport(),
        sink=ListSink(),
    )


async def drive(deploy: Deployment, stub: StubAdapter) -> Any:
    stub.store = deploy.storage.sessions
    start = await open_episode(stub, deploy)
    return await run_episode(stub, deploy, start)


async def test_answers_go_over_once_at_the_end_and_the_release_comes_last(
    tmp_path: pathlib.Path,
) -> None:
    """The ordering the whole design turns on.

    The harness grades the first submission it receives per stage, so both answers go
    over once, after the pipeline has settled; the release is not a third submission and
    has to follow the distilling pass; and the run status stays running until the
    environment has actually been handed back, because a poller that saw it go terminal
    would tear the environment down with the release still in flight.
    """
    stub = StubAdapter()
    deploy = deployment(tmp_path, model=SubmittingModel())
    result = await drive(deploy, stub)

    assert stub.calls == [
        "acquire",
        "ready:diagnosis",
        "submit:diagnosis",
        "ready:mitigation",
        "submit:mitigation",
        "feedback",
        "release",
    ]
    assert stub.run_status_at_release == "running"
    assert deploy.storage.sessions.get_session(result.session_id).run_status == "completed"
    assert stub.submitted[0][1] == ROOT_CAUSE["summary"]
    assert stub.submitted[1][1] == ""


async def test_a_halted_diagnosis_is_still_submitted(tmp_path: pathlib.Path) -> None:
    """A halted answer is a failed answer, and a failed answer still has to be graded.

    Withholding it left no result row at all, so the problem vanished from the
    denominator instead of counting against the arm that produced it.
    """
    stub = StubAdapter()
    await drive(deployment(tmp_path, model=SubmittingModel(halt=True)), stub)

    assert [kind for kind, _ in stub.submitted] == ["diagnosis", "mitigation"]
    assert stub.submitted[0][1]


async def test_a_stage_that_never_goes_live_stops_the_plan(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Firing at a stage machine that will reject it leaves the answer silently
    ungraded, so the driver stops and says the grading is incomplete."""
    stub = StubAdapter(ready=frozenset())
    with caplog.at_level("ERROR"):
        await drive(deployment(tmp_path, model=SubmittingModel()), stub)

    assert stub.submitted == []
    assert stub.releases == 1
    assert "never became live" in caplog.text


@pytest.mark.parametrize("failure", ["raises", "hangs"])
async def test_the_environment_is_released_even_when_distilling_does_not_finish(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """A slow or crashed distilling pass must never strand a deployed application."""

    async def broken(*args: Any, **kwargs: Any) -> str:
        if failure == "raises":
            raise RuntimeError("no model endpoint")
        import asyncio  # noqa: PLC0415

        await asyncio.sleep(60)
        return "distilled"

    monkeypatch.setattr(episode_module, "run_learning", broken)
    stub = StubAdapter()
    await drive(deployment(tmp_path, model=SubmittingModel(), learning=True), stub)

    assert stub.calls[-1] == "release"


async def test_an_episode_parked_on_a_person_is_not_released(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The environment belongs to the resume path: releasing it now would tear down the
    cluster the approved write is meant to act on."""

    async def parked(*args: Any, **kwargs: Any) -> Any:
        raise ApprovalPending(
            request_id="r1",
            session_id="s1",
            stage="execution",
            tool_call_id="c1",
            tool_name="exec_kubectl_cmd_safely",
            tool_arguments={"cmd": "kubectl patch svc user -p '{}'"},
        )

    monkeypatch.setattr(episode_module, "run_incident", parked)
    stub = StubAdapter()
    result = await drive(deployment(tmp_path, model=SubmittingModel()), stub)

    assert result.paused is True
    assert stub.releases == 0


async def test_a_crashed_pipeline_leaves_a_run_a_poller_can_tell_is_over(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without this the run stays running forever and the harness blocks until its own
    outer timeout, which costs the whole cell rather than one run.

    A crash ends the episode like any other ending, so it owes the same two things: the
    environment back, or the conductor holds a deployed application until its own
    teardown timeout; and an answer, or the problem produces no result row and leaves
    the denominator instead of counting as the failure it was.
    """

    async def crash(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("boom")

    monkeypatch.setattr(episode_module, "run_incident", crash)
    stub = StubAdapter()
    deploy = deployment(tmp_path, model=SubmittingModel())
    result = await drive(deploy, stub)

    assert result.failed is True
    assert deploy.storage.sessions.get_session(result.session_id).run_status == "failed"
    assert stub.releases == 1
    assert stub.submitted[0] == ("diagnosis", "nothing was concluded")


async def test_the_storage_only_key_is_written_and_never_rendered(
    tmp_path: pathlib.Path,
) -> None:
    """The episode key is the ground-truth fault label. It reaches the session record
    and the prompt view drops it, which is what makes the blinding structural."""
    from kira.roles.prompting import visible_alert  # noqa: PLC0415

    stub = StubAdapter()
    deploy = deployment(tmp_path, model=SubmittingModel())
    start = await open_episode(stub, deploy)

    stored = deploy.storage.sessions.get_session(start.session_id).alert
    assert stored["_external_key"] == "k8s_target_port-misconfig"
    assert "k8s_target_port-misconfig" not in json.dumps(visible_alert(stored))


def test_an_adapter_that_could_not_load_is_reported_rather_than_missing() -> None:
    """Selecting an unavailable adapter is fatal and says why: a deployment that asked
    for an environment and silently did not get one grades nothing."""
    clear_adapters()
    try:

        @adapter("stub")
        def _factory(settings: Settings) -> EnvironmentAdapter:
            return StubAdapter()  # type: ignore[return-value]

        assert available_adapters() == {"stub": "ready"}
        with pytest.raises(ValueError, match="unknown environment adapter 'nope'"):
            get_adapter("nope")
    finally:
        clear_adapters()
        importlib.reload(importlib.import_module("kira.adapters.sregym"))


CONTRACT_MODULES = {"kira.adapters", "kira.adapters.base", "kira.adapters.registry"}


def test_no_core_module_imports_a_concrete_adapter() -> None:
    """Generalises the lint rule to adapters nobody has written yet, and to the
    import-by-name form a lint rule cannot see."""
    root = pathlib.Path("src/kira")
    offenders: list[str] = []
    for path in root.rglob("*.py"):
        if path.parts[:3] == ("src", "kira", "adapters") and len(path.parts) > 4:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            for name in _imported_modules(node):
                if name.startswith("kira.adapters") and name not in CONTRACT_MODULES:
                    offenders.append(f"{path}: {name}")
    assert offenders == []


def test_importing_the_core_does_not_import_an_adapter() -> None:
    """Catches an import that arrives through a re-export rather than a statement."""
    code = (
        "import sys, kira.orchestration.runner, kira.roles.registry, kira.tools.toolsets, "
        "kira.api.app;"
        "print([m for m in sys.modules if m.startswith('kira.adapters.') "
        "and m not in ('kira.adapters.base', 'kira.adapters.registry')])"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout
    assert out.strip() == "[]"


def _imported_modules(node: ast.AST) -> list[str]:
    """Every module one statement pulls in, including the import-by-name form."""
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if isinstance(node, ast.ImportFrom) and node.module:
        return [node.module]
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute | ast.Name)
        and getattr(node.func, "attr", getattr(node.func, "id", "")) == "import_module"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    ):
        return [node.args[0].value]
    return []
