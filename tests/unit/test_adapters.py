"""The adapter seam: the episode driver's orderings, and the boundary that keeps it one.

Every assertion here stands for something that cost a run. The submission ordering, the
deferred run status, the release in a finally and the refusal to release a parked
episode are the four decisions the prototype's episode driver was rewritten around, and
the import boundary is the property the whole rebuild claims.
"""

import ast
import asyncio
import json
import pathlib
import subprocess
import sys
import tomllib
from typing import Any

import httpx
import pytest
from fastapi import APIRouter

from maver.adapters.base import (
    AdapterTool,
    EnvironmentAdapter,
    EpisodeContext,
    ExternalFeedback,
    LifecycleRunsElsewhere,
    RoleOverlay,
    SubmissionRecord,
)
from maver.adapters.registry import adapter, available_adapters, clear_adapters, get_adapter
from maver.api.app import build_deployment, create_app
from maver.audit.durable_sink import DurableAuditSink
from maver.audit.export import export_incident
from maver.audit.journal import ListSink
from maver.llm.types import ModelRequest, ModelResponse, ToolCall, Usage
from maver.memory.factory import build_storage
from maver.memory.ports import SessionMemory
from maver.middleware.authority import ApprovalPending
from maver.orchestration import episode as episode_module
from maver.orchestration.episode import open_episode, run_episode
from maver.orchestration.runner import Deployment
from maver.settings.groups import (
    IntakeSettings,
    MemorySettings,
    ServerSettings,
    VerifySettings,
)
from maver.settings.profiles import load_profile
from maver.settings.root import Settings
from maver.tools.registry import ToolSpec

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
        self.seen: list[tuple[str, ModelRequest]] = []

    async def complete(self, request: ModelRequest, *, caller: str) -> ModelResponse:
        self.calls += 1
        self.seen.append((caller, request))
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

    def __init__(
        self,
        *,
        ready: frozenset[str] = frozenset({"diagnosis", "mitigation"}),
        withheld: tuple[str, ...] = (),
    ) -> None:
        self._ready = ready
        self._withheld = withheld
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

    def withheld_records(self, ctx: EpisodeContext, records: Any) -> tuple[str, ...]:
        self.calls.append("withheld")
        return self._withheld

    def tool_transport(self) -> Any:
        return None

    def evidence_policy(self) -> Any:
        return None

    def extra_tools(self) -> tuple[Any, ...]:
        return ()

    def role_overlay(self, role: str, state: Any) -> Any:
        return None

    def routes(self) -> Any:
        # An adapter that opens episodes is one an outside driver reaches, and the
        # driver refuses to open one for an adapter with no route at all.
        return APIRouter()


def deployment(
    tmp_path: pathlib.Path, *, model: Any, learning: bool = False, memory: bool = False
) -> Deployment:
    settings = Settings(
        server=ServerSettings(workdir=tmp_path),
        memory=MemorySettings(
            enabled=learning or memory,
            learning_enabled=learning,
            learning_timeout_s=0,
            backend="embedded",
            # Off, because the corpus below is two records: the floor is calibrated
            # against hundreds, and on a pair nothing clears it.
            min_score_per_token=0.0,
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
        # Before the pipeline, because what a run may not retrieve has to be settled
        # before the first role can retrieve anything.
        "withheld",
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
    from maver.roles.prompting import visible_alert  # noqa: PLC0415

    stub = StubAdapter()
    deploy = deployment(tmp_path, model=SubmittingModel())
    start = await open_episode(stub, deploy)

    stored = deploy.storage.sessions.get_session(start.session_id).alert
    assert stored["_external_key"] == "k8s_target_port-misconfig"
    assert "k8s_target_port-misconfig" not in json.dumps(visible_alert(stored))


def test_an_adapter_that_could_not_load_is_reported_rather_than_missing(
    adapter_registry: None,
) -> None:
    """Selecting an unavailable adapter is fatal and says why: a deployment that asked
    for an environment and silently did not get one grades nothing."""
    clear_adapters()

    @adapter("stub")
    def _factory(settings: Settings) -> EnvironmentAdapter:
        return StubAdapter()  # type: ignore[return-value]

    assert available_adapters() == {"stub": "ready"}
    with pytest.raises(ValueError, match="unknown environment adapter 'nope'"):
        get_adapter("nope")


CONTRACT_MODULES = {"maver.adapters", "maver.adapters.base", "maver.adapters.registry"}


def test_no_core_module_imports_a_concrete_adapter() -> None:
    """Generalises the lint rule to adapters nobody has written yet, and to the
    import-by-name form a lint rule cannot see."""
    root = pathlib.Path("src/maver")
    offenders: list[str] = []
    for path in root.rglob("*.py"):
        if path.parts[:3] == ("src", "maver", "adapters") and len(path.parts) > 4:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            for name in _imported_modules(node):
                if name.startswith("maver.adapters") and name not in CONTRACT_MODULES:
                    offenders.append(f"{path}: {name}")
    assert offenders == []


def test_every_in_tree_adapter_package_is_banned_from_core_imports() -> None:
    """The lint rule and the walk above have to name the same set.

    The walk catches a core module that imports an adapter; the ban list is what catches
    it in the editor, before it is written. A new adapter package added without its line
    is a rule that silently covers one environment and not the other, so the list is
    derived from what is on disk rather than maintained by memory.
    """
    packages = {
        path.name
        for path in pathlib.Path("src/maver/adapters").iterdir()
        if (path / "__init__.py").exists()
    }
    banned = set(
        tomllib.loads(pathlib.Path("pyproject.toml").read_text(encoding="utf-8"))["tool"]["ruff"][
            "lint"
        ]["flake8-tidy-imports"]["banned-api"]
    )

    assert packages
    assert {f"maver.adapters.{name}" for name in packages} <= banned


def test_importing_the_core_does_not_import_an_adapter() -> None:
    """Catches an import that arrives through a re-export rather than a statement."""
    code = (
        "import sys, maver.orchestration.runner, maver.roles.registry, maver.tools.toolsets, "
        "maver.api.app;"
        "print([m for m in sys.modules if m.startswith('maver.adapters.') "
        "and m not in ('maver.adapters.base', 'maver.adapters.registry')])"
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


PROBE_QUERY = "readiness probe connection refused on the wrong port"


class RetrievingModel:
    """Reads the corpus by every path a role has, then submits."""

    def __init__(self, own: str, twin: str) -> None:
        self.calls = 0
        self.transcripts: list[list[dict[str, Any]]] = []
        self._own = own
        self._twin = twin
        self._read = False

    async def complete(self, request: ModelRequest, *, caller: str) -> ModelResponse:
        self.calls += 1
        self.transcripts.append([dict(message) for message in request.messages])
        if caller == "rca" and not self._read:
            self._read = True
            return _calls(
                self.calls,
                ("memory_search", {"query": PROBE_QUERY, "scope": "evidence"}),
                ("open_memory", {"session_id": self._own}),
                ("open_memory", {"session_id": self._twin}),
                ("session_trace_summary", {"session_id": self._own}),
                ("session_trace_summary", {"session_id": self._twin}),
            )
        name, arguments = {
            "rca": ("submit_root_cause", ROOT_CAUSE),
            "planning": ("submit_plan", PLAN),
            "execution": ("submit_execution_result", EXECUTION),
        }[caller]
        return _calls(self.calls, (name, arguments))


def _calls(index: int, *wanted: tuple[str, dict[str, Any]]) -> ModelResponse:
    return ModelResponse(
        text=None,
        tool_calls=tuple(
            ToolCall(
                id=f"c{index}-{position}",
                name=name,
                arguments=arguments,
                raw_arguments=json.dumps(arguments),
            )
            for position, (name, arguments) in enumerate(wanted)
        ),
        finish_reason="stop",
        reasoning=None,
        usage=Usage(),
        duration_ms=1,
        attempts=1,
    )


def _corpus(deploy: Deployment) -> tuple[str, str]:
    """One record from the episode's own problem, and one from a twin of it.

    Sessions as well as records, because reading a past run's trace goes to the session
    store: a record whose session is missing is not-found for a reason that has nothing
    to do with being withheld, and the control has to tell the two apart.
    """
    written = []
    for text in (
        "readiness probe on payment-service refused on 8080, the container listens on 9090",
        "readiness probe on checkout-service refused on 8080, the container listens on 9090",
    ):
        session_id = deploy.storage.sessions.create_session(
            {"alertname": "ProbeFailure"}
        ).session_id
        deploy.storage.index.upsert_summary(
            SessionMemory(
                session_id=session_id,
                evidence_text=text,
                lesson_text=text,
                incident_summary=text,
                lifecycle_state="human_verified",
                fault_category="misconfiguration",
                alert_name="ProbeFailure",
                namespace="shop",
                affected_resource="shop/Deployment/payment",
            )
        )
        written.append(session_id)
    return written[0], written[1]


async def test_an_episode_denies_its_withheld_records_by_every_path_a_role_has(
    tmp_path: pathlib.Path,
) -> None:
    """What the adapter withholds, no role can reach, and only what it withheld is gone.

    Search filters the pool, and both tools that take an identifier accept a raw one, so
    a run that filtered search alone would leave the record one well-formed call away.
    The twin is the control: an exclusion that also removed the cross-application twin
    would be measuring a smaller corpus rather than generalisation.
    """
    deploy = deployment(tmp_path, model=None, memory=True)
    own, twin = _corpus(deploy)
    model = RetrievingModel(own, twin)
    object.__setattr__(deploy, "completion", model)

    await drive(deploy, StubAdapter(withheld=(own,)))

    answers = {
        message.get("tool_call_id"): str(message.get("content"))
        for transcript in model.transcripts
        for message in transcript
        if message.get("role") == "tool"
    }
    searched, own_record, twin_record, own_trace, twin_trace = (
        answers[f"c1-{position}"] for position in range(5)
    )
    assert "checkout-service" in searched, "the twin is still a precedent"
    assert "payment-service" not in searched
    assert json.loads(own_record)["error"] == "not found"
    assert json.loads(twin_record)["evidence_text"].startswith("readiness probe on checkout")
    assert json.loads(own_trace)["error"] == "not found"
    assert json.loads(twin_trace).get("error") is None


async def test_the_episode_records_what_it_was_denied(tmp_path: pathlib.Path) -> None:
    """A cell's withheld set has to be reconstructible from the run's own log.

    The second pass records how many records it was denied and never which, and the
    archive could not say what any one cell was denied at all.
    """
    deploy = deployment(tmp_path, model=SubmittingModel(), memory=True)
    stub = StubAdapter(withheld=("sess-own", "sess-twin"))

    await drive(deploy, stub)

    acquired = [
        event.payload
        for event in deploy.sink.events
        if event.payload.kind == "environment" and event.payload.action == "acquired"
    ]
    assert len(acquired) == 1
    assert json.loads(acquired[0].detail)["withheld"] == ["sess-own", "sess-twin"]


PROBE_SUFFIX = "## Environment contract\nProbe the entry point before concluding."


async def _probe(ctx: Any, service: str) -> str:
    """The adapter's own tool, which exists in no shipped catalogue."""
    return await ctx.transport.kubectl(["get", "svc", service])


PROBE_TOOL = ToolSpec(
    name="probe_service",
    parameters={
        "type": "object",
        "properties": {"service": {"type": "string"}},
        "required": ["service"],
    },
    safety="read_only",
    dispatch=_probe,
    descriptions={"probe_service": {"default": "Probe one Service. Takes its name."}},
)


class ProcessTransport:
    """The transport an environment supplies to a process it does not drive episodes on."""

    name = "process"
    capabilities = frozenset({"kubernetes"})
    timeout_seconds = 30

    def __init__(self) -> None:
        self.served: list[list[str]] = []

    async def kubectl(self, argv: list[str]) -> str:
        self.served.append(list(argv))
        return "ok"


class ProcessAdapter:
    """An environment whose episodes are driven by another process entirely.

    It supplies the agent half — a transport, a tool and an overlay — and refuses the
    lifecycle half, because nothing here holds the environment. It mounts no episode
    route, which is what says so.
    """

    name = "process"

    def __init__(self) -> None:
        self.transport = ProcessTransport()

    def tool_transport(self) -> Any:
        return self.transport

    def evidence_policy(self) -> Any:
        return None

    def extra_tools(self) -> tuple[AdapterTool, ...]:
        return (AdapterTool(spec=PROBE_TOOL, roles=frozenset({"rca"})),)

    def role_overlay(self, role: str, state: Any) -> Any:
        return RoleOverlay(system_suffix=PROBE_SUFFIX) if role == "rca" else None

    def routes(self) -> Any:
        return None

    def build_signal(self, ctx: EpisodeContext) -> dict[str, Any]:
        raise LifecycleRunsElsewhere("the driver builds it")

    def bind_session(self, session_id: str) -> None:
        raise LifecycleRunsElsewhere("the driver binds it")

    def withheld_records(self, ctx: EpisodeContext, records: Any) -> tuple[str, ...]:
        raise LifecycleRunsElsewhere("the driver decides it")

    def encode(self, kind: str, state: Any, ctx: EpisodeContext) -> Any:
        raise LifecycleRunsElsewhere("the driver encodes it")

    def holds_environment_for_grading(self) -> bool:
        raise LifecycleRunsElsewhere("the driver knows")

    async def acquire(self) -> EpisodeContext:
        raise LifecycleRunsElsewhere("the driver acquires it")

    async def await_ready(self, kind: str) -> bool:
        raise LifecycleRunsElsewhere("the driver waits")

    async def submit(self, kind: str, payload: Any) -> SubmissionRecord:
        raise LifecycleRunsElsewhere("the driver submits it")

    async def fetch_feedback(self) -> ExternalFeedback | None:
        raise LifecycleRunsElsewhere("the driver grades it")

    async def release(self, *, state: Any) -> None:
        raise LifecycleRunsElsewhere("the driver releases it")


def process_settings(tmp_path: pathlib.Path, *names: str) -> Settings:
    return Settings(
        server=ServerSettings(workdir=tmp_path),
        intake=IntakeSettings(adapters=frozenset(names)),
        memory=MemorySettings(enabled=False, learning_enabled=False),
        verify=VerifySettings(enabled=False),
    )


async def test_an_adapter_with_no_episode_route_serves_the_generic_intake(
    tmp_path: pathlib.Path, adapter_registry: None
) -> None:
    """The second shape an adapter can be, end to end through the ordinary route.

    An environment that drives us rather than the other way round never opens an
    episode here: it posts one incident to the generic intake and pulls the answer back
    out. So everything it supplies has to be bound for the process instead — the
    prototype ran its whole container in that mode — and this proves the three that
    reach a role: its transport carries the tool calls, its tool is offered to the role
    it named, and its overlay reaches the system prompt.
    """

    @adapter("process")
    def _factory(settings: Settings) -> EnvironmentAdapter:
        return ProcessAdapter()  # type: ignore[return-value]

    deploy = build_deployment(process_settings(tmp_path, "process"))
    model = SubmittingModel()
    object.__setattr__(deploy, "completion", model)
    assert deploy.adapter is not None and deploy.adapter.name == "process"

    app = create_app(deployment=deploy, schedule_resume=lambda request_id: None)
    transport = deploy.transport_for("s")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://maver"
    ) as http:
        accepted = await http.post(
            "/api/v1/incidents", json={"alert": {"alertname": "X", "namespace": "shop"}}
        )
        assert accepted.status_code == 202
        while pending := {t for t in app.state.service.runs.tasks() if not t.done()}:
            await asyncio.gather(*pending, return_exceptions=True)

    assert isinstance(transport, ProcessTransport)
    assert transport.served, "the incident ran on the environment's transport"
    opening = next(request for caller, request in model.seen if caller == "rca")
    assert "probe_service" in {tool.name for tool in opening.tools}
    assert f"\n\n{PROBE_SUFFIX}\n\n" in str(opening.system)

    # Verify the actual service composition preserves this completed incident offline.
    assert isinstance(deploy.sink, DurableAuditSink)
    deploy.sink.close()
    bundle = export_incident(
        tmp_path / "audit", accepted.json()["session_id"], tmp_path / "audit-export"
    )
    assert bundle["evidence_integrity"] == "complete"
    assert bundle["event_kinds"]["model_call"] == len(model.seen)
    assert bundle["started_without_terminal"]["model"] == []
    assert bundle["database_projection"]["coverage"] == "unknown"
    assert bundle["artifacts"], "full request and response evidence survives service composition"
    deploy.storage.close()


def test_an_adapter_that_drives_its_own_episodes_is_not_process_bound(
    tmp_path: pathlib.Path,
) -> None:
    """The other shape, and the one every published number was taken under.

    An adapter that mounts an episode route is bound per episode by the episode driver,
    with the correlation id that episode acquired. Binding it for the process as well
    would give every incident a transport that belongs to one episode.
    """
    deploy = build_deployment(process_settings(tmp_path, "sregym"))

    assert deploy.adapter is None
    assert deploy.tool_extras == ()


async def test_an_episode_cannot_be_opened_for_an_adapter_that_drives_none(
    tmp_path: pathlib.Path,
) -> None:
    """Unreachable through a route, because there is no route. The refusal is for the
    next caller: an episode opened here would acquire nothing and release nothing."""
    deploy = deployment(tmp_path, model=SubmittingModel())

    with pytest.raises(LifecycleRunsElsewhere, match="mounts no episode route"):
        await open_episode(ProcessAdapter(), deploy)  # type: ignore[arg-type]
