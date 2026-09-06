"""One incident, end to end, with a fake model and a fake cluster.

The first test that exercises the composition root: real roles, real routing, a real
SQLite store, and nothing global. What it proves is that the pieces compose — that a
role's context can be built from a deployment's resources and that what the pipeline
concluded reaches the record a later incident reads.
"""

import contextlib
import json
import pathlib
from typing import Any

import pytest

from kira.audit.journal import Journal, ListSink
from kira.contracts import SCHEMA_VERSION
from kira.contracts.events import InterventionEvent
from kira.llm.types import ModelRequest, ModelResponse, ToolCall, Usage
from kira.memory.factory import build_storage
from kira.orchestration.runner import (
    Deployment,
    incident_context,
    run_incident,
    tool_context_for,
)
from kira.settings.authority import AuthoritySettings
from kira.settings.groups import (
    MemorySettings,
    PipelineSettings,
    ServerSettings,
    VerifySettings,
)
from kira.settings.profiles import load_profile
from kira.settings.root import Settings
from kira.tools.kubernetes import exec_kubectl_cmd_safely

PROFILE = load_profile("qwen3.6-35b-a3b", pathlib.Path("configs/models"))

ALERT = {"alertname": "HighErrorRate", "namespace": "shop", "_problem_id": "leak-the-answer"}

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
            "justification": "repoint the service at the port the container listens on",
        }
    ],
    "expected_outcome": "svc/user has endpoints and the 5xx rate falls",
    "confidence": 0.8,
}
EXECUTION: dict[str, Any] = {
    "succeeded": True,
    "steps": [
        {
            "step_index": 0,
            "succeeded": True,
            "observation": "targetPort was 8081; patched to 8080; endpoints now non-empty",
        }
    ],
    "final_state_summary": "svc/user has endpoints and the 5xx rate fell",
    "confidence": 0.8,
}

SUBMISSIONS = {
    "rca": ("submit_root_cause", ROOT_CAUSE),
    "planning": ("submit_plan", PLAN),
    "execution": ("submit_execution_result", EXECUTION),
}


class FakeTransport:
    """A cluster that answers reads and records the namespaces it was asked about."""

    name = "fake"
    capabilities = frozenset({"kubernetes", "metrics", "logs", "traces"})
    timeout_seconds = 30

    def __init__(self) -> None:
        self.commands: list[list[str]] = []

    async def kubectl(self, argv: list[str]) -> str:
        self.commands.append(list(argv))
        return "targetPort: 8081"


class FakeModel:
    """Reads once where a role demands it, then submits that role's artifact."""

    def __init__(self) -> None:
        self.callers: list[str] = []

    async def complete(self, request: ModelRequest, *, caller: str) -> ModelResponse:
        self.callers.append(caller)
        offered = {tool.name for tool in request.tools}
        name, arguments = SUBMISSIONS[caller]
        if caller == "execution" and self.callers.count("execution") == 1:
            name, arguments = "exec_kubectl_cmd_safely", {"command": "kubectl get svc user"}
        elif name not in offered:
            raise AssertionError(f"{caller} was not offered {name}")
        return ModelResponse(
            text=None,
            tool_calls=(
                ToolCall(
                    id=f"c{len(self.callers)}",
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


def deployment(tmp_path: pathlib.Path, **overrides: Any) -> tuple[Deployment, ListSink, FakeModel]:
    settings = Settings(
        server=ServerSettings(workdir=tmp_path),
        pipeline=PipelineSettings(**overrides.pop("pipeline", {})),
        memory=MemorySettings(enabled=False, learning_enabled=False),
        verify=VerifySettings(enabled=False),
    )
    sink = ListSink()
    model = FakeModel()
    transport = overrides.pop("transport", None) or FakeTransport()
    return (
        Deployment(
            settings=settings,
            storage=build_storage(settings),
            completion=model,
            profile=PROFILE,
            transport_for=lambda session_id: transport,
            sink=sink,
            hooks=overrides.pop("hooks", ()),
        ),
        sink,
        model,
    )


async def test_an_incident_runs_end_to_end_and_reaches_the_record(
    tmp_path: pathlib.Path,
) -> None:
    deploy, _, model = deployment(tmp_path)
    session = deploy.storage.sessions.create_session(ALERT)

    final = await run_incident(ALERT, session.session_id, deploy, fire_post_hooks=False)

    assert model.callers == ["rca", "planning", "execution", "execution"]
    assert final["execution_result"] is not None and final["execution_result"].succeeded

    stored = deploy.storage.sessions.get_session(session.session_id)
    assert stored is not None
    assert stored.root_cause is not None and stored.plan is not None
    assert stored.state == "tentative_success"
    assert stored.run_status == "completed"
    assert len(deploy.storage.runs.runs_for(session.session_id)) == 3
    deploy.storage.close()


async def test_the_configuration_is_the_first_thing_written(tmp_path: pathlib.Path) -> None:
    """A campaign is reconstructible only if what it ran under is in the record beside
    its result, and only if it was written before anything could change it."""
    deploy, sink, _ = deployment(tmp_path)
    session = deploy.storage.sessions.create_session(ALERT)

    await run_incident(ALERT, session.session_id, deploy, fire_post_hooks=False)

    first = sink.events[0]
    assert first.payload.stage == "pipeline" and first.payload.event == "entered"
    snapshot = json.loads(first.payload.detail)
    assert "settings" in snapshot
    # Which schemas produced the result, so a later reader does not infer them from the
    # date the run happens to carry.
    assert snapshot["schema_version"] == SCHEMA_VERSION
    deploy.storage.close()


async def test_a_deferred_finalisation_leaves_the_run_going(tmp_path: pathlib.Path) -> None:
    """So a poller keeps waiting through the work that follows the pipeline, instead of
    reading "finished" before the answer has been handed over."""
    deploy, _, _ = deployment(tmp_path)
    session = deploy.storage.sessions.create_session(ALERT)

    await run_incident(
        ALERT, session.session_id, deploy, fire_post_hooks=False, finalize_run_status=False
    )

    stored = deploy.storage.sessions.get_session(session.session_id)
    assert stored is not None
    assert stored.run_status == "running"
    assert stored.state == "tentative_success"
    deploy.storage.close()


async def test_a_diagnosis_only_task_never_touches_the_cluster(
    tmp_path: pathlib.Path,
) -> None:
    """Running the remediating roles would mutate the environment before the read-only
    answer had been graded, which invalidates the comparison the task exists to make."""
    transport = FakeTransport()
    deploy, _, model = deployment(tmp_path, transport=transport)
    session = deploy.storage.sessions.create_session(ALERT)

    final = await run_incident(
        ALERT, session.session_id, deploy, target_stage="diagnose", fire_post_hooks=False
    )

    assert model.callers == ["rca"]
    assert final["plan"] is None and final["execution_result"] is None
    assert not any("patch" in " ".join(argv) for argv in transport.commands)
    deploy.storage.close()


async def test_an_unknown_target_stage_is_refused(tmp_path: pathlib.Path) -> None:
    deploy, _, _ = deployment(tmp_path)

    with pytest.raises(ValueError, match="unsupported target_stage"):
        await run_incident(ALERT, "s1", deploy, target_stage="mitigate")  # type: ignore[arg-type]
    deploy.storage.close()


async def test_both_engines_run_the_same_incident(tmp_path: pathlib.Path) -> None:
    """The framework-free executor is a supported configuration, not a fixture."""
    deploy, _, model = deployment(tmp_path, pipeline={"engine": "sequential"})
    session = deploy.storage.sessions.create_session(ALERT)

    final = await run_incident(ALERT, session.session_id, deploy, fire_post_hooks=False)

    assert model.callers == ["rca", "planning", "execution", "execution"]
    assert final["execution_result"] is not None
    deploy.storage.close()


async def test_the_incident_span_records_how_the_incident_ended(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The span is opened by the runner and handed down so it can be annotated — each
    role annotates its own with what it submitted — and the incident's carried only how
    it started.

    A trace therefore showed every incident alike, and the one question a trace is opened
    to answer had to be taken from the database instead.
    """
    recorded: dict[str, object] = {}

    class _Span:
        def set_attribute(self, key: str, value: object) -> None:
            recorded[key] = value

    @contextlib.contextmanager
    def fake_span(name: str, attributes: dict[str, object] | None = None):
        yield _Span() if name == "kira.incident" else _Noop()

    class _Noop:
        def set_attribute(self, key: str, value: object) -> None:
            return None

    monkeypatch.setattr("kira.orchestration.runner.span", fake_span)
    deploy, _, _ = deployment(tmp_path)
    session = deploy.storage.sessions.create_session(ALERT)

    await run_incident(ALERT, session.session_id, deploy, fire_post_hooks=False)

    assert recorded["kira.outcome"] == "success"
    assert recorded["kira.lifecycle"] == "tentative_success"
    deploy.storage.close()


async def test_a_run_row_records_what_the_model_was_not_offered(
    tmp_path: pathlib.Path,
) -> None:
    """The toolset resolves against the transport's capabilities as well as the arm, so
    the same role in the same arm offers different tools on different hosts.

    Which tools a run actually held is therefore not reconstructible from the arm, the
    code, or the transcript — and whether a memory arm held its retrieval tools is the
    question its result rests on. The resolved toolset recorded every drop with its
    reason, and the field was read by nothing.
    """
    deploy, _, _ = deployment(tmp_path)
    session = deploy.storage.sessions.create_session(ALERT)

    await run_incident(ALERT, session.session_id, deploy, fire_post_hooks=False)

    rca, planning, *_ = deploy.storage.runs.runs_for(session.session_id)

    withheld = rca.metadata["tools_withheld"]
    assert withheld["memory_search"] == "no experience store in this configuration"
    assert "submit_root_cause" not in withheld
    # A role's own recorded facts reach the row too; they were dropped entirely before.
    assert rca.metadata["reassess_mode"]
    # And the pipeline edge written when the row opened survives the merge rather than
    # being replaced by what the role recorded when it closed.
    assert planning.metadata["after"] == rca.agent_run_id
    assert "tools_withheld" in planning.metadata
    deploy.storage.close()


async def test_a_hook_that_raises_is_recorded_and_the_incident_carries_on(
    tmp_path: pathlib.Path,
) -> None:
    """A shaping hook that fails is skipped, and skipped silently it is a concern that
    was configured, was reached, and did not run — reported to a Python logger no run
    reconstruction will ever see, on a claim that says every harness intervention
    reaches the incident's own log."""

    class Broken:
        name = "broken"

        def before_model(self, ctx: Any, request: Any) -> Any:
            raise RuntimeError("boom")

    deploy, sink, _ = deployment(tmp_path, hooks=[Broken()])
    session = deploy.storage.sessions.create_session(ALERT)

    final = await run_incident(ALERT, session.session_id, deploy, fire_post_hooks=False)

    assert final["execution_result"] is not None
    failures = [
        e.payload
        for e in sink.events
        if isinstance(e.payload, InterventionEvent) and e.payload.control == "hook_failure"
    ]
    assert failures and failures[0].action == "bypassed"
    assert failures[0].detail == "broken raised in before_model: RuntimeError: boom"
    deploy.storage.close()


async def test_a_deployment_that_widens_writes_reaches_the_tool_and_not_only_the_gate(
    tmp_path: pathlib.Path,
) -> None:
    """The tool's own stage check must screen against the settings the gate was built on.

    It read the environment instead, cached for the process, so a deployment assembled
    in code — an evaluation harness, a test, anything that builds Settings rather than
    exporting variables — had the gate allow the write and the tool refuse it, with
    nothing anywhere to say why.
    """
    settings = Settings(
        server=ServerSettings(workdir=tmp_path),
        memory=MemorySettings(enabled=False, learning_enabled=False),
        verify=VerifySettings(enabled=False),
        authority=AuthoritySettings(additional_write_stages=frozenset({"verify"})),
    )
    deploy = Deployment(
        settings=settings,
        storage=build_storage(settings),
        completion=FakeModel(),
        profile=PROFILE,
        transport_for=lambda session_id: FakeTransport(),
    )
    async with incident_context("s1", ALERT, deploy) as resources:
        ctx = tool_context_for(
            deploy, resources, "verify", "s1", Journal(session_id="s1", stage="verify")
        )
        answer = await exec_kubectl_cmd_safely(ctx, cmd="kubectl delete deploy web -n shop")

    assert not answer.startswith("error:")


def test_the_auxiliary_sampling_set_is_pinned_at_startup_too(tmp_path: pathlib.Path) -> None:
    """Both sets are sent, so both are asserted where the deployment is composed.

    The gate, the nominator and the selector run on the auxiliary set and nothing
    checked it. A set that cannot be sent failed at the first helper call instead, in
    the middle of an incident, on a configuration nobody could reconstruct afterwards.
    """
    settings = Settings(
        server=ServerSettings(workdir=tmp_path),
        memory=MemorySettings(enabled=False, learning_enabled=False),
    )
    profile = PROFILE.model_copy(
        update={
            "aux_sampling": PROFILE.sampling.model_copy(
                update={"thinking_token_budget": 4096, "max_tokens": 512}
            )
        }
    )

    with pytest.raises(ValueError, match="must be below max_tokens"):
        Deployment(
            settings=settings,
            storage=build_storage(settings),
            completion=FakeModel(),
            profile=profile,
            transport_for=lambda session_id: FakeTransport(),
        )
