"""One incident, end to end, with a fake model and a fake cluster.

The first test that exercises the composition root: real roles, real routing, a real
SQLite store, and nothing global. What it proves is that the pieces compose — that a
role's context can be built from a deployment's resources and that what the pipeline
concluded reaches the record a later incident reads.
"""

import json
import pathlib
from typing import Any

import pytest

from kira.audit.journal import ListSink
from kira.llm.types import ModelRequest, ModelResponse, ToolCall, Usage
from kira.memory.factory import build_storage
from kira.orchestration.runner import Deployment, run_incident
from kira.settings.groups import (
    MemorySettings,
    PipelineSettings,
    ServerSettings,
    VerifySettings,
)
from kira.settings.profiles import load_profile
from kira.settings.root import Settings

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
    capabilities = frozenset({"kubernetes", "kubernetes.write", "metrics", "logs", "traces"})
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
        ),
        sink,
        model,
    )


async def test_an_incident_runs_end_to_end_and_reaches_the_record(
    tmp_path: pathlib.Path,
) -> None:
    deploy, _, model = deployment(tmp_path)
    session = deploy.storage.sessions.create_session(ALERT)
    seen: list[str] = []

    async def on_stage(stage: str, state: Any) -> None:
        seen.append(stage)

    final = await run_incident(
        ALERT, session.session_id, deploy, on_stage=on_stage, fire_post_hooks=False
    )

    assert model.callers == ["rca", "planning", "execution", "execution"]
    assert seen == ["rca", "planning", "execution"]
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
    assert "settings" in json.loads(first.payload.detail)
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
