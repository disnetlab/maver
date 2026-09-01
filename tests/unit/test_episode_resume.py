"""Continuing an episode that stopped for a person, under the composition it ran under.

Two mechanisms meet here and were nearly independent of each other. The pause is a
database write, so a decision that arrives twenty minutes later continues the same
incident; an adapter instance is one episode's live binding, so the transport its tools
speak and the correlation id its harness keys on exist only for as long as that episode
does. A continuation that honours the first and loses the second replays an operator's
approved write over the process's default transport, outside the session being graded,
and the adapter's own tools are not there to replay at all.

So: the approved call goes over the episode's transport, the episode still reaches an
ending, and a continuation that can have neither says so rather than pretending.
"""

import importlib
import json
import pathlib
import sys
from collections.abc import Iterator
from typing import Any

import pytest

from kira.adapters.base import AdapterTool, EpisodeContext, ExternalFeedback, SubmissionRecord
from kira.adapters.registry import adapter, clear_adapters
from kira.audit.journal import ListSink
from kira.llm.types import ModelRequest, ModelResponse, ToolCall, Usage
from kira.memory.factory import build_storage
from kira.middleware.human_approval import HumanApproval
from kira.orchestration.episode import EpisodeStart, open_episode, run_episode
from kira.orchestration.inflight import drop_episode
from kira.orchestration.jobs import bind_deployment, resume_approval, unbind_deployment
from kira.orchestration.runner import Deployment
from kira.settings.authority import AuthoritySettings
from kira.settings.groups import MemorySettings, PipelineSettings, ServerSettings, VerifySettings
from kira.settings.profiles import load_profile
from kira.settings.root import Settings
from kira.tools.context import ToolContext
from kira.tools.policy import always_writes
from kira.tools.registry import ToolSpec

PROFILE = load_profile("qwen3.6-35b-a3b", pathlib.Path("configs/models"))

ADAPTER_NAME = "episode-stub"

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
    "steps": [{"step_index": 0, "succeeded": True, "observation": "rolled the write back"}],
    "final_state_summary": "svc/user has endpoints",
    "confidence": 0.8,
}


# --- an environment that supplies its own transport and its own tool ------------------


class EpisodeTransport:
    """The transport the adapter builds for one episode, and the only one that rolls back."""

    name = "episode"
    capabilities = frozenset(
        {"kubernetes", "kubernetes.write", "kubernetes.rollback", "metrics", "logs", "traces"}
    )
    timeout_seconds = 30

    def __init__(self) -> None:
        self.rollbacks = 0

    async def kubectl(self, argv: list[str]) -> str:
        return "ok"

    async def rollback_last(self) -> str:
        self.rollbacks += 1
        return "reverted: kubectl patch svc user"


class BaseTransport:
    """The deployment's own transport, which knows nothing about any harness session."""

    name = "base"
    capabilities = frozenset({"kubernetes", "kubernetes.write", "metrics", "logs", "traces"})
    timeout_seconds = 30

    def __init__(self) -> None:
        self.served: list[list[str]] = []

    async def kubectl(self, argv: list[str]) -> str:
        self.served.append(list(argv))
        return "ok"


async def _rollback(ctx: ToolContext) -> str:
    return await ctx.transport.rollback_last()


ROLLBACK = ToolSpec(
    name="episode_rollback",
    parameters={"type": "object", "properties": {}},
    safety="cluster_write",
    dispatch=_rollback,
    authority=lambda arguments: "write",
    requires=frozenset({"kubernetes.rollback"}),
    descriptions={"episode_rollback": {"default": "Undo the last write. Takes no arguments."}},
)

always_writes(
    "episode_rollback",
    stages=frozenset({"execution"}),
    refusal=lambda stage: f"error: rollback is not permitted for the {stage} agent",
)


class EpisodeAdapter:
    """One episode against a harness that records rather than grades."""

    name = ADAPTER_NAME

    def __init__(self) -> None:
        self.transport = EpisodeTransport()
        self.session_id = ""
        self.submitted: list[tuple[str, Any]] = []
        self.releases = 0

    async def acquire(self) -> EpisodeContext:
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
        return True

    def encode(self, kind: str, state: Any, ctx: EpisodeContext) -> Any:
        if kind != "diagnosis":
            return ""
        return getattr(state.get("root_cause"), "summary", "") or "nothing was concluded"

    async def submit(self, kind: str, payload: Any) -> SubmissionRecord:
        self.submitted.append((kind, payload))
        return SubmissionRecord(kind=kind, payload=payload)

    def holds_environment_for_grading(self) -> bool:
        return True

    async def fetch_feedback(self) -> ExternalFeedback | None:
        return None

    async def release(self, *, state: Any) -> None:
        self.releases += 1

    def tool_transport(self) -> Any:
        return self.transport

    def evidence_policy(self) -> Any:
        return None

    def extra_tools(self) -> tuple[AdapterTool, ...]:
        return (AdapterTool(spec=ROLLBACK, roles=frozenset({"execution"})),)

    def routes(self) -> Any:
        return None


class ParkingModel:
    """Diagnoses, plans, then calls the adapter's own write tool — which asks a human."""

    def __init__(self) -> None:
        self.callers: list[str] = []

    async def complete(self, request: ModelRequest, *, caller: str) -> ModelResponse:
        self.callers.append(caller)
        if caller == "rca":
            name, arguments = "submit_root_cause", ROOT_CAUSE
        elif caller == "planning":
            name, arguments = "submit_plan", PLAN
        elif self.callers.count("execution") == 1:
            name, arguments = "episode_rollback", {}
        else:
            name, arguments = "submit_execution_result", EXECUTION
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


def deployment(tmp_path: pathlib.Path, base: BaseTransport) -> Deployment:
    settings = Settings(
        server=ServerSettings(workdir=tmp_path),
        pipeline=PipelineSettings(),
        memory=MemorySettings(enabled=False, learning_enabled=False),
        verify=VerifySettings(enabled=False),
        authority=AuthoritySettings(mode="prompt", timeout_s=1800),
    )
    storage = build_storage(settings)
    return Deployment(
        settings=settings,
        storage=storage,
        completion=ParkingModel(),
        profile=PROFILE,
        transport_for=lambda session_id: base,
        authority=HumanApproval(
            store=storage.approvals,
            settings=settings.authority,
            arm_timeout=lambda request_id, expires_at: None,
        ),
        sink=ListSink(),
    )


async def park(
    tmp_path: pathlib.Path, base: BaseTransport
) -> tuple[Deployment, EpisodeAdapter, EpisodeStart, str]:
    """Run one episode up to the write that needs an answer."""
    deploy = deployment(tmp_path, base)
    stub = EpisodeAdapter()
    start = await open_episode(stub, deploy)
    result = await run_episode(stub, deploy, start)

    assert result.paused is True
    stored = deploy.storage.sessions.get_session(start.session_id)
    assert stored is not None and stored.awaiting_request_id
    return deploy, stub, start, stored.awaiting_request_id


async def resumed(deploy: Deployment, request_id: str) -> str:
    """Answer through the path a decided request actually takes."""
    deploy.storage.approvals.decide(
        request_id, status="approved", decided_by="an operator", reason="ok"
    )
    bind_deployment(deploy)
    try:
        return await resume_approval(request_id)
    finally:
        unbind_deployment()


async def test_an_approved_adapter_tool_is_still_there_when_the_incident_resumes(
    tmp_path: pathlib.Path,
) -> None:
    """The sharpest form of losing the episode's composition: the write silently vanishes.

    A rollback exists only while the harness holds the session state its journal of
    inverse commands is keyed on, so the adapter supplies the tool and the transport
    together. Resumed against the process's default deployment, the tool is not in the
    toolset at all — the replay comes back as "not available in this configuration", the
    operator's approval is honoured on paper, and nothing is written anywhere.
    """
    base = BaseTransport()
    deploy, stub, _start, request_id = await park(tmp_path, base)

    await resumed(deploy, request_id)

    assert stub.transport.rollbacks == 1
    assert base.served == [], "the approved write must not go over the default transport"
    deploy.storage.close()


async def test_a_resumed_episode_is_submitted_graded_and_released(
    tmp_path: pathlib.Path,
) -> None:
    """The episode driver's tail is not optional; it is the whole point of the episode.

    Parking returns to the driver with the environment still held, deliberately, because
    the approved write is meant to act on that cluster. If the continuation then finishes
    the incident and stops there, nobody submits the answers, nobody collects the grade,
    nobody releases the environment and nobody writes the terminal run status: the
    cluster is stranded and the problem leaves the run with no result at all rather than
    a failed one.
    """
    base = BaseTransport()
    deploy, stub, start, request_id = await park(tmp_path, base)

    await resumed(deploy, request_id)

    assert [kind for kind, _ in stub.submitted] == ["diagnosis", "mitigation"]
    assert stub.submitted[0][1] == ROOT_CAUSE["summary"]
    assert stub.releases == 1
    stored = deploy.storage.sessions.get_session(start.session_id)
    assert stored is not None and stored.run_status == "completed"
    deploy.storage.close()


@pytest.fixture
def registered_adapter() -> Iterator[list[EpisodeAdapter]]:
    """Register the stub in the adapter registry, and leave the registry as it was."""
    built: list[EpisodeAdapter] = []

    @adapter(ADAPTER_NAME)
    def _factory(settings: Settings) -> Any:
        built.append(EpisodeAdapter())
        return built[-1]

    try:
        yield built
    finally:
        # A name cannot be registered twice, so a throwaway one is undone by emptying the
        # registry and re-running whatever had already registered into it.
        shipped = sys.modules.get("kira.adapters.sregym")
        clear_adapters()
        if shipped is not None:
            importlib.reload(shipped)


async def test_an_episode_this_process_no_longer_holds_is_ended_not_resumed(
    tmp_path: pathlib.Path,
    registered_adapter: list[EpisodeAdapter],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A pending approval does not survive a restart, and the honest answer is to say so.

    A fresh adapter built from the checkpoint's name has acquired nothing: no transport,
    no episode, none of the harness state its correlation id addresses. Continuing under
    the process's default composition would produce a run that looks graded and was
    served by the wrong transport, which is worse than losing the problem. So the run is
    marked failed, which is what stops a driver polling it, and the environment is handed
    back rather than stranded.
    """
    base = BaseTransport()
    deploy, stub, start, request_id = await park(tmp_path, base)
    # What a restart leaves behind: the checkpoint, and no live episode to serve it.
    drop_episode(start.session_id)

    with caplog.at_level("ERROR"):
        status = await resumed(deploy, request_id)

    assert status == "abandoned"
    assert base.served == [] and stub.transport.rollbacks == 0
    stored = deploy.storage.sessions.get_session(start.session_id)
    assert stored is not None and stored.run_status == "failed"
    assert registered_adapter and registered_adapter[0].releases == 1
    assert "does not survive a restart" in caplog.text
    deploy.storage.close()
