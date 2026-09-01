"""Distilling a closed incident from what was written down."""

import json
import pathlib
from typing import Any

from kira.audit.journal import ListSink
from kira.llm.types import ModelRequest, ModelResponse, ToolCall, Usage
from kira.memory.factory import build_storage
from kira.memory.ports import AgentRunRecord
from kira.orchestration.bootstrap import register_post_incident_hooks
from kira.orchestration.hooks import clear_hooks, registered_hooks
from kira.orchestration.learning import build_packet, run_learning
from kira.orchestration.outcomes import Outcome
from kira.orchestration.runner import Deployment
from kira.settings.groups import MemorySettings, ServerSettings
from kira.settings.profiles import load_profile
from kira.settings.root import Settings

PROFILE = load_profile("qwen3.6-35b-a3b", pathlib.Path("configs/models"))
ALERT = {"alertname": "HighErrorRate", "namespace": "shop"}

RESULT: dict[str, Any] = {
    "observations": ["svc/user had no endpoints"],
    "attempted_hypothesis": "the service targeted a port nothing listens on",
    "attempted_action": "patched targetPort to 8080",
    "observed_outcome": "endpoints appeared and the 5xx rate fell",
    "case_negative_constraints": ["restarting the deployment does not help"],
    "corrected_cause": "",
    "discriminator": "check svc endpoints before restarting anything",
    "summary": "a Service targeting a port no container listens on",
    "fault_category": "misconfiguration",
    "verification_verdict": "recovered",
}


class FakeTransport:
    name = "fake"
    capabilities = frozenset({"kubernetes", "metrics", "logs", "traces"})
    timeout_seconds = 30

    async def kubectl(self, argv: list[str]) -> str:
        return "no resources found"


class FakeModel:
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, request: ModelRequest, *, caller: str) -> ModelResponse:
        self.calls += 1
        return ModelResponse(
            text=None,
            tool_calls=(
                ToolCall(
                    id="c1",
                    name="submit_learning_result",
                    arguments=RESULT,
                    raw_arguments=json.dumps(RESULT),
                ),
            ),
            finish_reason="stop",
            reasoning=None,
            usage=Usage(),
            duration_ms=1,
            attempts=1,
        )


def deployment(tmp_path: pathlib.Path, **memory: Any) -> tuple[Deployment, FakeModel]:
    settings = Settings(
        server=ServerSettings(workdir=tmp_path),
        memory=MemorySettings(**{"enabled": True, "learning_enabled": True, **memory}),
    )
    model = FakeModel()
    return (
        Deployment(
            settings=settings,
            storage=build_storage(settings),
            completion=model,
            profile=PROFILE,
            transport_for=lambda session_id: FakeTransport(),
            sink=ListSink(),
        ),
        model,
    )


def closed_incident(deploy: Deployment) -> str:
    session = deploy.storage.sessions.create_session(ALERT)
    deploy.storage.sessions.update_artefacts(
        session.session_id,
        root_cause={"affected_resource": "shop/Service/user", "fault_type": "misconfiguration"},
        execution_result={"succeeded": True},
    )
    deploy.storage.runs.start_run(
        AgentRunRecord(
            agent_run_id="r1",
            pipeline_id=session.session_id,
            agent_name="rca",
            attempt=1,
            status="succeeded",
            started_at="2026-09-01T00:00:00Z",
            trace_summary={
                "role": "rca",
                "goal": "diagnose",
                "key_observations": ["endpoints empty for svc/user"],
            },
        )
    )
    return session.session_id


async def test_a_closed_incident_becomes_a_record(tmp_path: pathlib.Path) -> None:
    deploy, model = deployment(tmp_path)
    session_id = closed_incident(deploy)

    assert await run_learning(session_id, deploy) == "distilled"

    assert model.calls == 1
    assert deploy.storage.index.open_record(session_id) is not None
    stored = deploy.storage.sessions.get_session(session_id)
    assert stored is not None and stored.distilled is True
    deploy.storage.close()


async def test_a_session_already_distilled_costs_no_model_call(
    tmp_path: pathlib.Path,
) -> None:
    """Re-distilling would replace rather than duplicate, so this is a cost guard: a
    retried job should not spend a call re-deriving a lesson that is already stored."""
    deploy, model = deployment(tmp_path)
    session_id = closed_incident(deploy)
    await run_learning(session_id, deploy)

    assert await run_learning(session_id, deploy) == "nothing_to_distil"
    assert model.calls == 1
    deploy.storage.close()


def test_the_packet_is_assembled_from_the_record_not_from_live_state(
    tmp_path: pathlib.Path,
) -> None:
    """The job that starts this has nothing but a session identifier, which is the right
    constraint: a lesson that could only be distilled from live state would be lost to
    any restart."""
    deploy, _ = deployment(tmp_path)
    session_id = closed_incident(deploy)

    packet = build_packet(deploy, session_id)

    assert packet is not None
    assert packet.alert == ALERT
    assert packet.root_cause is not None
    assert [s.role for s in packet.trace_summaries] == ["rca"]
    deploy.storage.close()


async def test_nothing_is_distilled_for_a_session_that_is_gone(
    tmp_path: pathlib.Path,
) -> None:
    deploy, model = deployment(tmp_path)

    assert await run_learning("no-such-session", deploy) == "nothing_to_distil"
    assert model.calls == 0
    deploy.storage.close()


async def test_the_loop_closes_on_every_terminal_outcome(tmp_path: pathlib.Path) -> None:
    """A run that stopped deliberately, with a stated reason, is one of the more
    informative things a corpus can hold: it records a judgement that the system should
    not act, which a later incident facing the same evidence needs to know was already
    considered."""
    clear_hooks()
    deploy, _ = deployment(tmp_path)
    try:
        register_post_incident_hooks(deploy.settings)
        # Registering twice is what a restarted process does, and it must not raise.
        register_post_incident_hooks(deploy.settings)

        hook = registered_hooks()["learning"]
        assert hook.on_outcomes == frozenset(Outcome)
        assert hook.delay_seconds == deploy.settings.memory.learning_delay_s
    finally:
        clear_hooks()
        deploy.storage.close()


async def test_no_hook_is_registered_when_nothing_will_be_distilled(
    tmp_path: pathlib.Path,
) -> None:
    clear_hooks()
    deploy, _ = deployment(tmp_path, learning_enabled=False)
    try:
        register_post_incident_hooks(deploy.settings)
        assert registered_hooks() == {}
    finally:
        clear_hooks()
        deploy.storage.close()
