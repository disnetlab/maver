"""Distilling a closed incident from what was written down."""

import asyncio
import json
import pathlib
from dataclasses import replace
from typing import Any

import pytest

from maver.agents import AgentConfig, ComponentConfig, LearningAgent
from maver.agents import builtin_registry as agent_registry
from maver.audit.journal import ListSink
from maver.contracts.runtime import AgentState, RunState
from maver.llm.types import ModelRequest, ModelResponse, ToolCall, Usage
from maver.memory.factory import build_storage
from maver.memory.ports import AgentRunRecord
from maver.middleware.base import TurnContext
from maver.orchestration.bootstrap import register_post_incident_hooks
from maver.orchestration.hooks import clear_hooks, registered_hooks
from maver.orchestration.learning import build_packet, run_learning
from maver.orchestration.outcomes import Outcome
from maver.orchestration.runner import Deployment
from maver.pipelines import PipelineCatalog, PipelineChangedError, select_pipeline
from maver.pipelines import builtin_registry as pipeline_registry
from maver.roles.base import RoleContext, RoleOutcome
from maver.settings.groups import MemorySettings, PromptSettings, ServerSettings
from maver.settings.profiles import load_profile
from maver.settings.root import Settings

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
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest, *, caller: str) -> ModelResponse:
        self.calls += 1
        self.requests.append(request)
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


def closed_incident(deploy: Deployment, alert: dict[str, Any] = ALERT) -> str:
    session = deploy.storage.sessions.create_session(alert)
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
    [learning] = [
        run for run in deploy.storage.runs.runs_for(session_id) if run.agent_name == "learning"
    ]
    assert learning.status == "succeeded"
    assert learning.output == RESULT
    assert learning.metadata["agent_instance"] == learning.agent_run_id
    assert learning.metadata["transcript"]["inline"]
    assert learning.metadata["distilled"] is True
    learning_events = [event for event in deploy.sink.events if event.stage == "learning"]
    assert all(event.run_id == learning.agent_run_id for event in learning_events)
    assert learning_events[-1].payload.source == "learning"
    manifest = learning_events[-1].payload.run["inline"]
    assert manifest["run_id"] == learning.agent_run_id
    assert manifest["metadata"]["transcript"] == learning.metadata["transcript"]
    assert manifest["output"] == learning.output
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


async def test_the_ground_truth_label_reaches_no_prompt_of_this_pass_either(
    tmp_path: pathlib.Path,
) -> None:
    """The underscore prefix is what makes the blinding structural rather than a
    convention each role has to remember, and this pass renders the packet whole — so it
    was the one role shown the answer key, in every feedback mode. Its output is the
    corpus record a later incident retrieves, which is the worst place to leak it.
    """
    deploy, model = deployment(tmp_path)
    session_id = closed_incident(deploy, {**ALERT, "_problem_id": "leak-the-answer"})

    await run_learning(session_id, deploy)

    request = model.requests[0]
    rendered = request.system + json.dumps(request.messages)
    assert "leak-the-answer" not in rendered
    assert "_problem_id" not in rendered
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


async def test_the_distilling_call_is_counted_like_every_other(
    tmp_path: pathlib.Path,
) -> None:
    """The rule is that a model call anywhere is counted, and this stage was the
    exception: it was handed the raw client rather than the recording wrapper, so it
    spent tokens that appear in no event, in the arm whose claim is that its memory is
    worth what it costs to build.
    """
    deploy, model = deployment(tmp_path)
    session_id = closed_incident(deploy)

    assert await run_learning(session_id, deploy) == "distilled"

    calls = [e for e in deploy.sink.events if e.payload.kind == "model_call"]
    assert len(calls) == model.calls == 1
    assert calls[0].payload.caller == "learning"
    assert calls[0].payload.sampling["digest"] == deploy.sampling.digest
    deploy.storage.close()


def _with_learning_agent(
    deploy: Deployment,
    implementation: type[LearningAgent],
    *,
    config: dict[str, Any] | None = None,
    middleware: Any = None,
    profiles: dict[str, Any] | None = None,
    completion_for: Any = None,
) -> Deployment:
    registry = agent_registry()
    registry.register("custom_learning", implementation)
    if middleware is not None:
        registry.register_middleware("domain_context", middleware)
    agents = registry.resolve_many(
        {
            **{
                name: {"implementation": name}
                for name in ("rca", "planning", "execution", "verify", "learning")
            },
            "specialist": {"implementation": "custom_learning", "config": config or {}},
        },
        profiles=profiles,
        override_dir=deploy.settings.prompts.override_dir,
    )
    pipeline = pipeline_registry().resolve(
        "multi", {"implementation": "multi", "learning_agent": "specialist"}, agents=agents
    )
    return replace(
        deploy,
        pipelines=PipelineCatalog({"multi": pipeline}, "multi"),
        completion_for=completion_for,
    )


async def _pin_learning_session(deploy: Deployment, session_id: str) -> None:
    assert deploy.pipelines is not None
    selected = await select_pipeline(deploy.pipelines, alert=ALERT)
    runtime = RunState(
        selection=selected.manifest(),
        deployment_digest=deploy.runtime_digest,
        metadata={"private_note": "PRIVATE_LEARNING_METADATA"},
        steps=4,
        active_agent="foreground",
        agents={
            "foreground": AgentState(
                id="foreground",
                definition="execution",
                stage="execution",
                role="execution",
                messages=[{"role": "user", "content": "PRIVATE_FOREGROUND_CONVERSATION"}],
            )
        },
    )
    deploy.storage.sessions.set_runtime(session_id, runtime.model_dump(mode="json"))


async def test_pinned_learning_agent_uses_its_subclass_model_tools_prompt_and_middleware(
    tmp_path: pathlib.Path,
) -> None:
    deploy, foreground_model = deployment(tmp_path)
    prompts = tmp_path / "prompts"
    (prompts / "roles").mkdir(parents=True)
    (prompts / "roles" / "domain_learning.txt").write_text(
        "DOMAIN_LEARNING_SYSTEM", encoding="utf-8"
    )
    deploy = replace(
        deploy,
        settings=deploy.settings.model_copy(
            update={"prompts": PromptSettings(override_dir=prompts)}
        ),
    )
    learning_model = FakeModel()
    profile = PROFILE.model_copy(update={"name": "learning_model", "model": "local/learning"})
    selected_profiles: list[str] = []

    class DomainConfig(AgentConfig):
        domain: str = "checkout"

    class DomainLearning(LearningAgent):
        config_type = DomainConfig

        def prepare_context(self, context: RoleContext) -> RoleContext:
            assert context.agent_state is not None
            context.agent_state.metadata["domain"] = self.config.domain
            return super().prepare_context(context)

    class DomainContext:
        name = "domain_context"

        def before_model(self, ctx: TurnContext, request: ModelRequest) -> None:
            assert ctx.run_state is not None and ctx.agent_state is not None
            ctx.agent_state.middleware["turn_seen"] = ctx.turn
            ctx.run_state.middleware["learner_seen"] = True
            ctx.run_state.add_context(
                "DOMAIN_LEARNING_CONTEXT", source=self.name, recipient=ctx.agent_state.id
            )

    def factory(config: ComponentConfig, context: RoleContext) -> DomainContext:
        return DomainContext()

    def model_for(selected: Any, settings: Settings) -> FakeModel:
        selected_profiles.append(selected.name)
        return learning_model

    deploy = _with_learning_agent(
        deploy,
        DomainLearning,
        config={
            "model_profile": "learning_model",
            "prompt": "domain_learning",
            "tools": ["submit_learning_result"],
            "middleware": [{"name": "domain_context"}],
            "domain": "payments",
        },
        middleware=factory,
        profiles={"learning_model": profile},
        completion_for=model_for,
    )
    session_id = closed_incident(deploy)
    await _pin_learning_session(deploy, session_id)
    packet = build_packet(deploy, session_id)
    assert packet is not None and "PRIVATE_LEARNING_METADATA" not in packet.model_dump_json()

    assert await run_learning(session_id, deploy) == "distilled"

    assert foreground_model.calls == 0 and learning_model.calls == 1
    assert selected_profiles == ["learning_model"]
    request = learning_model.requests[0]
    rendered = request.system + json.dumps(request.messages)
    assert "DOMAIN_LEARNING_SYSTEM" in request.system
    assert "DOMAIN_LEARNING_CONTEXT" in rendered
    assert "PRIVATE_LEARNING_METADATA" not in rendered
    assert "PRIVATE_FOREGROUND_CONVERSATION" not in rendered
    assert [tool.name for tool in request.tools] == ["submit_learning_result"]
    stored = deploy.storage.sessions.get_session(session_id)
    assert stored is not None
    runtime = RunState.model_validate(stored.runtime)
    learner = next(agent for agent in runtime.agents.values() if agent.role == "learning")
    assert learner.definition == "specialist"
    assert learner.metadata == {"domain": "payments"}
    assert learner.middleware == {"turn_seen": 1}
    assert learner.messages
    assert runtime.middleware == {"learner_seen": True}
    assert runtime.steps == 4 and runtime.active_agent == "foreground"
    assert runtime.agents["foreground"].messages[0]["content"] == "PRIVATE_FOREGROUND_CONVERSATION"
    calls = [event for event in deploy.sink.events if event.payload.kind == "model_call"]
    assert calls[-1].payload.model == "local/learning"
    deploy.storage.close()


async def test_learning_retries_get_separate_durable_invocation_state(
    tmp_path: pathlib.Path,
) -> None:
    deploy, model = deployment(tmp_path)

    class NoSubmission(LearningAgent):
        async def run(self, context: RoleContext) -> RoleOutcome:
            assert context.run_state is not None and context.agent_state is not None
            count = int(context.run_state.middleware.get("learning_attempts", 0)) + 1
            context.run_state.middleware["learning_attempts"] = count
            context.agent_state.metadata["attempt"] = count
            return RoleOutcome(
                None, [{"role": "user", "content": f"attempt {count}"}], error="no lesson"
            )

    deploy = _with_learning_agent(deploy, NoSubmission)
    session_id = closed_incident(deploy)
    await _pin_learning_session(deploy, session_id)
    assert await run_learning(session_id, deploy) == "no_submission"
    assert await run_learning(session_id, deploy) == "no_submission"
    stored = deploy.storage.sessions.get_session(session_id)
    assert stored is not None
    runtime = RunState.model_validate(stored.runtime)
    learners = [agent for agent in runtime.agents.values() if agent.role == "learning"]
    assert len(learners) == 2
    assert [agent.metadata["attempt"] for agent in learners] == [1, 2]
    assert learners[0].id != learners[1].id
    assert learners[0].messages != learners[1].messages
    assert runtime.middleware["learning_attempts"] == 2
    assert runtime.steps == 4 and runtime.active_agent == "foreground"
    assert model.calls == 0
    deploy.storage.close()


async def test_learning_refuses_changed_pinned_agent_before_a_model_call(
    tmp_path: pathlib.Path,
) -> None:
    deploy, model = deployment(tmp_path)
    deploy = _with_learning_agent(deploy, LearningAgent, config={"limits": {"max_turns": 2}})
    session_id = closed_incident(deploy)
    await _pin_learning_session(deploy, session_id)
    changed = _with_learning_agent(deploy, LearningAgent, config={"limits": {"max_turns": 3}})
    with pytest.raises(PipelineChangedError, match="changed since admission"):
        await run_learning(session_id, changed)
    assert model.calls == 0
    deploy.storage.close()


async def test_pipeline_can_explicitly_disable_its_learning_agent(
    tmp_path: pathlib.Path,
) -> None:
    deploy, model = deployment(tmp_path)
    agents = agent_registry().resolve_many(
        {
            name: {"implementation": name}
            for name in ("rca", "planning", "execution", "verify", "learning")
        }
    )
    registry = pipeline_registry()
    inherited = registry.resolve("multi", {"implementation": "multi"}, agents=agents)
    assert inherited.learning_agent is not None
    disabled = registry.resolve(
        "multi", {"implementation": "multi", "learning_agent": None}, agents=agents
    )
    assert disabled.learning_agent is None and disabled.digest != inherited.digest
    assert (
        registry.resolve("diagnose", {"implementation": "diagnose"}, agents=agents).learning_agent
        is None
    )
    deploy = replace(deploy, pipelines=PipelineCatalog({"multi": disabled}, "multi"))
    session_id = closed_incident(deploy)
    await _pin_learning_session(deploy, session_id)
    assert await run_learning(session_id, deploy) == "disabled"
    assert model.calls == 0
    deploy.storage.close()


def test_learning_binding_requires_the_learning_contract(tmp_path: pathlib.Path) -> None:
    deploy, _ = deployment(tmp_path)
    agents = agent_registry().resolve_many({"rca": {"implementation": "rca"}})
    registry = pipeline_registry()
    with pytest.raises(ValueError, match="unknown learning agent"):
        registry.resolve(
            "diagnose", {"implementation": "diagnose", "learning_agent": "missing"}, agents=agents
        )
    with pytest.raises(ValueError, match="must have the learning role"):
        registry.resolve(
            "diagnose", {"implementation": "diagnose", "learning_agent": "rca"}, agents=agents
        )
    deploy.storage.close()


async def test_learning_checks_deployment_defaults_with_a_manually_supplied_catalog(
    tmp_path: pathlib.Path,
) -> None:
    deploy, model = deployment(tmp_path)
    deploy = _with_learning_agent(deploy, LearningAgent)
    session_id = closed_incident(deploy)
    await _pin_learning_session(deploy, session_id)
    changed = replace(
        deploy,
        settings=deploy.settings.model_copy(
            update={"runtime": deploy.settings.runtime.model_copy(update={"no_tool_nudges": 17})}
        ),
    )
    with pytest.raises(PipelineChangedError, match="deployment behavior changed"):
        await run_learning(session_id, changed)
    assert model.calls == 0
    deploy.storage.close()


async def test_cancelled_learning_keeps_its_run_and_partial_conversation(
    tmp_path: pathlib.Path,
) -> None:
    """An interrupted async invocation must remain joinable to its model-call audit."""
    deploy, _ = deployment(tmp_path)
    session_id = closed_incident(deploy)
    entered = asyncio.Event()

    class Slow:
        async def complete(self, request, *, caller):
            entered.set()
            await asyncio.sleep(30)

    deploy = replace(deploy, completion=Slow())
    task = asyncio.create_task(run_learning(session_id, deploy))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    [learning] = [
        run for run in deploy.storage.runs.runs_for(session_id) if run.agent_name == "learning"
    ]
    assert learning.status == "cancelled"
    assert learning.metadata["transcript"]["inline"]
    stored = deploy.storage.sessions.get_session(session_id)
    assert stored is not None and not stored.distilled
    assert deploy.sink.events[-1].payload.event == "cancelled"
    assert deploy.sink.events[-1].run_id == learning.agent_run_id
    deploy.storage.close()
