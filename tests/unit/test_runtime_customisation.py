"""Private shared state, scoped context, and real configured pipeline continuations."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
from test_approval_pause import ALERT, COMMANDS, ResumingModel, parked_with
from test_approval_pause import deployment as approval_deployment
from test_loop import DISPATCH, SUBMIT, TOOLS, FakeModel, call, turn
from test_runner import deployment

from maver.agents import AgentEntry, builtin_registry
from maver.audit.journal import Journal, ListSink
from maver.contracts.artifacts import RootCause
from maver.contracts.runtime import AgentState, RunState
from maver.llm.types import ModelRequest
from maver.memory.factory import build_storage
from maver.middleware.authority import ApprovalPending
from maver.middleware.base import MiddlewareStack, TurnContext
from maver.orchestration.runner import resume_after_decision, run_incident
from maver.orchestration.runtime import load_runtime
from maver.pipelines import (
    PipelineCatalog,
    PipelineChangedError,
    PipelineConfig,
    PipelineRegistry,
    StageDefinition,
    Transition,
    WorkflowDefinition,
    select_pipeline,
)
from maver.roles.loop import run_until_submit


async def test_scoped_context_has_explicit_lifetime_and_private_state_stays_private():
    runtime = RunState(metadata={"tenant_secret": "private-value"})
    agent = AgentState(id="s:first:1", definition="same", stage="first", role="rca")
    runtime.agents[agent.id] = agent
    runtime.add_context("one call only", source="test", recipient=agent.id)
    runtime.add_context("whole invocation", source="test", recipient="first", lifetime="invocation")
    runtime.add_context("future agent", source="test", recipient="second")
    sink = ListSink()
    model = FakeModel([turn(call("get_logs", {})), turn(call(SUBMIT, {}))])
    result = await run_until_submit(
        model=model,
        role="rca",
        system="system",
        tools=TOOLS,
        dispatch=DISPATCH,
        submit_tool=SUBMIT,
        messages=[{"role": "user", "content": "incident"}],
        middleware=MiddlewareStack(run_state=runtime, agent_state=agent),
        journal=Journal(sink, session_id="s", stage="rca"),
    )
    first, second = [json.dumps(request.messages) for request in model.requests]
    assert "one call only" in first and "one call only" not in second
    assert "whole invocation" in first and "whole invocation" in second
    assert all(
        "private-value" not in text and "future agent" not in text for text in (first, second)
    )
    assert agent.messages == result.transcript
    assert "one call only" not in json.dumps(agent.messages)
    assert [a.recipient for a in runtime.additions] == [agent.id, "second"]
    injected = [e.payload for e in sink.events if e.payload.kind == "intervention"]
    assert len(injected) == 2 and all(e.control == "middleware_context" for e in injected)
    assert injected[0].context[0]["content"] == "one call only"
    restored = RunState.model_validate_json(runtime.model_dump_json())
    assert restored.metadata == runtime.metadata
    assert restored.agents[agent.id].messages == result.transcript


async def test_failed_model_call_retains_pending_context():
    class Broken:
        async def complete(self, request, *, caller):
            raise RuntimeError("offline")

    state = RunState()
    agent = AgentState(id="id", definition="rca", stage="rca", role="rca")
    key = state.add_context("retry me", source="test", recipient="id")
    with pytest.raises(RuntimeError, match="offline"):
        await run_until_submit(
            model=Broken(),
            role="rca",
            system="",
            tools=TOOLS,
            dispatch=DISPATCH,
            submit_tool=SUBMIT,
            messages=[],
            middleware=MiddlewareStack(run_state=state, agent_state=agent),
        )
    assert [item.id for item in state.additions] == [key]


class Track:
    name = "track"

    def before_model(self, ctx: TurnContext, request: ModelRequest):
        assert ctx.run_state is not None and ctx.agent_state is not None
        own = ctx.agent_state.middleware
        own["turns"] = int(own.get("turns", 0)) + 1
        ctx.run_state.middleware["last_stage"] = ctx.agent_state.stage
        if own["turns"] == 1:
            ctx.run_state.add_context(
                f"only for {ctx.agent_state.stage}",
                source=self.name,
                recipient=ctx.agent_state.id,
            )
        return request


@pytest.mark.parametrize("engine", ["sequential", "langgraph"])
async def test_same_configured_agent_in_two_stages_has_separate_conversations(tmp_path, engine):
    deploy, _, model = deployment(tmp_path, pipeline={"engine": engine})
    registry = builtin_registry()
    registry.register_middleware("track", lambda config, context: Track())
    agents = registry.resolve_many(
        {"my_rca": AgentEntry(implementation="rca", config={"middleware": [{"name": "track"}]})}
    )
    workflows = PipelineRegistry()
    workflows.register(
        WorkflowDefinition(
            name="two_opinions",
            version="1",
            entry="first",
            stages=tuple(
                StageDefinition(name, "rca", RootCause, "root_cause", "my_rca")
                for name in ("first", "second")
            ),
            transitions={
                "first": Transition(("second",), lambda s, c, p: "second"),
                "second": Transition((None,), lambda s, c, p: None),
            },
        )
    )
    catalog = PipelineCatalog(
        workflows.resolve_many(
            {"review": PipelineConfig(implementation="two_opinions")}, agents=agents
        ),
        "review",
    )
    requests = []
    complete = model.complete

    async def capture(request, *, caller):
        requests.append(request)
        return await complete(request, caller=caller)

    model.complete = capture
    deploy = replace(deploy, pipelines=catalog)
    session = deploy.storage.sessions.create_session(ALERT)
    final = await run_incident(
        ALERT,
        session.session_id,
        deploy,
        fire_post_hooks=False,
        metadata={"token": "never in a prompt"},
    )
    runtime = final["runtime"]
    first, second = runtime.agents.values()
    assert first.id != second.id and first.definition == second.definition == "my_rca"
    assert first.stage == "first" and second.stage == "second"
    assert first.messages is not second.messages
    assert first.middleware == second.middleware == {"turns": 1}
    assert "only for first" in json.dumps(requests[0].messages)
    assert "only for first" not in json.dumps(requests[1].messages)
    assert "only for second" in json.dumps(requests[1].messages)
    assert "never in a prompt" not in json.dumps([r.messages for r in requests])
    saved = load_runtime(deploy.storage.sessions, session.session_id)
    assert saved == runtime and saved.steps == 2
    deploy.storage.close()


async def test_restart_resumes_same_instance_and_private_middleware_state(tmp_path: Path):
    deploy, _ = approval_deployment(tmp_path)
    model = ResumingModel()
    deploy = replace(deploy, completion=model, hooks=(Track(),))
    session = deploy.storage.sessions.create_session(ALERT)
    with pytest.raises(ApprovalPending):
        await run_incident(
            ALERT,
            session.session_id,
            deploy,
            fire_post_hooks=False,
            metadata={"secret": "do not reveal"},
        )
    before = load_runtime(deploy.storage.sessions, session.session_id)
    assert before is not None and before.active_agent
    identity = before.active_agent
    assert before.agents[identity].middleware == {"turns": 1}
    record = deploy.storage.sessions.get_session(session.session_id)
    request_id = record.awaiting_request_id
    deploy.storage.approvals.decide(request_id, status="approved", decided_by="operator")
    deploy.storage.close()
    # A fresh store, model, and hook instance must recover the durable state.
    model = ResumingModel()
    model.callers = ["execution"]
    fresh = replace(
        deploy, storage=build_storage(deploy.settings), completion=model, hooks=(Track(),)
    )
    assert await resume_after_decision(request_id, fresh) == "success"
    after = load_runtime(fresh.storage.sessions, session.session_id)
    assert after is not None
    assert len(after.agents) == 3 and after.active_agent == identity
    assert after.agents[identity].middleware == {"turns": 2}
    assert after.steps == before.steps == 3
    assert after.metadata == {"secret": "do not reveal"}
    sent = json.dumps(model.requests[0].messages)
    assert "do not reveal" not in sent and "only for execution" not in sent
    assert any(m.get("tool_call_id") == "write" for m in after.agents[identity].messages)
    fresh.storage.close()


async def test_changed_composition_is_refused_before_approved_write(tmp_path):
    deploy, session_id, request_id = await parked_with(tmp_path)
    deploy.storage.approvals.decide(request_id, status="approved", decided_by="operator")
    original = deploy.pipelines
    altered = original.get("multi")
    altered = replace(altered, digest="changed")
    deploy = replace(
        deploy, pipelines=PipelineCatalog({**original.pipelines, "multi": altered}, "single")
    )
    before = len(COMMANDS)
    with pytest.raises(PipelineChangedError):
        await resume_after_decision(request_id, deploy)
    assert len(COMMANDS) == before
    assert deploy.storage.sessions.get_session(session_id).run_status == "awaiting_human"
    deploy.storage.close()


async def test_pipeline_selected_once_for_direct_runner(tmp_path):
    class Selector:
        calls = 0

        def select(self, alert, metadata):
            self.calls += 1
            return "diagnose"

    deploy, _, model = deployment(tmp_path)
    selector = Selector()
    deploy = replace(deploy, pipeline_selectors=(selector,))
    chosen = await select_pipeline(deploy.pipelines, alert=ALERT, selectors=(selector,))
    session = deploy.storage.sessions.create_session(ALERT)
    await run_incident(ALERT, session.session_id, deploy, selection=chosen, fire_post_hooks=False)
    assert selector.calls == 1 and model.callers == ["rca"]
    deploy.storage.close()


async def test_selected_profile_uses_its_client_and_requires_a_factory(tmp_path):
    deploy, _, original_model = deployment(tmp_path)
    alternate = deploy.profile.model_copy(update={"name": "alternate", "model": "openai/alternate"})
    agents = builtin_registry().resolve_many(
        {"rca": AgentEntry(implementation="rca", config={"model_profile": "alternate"})},
        profiles={"alternate": alternate},
    )
    from_catalog = deploy.pipelines.get("diagnose")
    workflows = PipelineRegistry()
    workflows.register(from_catalog.workflow)
    catalog = PipelineCatalog(
        workflows.resolve_many(
            {"diagnose": PipelineConfig(implementation="diagnose")},
            agents=agents,
        ),
        "diagnose",
    )
    with pytest.raises(ValueError, match="completion_for"):
        replace(deploy, pipelines=catalog)
    alternate_deploy, _, alternate_model = deployment(tmp_path / "other")
    calls = []

    def factory(profile, settings):
        calls.append(profile.name)
        return alternate_model

    deploy = replace(deploy, pipelines=catalog, completion_for=factory)
    session = deploy.storage.sessions.create_session(ALERT)
    await run_incident(ALERT, session.session_id, deploy, fire_post_hooks=False)
    assert calls == ["alternate"]
    assert not original_model.callers and alternate_model.callers == ["rca"]
    alternate_deploy.storage.close()
    deploy.storage.close()


async def test_changed_global_behavior_refuses_resume_with_manually_supplied_catalog(tmp_path):
    deploy, session_id, request_id = await parked_with(tmp_path)
    deploy.storage.approvals.decide(request_id, status="approved", decided_by="operator")
    settings = deploy.settings.model_copy(
        update={"pipeline": deploy.settings.pipeline.model_copy(update={"max_turns": 7})}
    )
    changed = replace(deploy, settings=settings)
    before = len(COMMANDS)
    with pytest.raises(PipelineChangedError, match="deployment behavior changed"):
        await resume_after_decision(request_id, changed)
    assert len(COMMANDS) == before
    assert changed.storage.sessions.get_session(session_id).run_status == "awaiting_human"
    changed.storage.close()
