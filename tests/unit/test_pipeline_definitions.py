"""Workflow contracts, immutable compositions, and once-only admission selection."""

from collections.abc import Mapping
from dataclasses import replace
from typing import Any, cast

import pytest
from pydantic import Field, ValidationError

from maver.agents import AgentDefinition
from maver.agents import builtin_registry as agent_registry
from maver.contracts.artifacts import Plan, RootCause
from maver.contracts.lifecycle import LifecycleState, RunStatus
from maver.contracts.runtime import RunState
from maver.contracts.state import IncidentState
from maver.orchestration.engine import LangGraphEngine, SequentialEngine, StepLimitExceeded
from maver.orchestration.graph import build_graph
from maver.orchestration.outcomes import Outcome, OutcomeRecord, resolve_outcome
from maver.orchestration.routing import RoutingPolicy
from maver.pipelines import (
    MULTI,
    PipelineCatalog,
    PipelineChangedError,
    PipelineRefusal,
    PipelineRegistry,
    ResolvedPipeline,
    StageDefinition,
    Transition,
    WorkflowConfig,
    WorkflowDefinition,
    builtin_registry,
    restore_selection,
    select_pipeline,
)

POLICY = RoutingPolicy(verify_enabled=True, remediation_retries=2, inconclusive_retries=1)


@pytest.fixture
def agents() -> Mapping[str, AgentDefinition]:
    return agent_registry().resolve_many(
        {
            name: {"implementation": name}
            for name in ("rca", "planning", "execution", "verify", "single")
        }
    )


@pytest.fixture
def pipelines(agents: Mapping[str, AgentDefinition]) -> Mapping[str, ResolvedPipeline]:
    return builtin_registry().resolve_many(
        {name: {"implementation": name} for name in ("multi", "single", "diagnose")},
        agents=agents,
    )


@pytest.fixture
def catalog(pipelines: Mapping[str, ResolvedPipeline]) -> PipelineCatalog:
    return PipelineCatalog(pipelines, default="multi")


class LocalConfig(WorkflowConfig):
    second_opinion: bool = False
    labels: list[str] = Field(default_factory=list)


def first_route(state: IncidentState, config: WorkflowConfig, policy: RoutingPolicy) -> str | None:
    return "second" if cast(LocalConfig, config).second_opinion else None


def second_route(state: IncidentState, config: WorkflowConfig, policy: RoutingPolicy) -> str | None:
    return "first" if state.get("error") else None


def local_workflow() -> WorkflowDefinition:
    return WorkflowDefinition(
        name="local",
        version="1",
        entry="first",
        stages=(
            StageDefinition("first", "rca", RootCause, "root_cause", "rca"),
            StageDefinition("second", "rca", RootCause, "root_cause", "rca"),
        ),
        transitions={
            "first": Transition(("second", None), first_route),
            "second": Transition(("first", None), second_route),
        },
        config_type=LocalConfig,
    )


def resolve_local(agents: Mapping[str, AgentDefinition], **config: Any) -> ResolvedPipeline:
    registry = PipelineRegistry()
    registry.register(local_workflow())
    return registry.resolve(
        "local_configured", {"implementation": "local", "config": config}, agents=agents
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"entry": "absent"}, "entry"),
        ({"stages": (_ := StageDefinition("first", "rca", RootCause, "root_cause"), _)}, "unique"),
        ({"transitions": {"first": Transition((None,), first_route)}}, "every workflow stage"),
        (
            {
                "transitions": {
                    "first": Transition(("missing",), first_route),
                    "second": Transition((None,), second_route),
                }
            },
            "unknown stages",
        ),
    ],
)
def test_inconsistent_python_topology_fails_before_admission(
    overrides: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(local_workflow(), **overrides)


def test_yaml_cannot_invent_topology_or_import_an_implementation(
    agents: Mapping[str, AgentDefinition],
) -> None:
    registry = builtin_registry()
    with pytest.raises(ValidationError, match="Extra inputs"):
        registry.resolve("bad", {"implementation": "multi", "entry": "execution"}, agents=agents)
    with pytest.raises(ValueError, match="unknown workflow implementation"):
        registry.resolve("bad", {"implementation": "my.module:factory"}, agents=agents)
    with pytest.raises(ValidationError, match="Extra inputs"):
        registry.resolve(
            "bad", {"implementation": "multi", "config": {"unknown": True}}, agents=agents
        )
    with pytest.raises(ValidationError, match="greater than or equal"):
        registry.resolve(
            "bad", {"implementation": "multi", "config": {"max_steps": 0}}, agents=agents
        )


@pytest.mark.parametrize(
    ("stage", "message"),
    [
        (StageDefinition("first", "execution", RootCause, "root_cause", "rca"), "authority role"),
        (StageDefinition("first", "rca", Plan, "root_cause", "rca"), "incompatible output"),
        (StageDefinition("first", "rca", RootCause, "plan", "rca"), "requires 'plan'"),
    ],
)
def test_bindings_validate_authority_output_and_artifact_slot(
    agents: Mapping[str, AgentDefinition], stage: StageDefinition, message: str
) -> None:
    registry = PipelineRegistry()
    registry.register(replace(local_workflow(), stages=(stage, local_workflow().stages[1])))
    with pytest.raises(ValueError, match=message):
        registry.resolve("bad", {"implementation": "local"}, agents=agents)


def test_missing_and_unknown_binding_names_fail_at_resolution(
    agents: Mapping[str, AgentDefinition],
) -> None:
    registry = builtin_registry()
    with pytest.raises(ValueError, match="unknown stages"):
        registry.resolve(
            "bad", {"implementation": "multi", "agents": {"other": "rca"}}, agents=agents
        )
    with pytest.raises(ValueError, match="unknown agent"):
        registry.resolve(
            "bad", {"implementation": "multi", "agents": {"rca": "absent"}}, agents=agents
        )
    custom = PipelineRegistry()
    workflow = local_workflow()
    custom.register(
        replace(
            workflow, stages=(replace(workflow.stages[0], default_agent=None), workflow.stages[1])
        )
    )
    with pytest.raises(ValueError, match="needs an agent"):
        custom.resolve("bad", {"implementation": "local"}, agents=agents)


def test_python_branches_and_backedges_use_typed_yaml_settings(
    agents: Mapping[str, AgentDefinition],
) -> None:
    normal = resolve_local(agents)
    second_opinion = resolve_local(agents, second_opinion=True, max_steps=4)
    assert normal.next_stage("first", {}, POLICY) is None
    assert second_opinion.next_stage("first", {}, POLICY) == "second"
    assert second_opinion.next_stage("second", {"error": "revisit"}, POLICY) == "first"
    assert second_opinion.max_steps == 4
    # Two workflow positions use the same definition. Runtime instances stay separate.
    assert second_opinion.agents["first"] is second_opinion.agents["second"]
    assert second_opinion.entry == "first"
    with pytest.raises(ValueError, match="unknown stage"):
        second_opinion.next_stage("missing", {}, POLICY)


def test_router_cannot_escape_its_declared_destinations(
    agents: Mapping[str, AgentDefinition],
) -> None:
    workflow = local_workflow()
    registry = PipelineRegistry()
    registry.register(
        replace(
            workflow,
            transitions={
                "first": Transition((None,), first_route),
                "second": workflow.transitions["second"],
            },
        )
    )
    pipeline = registry.resolve(
        "bad_route", {"implementation": "local", "config": {"second_opinion": True}}, agents=agents
    )
    with pytest.raises(ValueError, match="undeclared destination 'second'"):
        pipeline.next_stage("first", {}, POLICY)


def test_resolved_composition_owns_nested_settings_bindings_and_manifest(
    agents: Mapping[str, AgentDefinition],
) -> None:
    labels = ["initial"]
    pipeline = resolve_local(agents, labels=labels)
    original_digest = pipeline.digest
    labels.append("caller")
    cast(LocalConfig, pipeline.config).labels.append("reader")
    manifest = pipeline.manifest()
    manifest["config"]["labels"].append("audit_reader")
    assert cast(LocalConfig, pipeline.config).labels == ["initial"]
    assert pipeline.manifest()["config"]["labels"] == ["initial"]
    assert pipeline.digest == original_digest
    with pytest.raises(TypeError):
        pipeline.agents["first"] = agents["single"]  # type: ignore[index]
    with pytest.raises(TypeError):
        pipeline.workflow.transitions["first"] = Transition((None,), first_route)  # type: ignore[index]
    assert resolve_local(agents, labels=["changed"]).digest != pipeline.digest


def test_workflow_outcome_and_completion_policy_do_not_require_builtin_artifacts(
    agents: Mapping[str, AgentDefinition],
) -> None:
    completed = OutcomeRecord(
        Outcome.SUCCESS, LifecycleState.TENTATIVE_SUCCESS, RunStatus.COMPLETED, "local check passed"
    )

    def local_outcome(state: IncidentState) -> OutcomeRecord:
        return completed

    def record_completion(
        state: IncidentState, outcome: OutcomeRecord, config: WorkflowConfig
    ) -> OutcomeRecord:
        return replace(outcome, notes=f"{outcome.notes}; bound {config.max_steps}")

    registry = PipelineRegistry()
    registry.register(
        replace(local_workflow(), outcome=local_outcome, completion=record_completion)
    )
    pipeline = registry.resolve("local_configured", {"implementation": "local"}, agents=agents)
    assert pipeline.resolve_outcome({}).notes == "local check passed; bound 25"
    assert pipeline.resolve_outcome({}).run_status is RunStatus.COMPLETED
    assert resolve_local(agents).resolve_outcome({}) == resolve_outcome({})


class Selector:
    def __init__(self, choice: str | None) -> None:
        self.choice = choice
        self.seen: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []

    def select(self, alert: Mapping[str, Any], metadata: Mapping[str, Any]) -> str | None:
        self.seen.append((alert, metadata))
        return self.choice


async def test_explicit_selection_skips_all_selectors(catalog: PipelineCatalog) -> None:
    selector = Selector("single")
    selected = await select_pipeline(catalog, alert={}, explicit="diagnose", selectors=(selector,))
    assert selected.pipeline.name == "diagnose"
    assert selected.source == "explicit"
    assert selector.seen == []
    with pytest.raises(ValueError, match="unknown configured pipeline"):
        await select_pipeline(catalog, alert={}, explicit="unknown", selectors=(selector,))
    assert selector.seen == []


async def test_selectors_can_abstain_choose_asynchronously_or_fall_back(
    catalog: PipelineCatalog,
) -> None:
    abstain = Selector(None)

    class AsyncSelector:
        async def select(self, alert: Mapping[str, Any], metadata: Mapping[str, Any]) -> str | None:
            return "single" if metadata["service"] == "checkout" else None

    selected = await select_pipeline(
        catalog,
        alert={"name": "latency"},
        metadata={"service": "checkout"},
        selectors=(abstain, AsyncSelector()),
    )
    assert selected.pipeline.name == "single"
    assert selected.source == "selector"
    assert len(abstain.seen) == 1
    fallback = await select_pipeline(catalog, alert={}, selectors=(abstain,))
    assert fallback.pipeline.name == "multi"
    assert fallback.source == "default"


async def test_selector_unknown_name_and_refusal_never_silently_fall_back(
    catalog: PipelineCatalog,
) -> None:
    with pytest.raises(ValueError, match="unknown configured pipeline"):
        await select_pipeline(catalog, alert={}, selectors=(Selector("unknown"),))

    class Refuse:
        def select(self, alert: Mapping[str, Any], metadata: Mapping[str, Any]) -> None:
            raise PipelineRefusal("environment requires an explicit choice")

    with pytest.raises(PipelineRefusal, match="explicit choice"):
        await select_pipeline(catalog, alert={}, selectors=(Refuse(),))


async def test_selector_context_is_private_detached_and_recursively_readonly(
    catalog: PipelineCatalog,
) -> None:
    selector = Selector(None)
    alert = {"labels": {"service": "web"}}
    metadata = {"private": {"route": ["single"]}}
    await select_pipeline(catalog, alert=alert, metadata=metadata, selectors=(selector,))
    seen_alert, seen_metadata = selector.seen[0]
    with pytest.raises(TypeError):
        seen_alert["labels"]["service"] = "changed"
    with pytest.raises(TypeError):
        seen_metadata["private"]["route"][0] = "multi"
    metadata["private"]["route"].append("changed")
    assert seen_metadata["private"]["route"] == ("single",)
    assert "private" not in seen_alert


async def test_resume_uses_pinned_choice_even_when_default_changes(
    catalog: PipelineCatalog, pipelines: Mapping[str, ResolvedPipeline]
) -> None:
    selected = await select_pipeline(catalog, alert={})
    changed_default = PipelineCatalog(pipelines, default="single")
    restored = restore_selection(changed_default, selected.manifest())
    assert restored.pipeline.name == "multi"
    assert restored.source == "default"
    assert restored.pipeline.digest == selected.pipeline.digest


async def test_resume_refuses_missing_changed_or_tampered_composition(
    catalog: PipelineCatalog,
    agents: Mapping[str, AgentDefinition],
    pipelines: Mapping[str, ResolvedPipeline],
) -> None:
    saved = (await select_pipeline(catalog, alert={})).manifest()
    removed = PipelineCatalog({"single": pipelines["single"]}, default="single")
    with pytest.raises(PipelineChangedError, match="no longer configured"):
        restore_selection(removed, saved)
    changed = builtin_registry().resolve(
        "multi", {"implementation": "multi", "config": {"max_steps": 5}}, agents=agents
    )
    with pytest.raises(PipelineChangedError, match="changed since admission"):
        restore_selection(PipelineCatalog({"multi": changed}, default="multi"), saved)
    saved["composition"]["config"]["max_steps"] = 500
    with pytest.raises(PipelineChangedError, match="changed since admission"):
        restore_selection(catalog, saved)


def test_catalog_and_registry_refuse_ambiguous_names(
    pipelines: Mapping[str, ResolvedPipeline],
) -> None:
    with pytest.raises(ValueError, match="default pipeline"):
        PipelineCatalog(pipelines, default="absent")
    with pytest.raises(ValueError, match="keys must match"):
        PipelineCatalog({"wrong": pipelines["multi"]}, default="wrong")
    registry = builtin_registry()
    with pytest.raises(ValueError, match="already registered"):
        registry.register(MULTI)


@pytest.mark.parametrize("engine", ["sequential", "langgraph"])
async def test_both_engines_execute_python_workflow_with_its_own_entry_and_bound(
    agents: Mapping[str, AgentDefinition], engine: str
) -> None:
    pipeline = resolve_local(agents, second_opinion=True, max_steps=2)
    seen: list[str] = []

    async def first(state: IncidentState) -> dict[str, Any]:
        seen.append("first")
        return {"error": None}

    async def second(state: IncidentState) -> dict[str, Any]:
        seen.append("second")
        return {}

    nodes = {"first": first, "second": second}
    runner = (
        SequentialEngine(nodes, POLICY, pipeline)
        if engine == "sequential"
        else LangGraphEngine(build_graph(nodes, POLICY, pipeline), pipeline)
    )
    result = await runner.run({})
    assert seen == ["first", "second"]
    assert result["runtime"].steps == 2
    assert result["runtime"].active_stage is None


@pytest.mark.parametrize("engine", ["sequential", "langgraph"])
async def test_python_backedges_cannot_exceed_durable_stage_budget(
    agents: Mapping[str, AgentDefinition], engine: str
) -> None:
    pipeline = resolve_local(agents, second_opinion=True, max_steps=3)
    seen: list[str] = []

    async def first(state: IncidentState) -> dict[str, Any]:
        seen.append("first")
        return {"error": "revisit"}

    async def second(state: IncidentState) -> dict[str, Any]:
        seen.append("second")
        return {}

    nodes = {"first": first, "second": second}
    runner = (
        SequentialEngine(nodes, POLICY, pipeline)
        if engine == "sequential"
        else LangGraphEngine(build_graph(nodes, POLICY, pipeline), pipeline)
    )
    # One prior completed stage remains spent even though this engine invocation is new.
    state: IncidentState = {"runtime": RunState(steps=1)}
    with pytest.raises(StepLimitExceeded, match="exceeded 3"):
        await runner.run(state)
    assert seen == ["first", "second"]
    assert state["runtime"].steps == 3


@pytest.mark.parametrize("engine", ["sequential", "langgraph"])
async def test_a_paused_stage_resumes_with_the_original_budget_charge(
    agents: Mapping[str, AgentDefinition], engine: str
) -> None:
    pipeline = resolve_local(agents, second_opinion=True, max_steps=2)
    paused = False
    seen: list[str] = []

    class Pause(Exception):
        pass

    async def first(state: IncidentState) -> dict[str, Any]:
        nonlocal paused
        if not paused:
            paused = True
            raise Pause
        seen.append("first")
        return {"error": None}

    async def second(state: IncidentState) -> dict[str, Any]:
        seen.append("second")
        return {}

    nodes = {"first": first, "second": second}
    runner = (
        SequentialEngine(nodes, POLICY, pipeline)
        if engine == "sequential"
        else LangGraphEngine(build_graph(nodes, POLICY, pipeline), pipeline)
    )
    state: IncidentState = {}
    with pytest.raises(Pause):
        await runner.run(state)
    assert state["runtime"].steps == 1
    assert state["runtime"].active_stage == "first"
    result = await SequentialEngine(nodes, POLICY, pipeline).run(state, start="first")
    assert seen == ["first", "second"]
    assert result["runtime"].steps == 2
    assert result["runtime"].active_stage is None


async def test_resume_at_a_nonentry_stage_does_not_restart_the_pipeline(
    agents: Mapping[str, AgentDefinition],
) -> None:
    pipeline = resolve_local(agents, second_opinion=True, max_steps=2)
    seen: list[str] = []

    async def first(state: IncidentState) -> dict[str, Any]:
        pytest.fail("a resumed second stage must not replay the first")

    async def second(state: IncidentState) -> dict[str, Any]:
        seen.append("second")
        return {}

    state: IncidentState = {"runtime": RunState(steps=2, active_stage="second")}
    result = await SequentialEngine({"first": first, "second": second}, POLICY, pipeline).run(
        state, start="second"
    )
    assert seen == ["second"]
    assert result["runtime"].steps == 2


def test_selecting_diagnose_finishes_without_a_separate_target_stage_flag(
    pipelines: Mapping[str, ResolvedPipeline],
) -> None:
    diagnosis = RootCause(
        fault_type="misconfiguration",
        affected_resource="service/web",
        confidence=0.8,
        evidence=["service targets an unused port"],
        summary="the service port differs from the listener",
    )
    state: IncidentState = {"target_stage": "full", "root_cause": diagnosis}
    result = pipelines["diagnose"].resolve_outcome(state)
    assert result.run_status is RunStatus.COMPLETED
    assert result.outcome is Outcome.SUCCESS
    assert pipelines["diagnose"].workflow.post_hooks is False
    assert state["target_stage"] == "full"
    assert pipelines["diagnose"].resolve_outcome({}).run_status is RunStatus.FAILED
