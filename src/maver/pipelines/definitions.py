"""Python-owned workflows and validated bindings to configured agents.

Topology and routing are ordinary Python. YAML can choose a registered workflow,
assign named agents to its stages, and populate only its declared settings. An agent
name identifies a configuration; a stage identifies its place in this workflow;
neither is the identity of a running conversation.
"""

import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from hashlib import sha256
from types import MappingProxyType
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from maver.agents import AgentDefinition
from maver.contracts.artifacts import LearningResult
from maver.contracts.state import IncidentState
from maver.orchestration.outcomes import OutcomeRecord, resolve_outcome
from maver.orchestration.routing import RoutingPolicy

__all__ = [
    "PipelineConfig",
    "PipelineRegistry",
    "ResolvedPipeline",
    "StageDefinition",
    "Transition",
    "WorkflowConfig",
    "WorkflowDefinition",
]

_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_-]*\Z")


def _check_name(name: str, kind: str) -> None:
    """Keep stage names distinct from the phase suffix used by resumable roles."""
    if not _NAME.fullmatch(name):
        raise ValueError(
            f"invalid {kind} name {name!r}; use letters, digits, hyphens and underscores"
        )


def _json(value: Any) -> str:
    """Canonicalise persisted composition without mutable nested configuration."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class WorkflowConfig(BaseModel):
    """Settings every workflow exposes; subclass to add workflow-specific knobs."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    max_steps: int = Field(default=25, ge=1, le=10000)


class PipelineConfig(BaseModel):
    """One YAML entry, whose edges can only come from its Python implementation."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    implementation: str
    agents: dict[str, str] = Field(default_factory=dict)
    learning_agent: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class StageDefinition:
    """A workflow position and the contract an agent must satisfy there.

    ``role`` is the canonical authority role, not a configured agent's name. Distinct
    stages can use the same role and configured agent without sharing conversations.
    """

    name: str
    role: str
    output_model: type[BaseModel]
    state_slot: str | None
    default_agent: str | None = None

    def __post_init__(self) -> None:
        """Validate identities before this stage reaches a workflow registry."""
        _check_name(self.name, "stage")
        _check_name(self.role, "role")
        if not issubclass(self.output_model, BaseModel):
            raise TypeError("a stage output_model must be a Pydantic model")


RouteFn = Callable[[IncidentState, WorkflowConfig, RoutingPolicy], str | None]


@dataclass(frozen=True, slots=True)
class Transition:
    """A Python predicate and all destinations it may return; None ends the run."""

    destinations: tuple[str | None, ...]
    decide: RouteFn

    def __post_init__(self) -> None:
        """Take ownership of destinations even when constructed from a mutable list."""
        object.__setattr__(self, "destinations", tuple(self.destinations))
        if not self.destinations or len(set(self.destinations)) != len(self.destinations):
            raise ValueError("a transition must declare distinct, nonempty destinations")


@dataclass(frozen=True, slots=True)
class WorkflowDefinition:
    """A complete, framework-independent workflow declared in Python."""

    name: str
    version: str
    entry: str
    stages: tuple[StageDefinition, ...]
    transitions: Mapping[str, Transition]
    config_type: type[WorkflowConfig] = WorkflowConfig
    outcome: Callable[[IncidentState], OutcomeRecord] = resolve_outcome
    post_hooks: bool = True
    completion: Callable[[IncidentState, OutcomeRecord, WorkflowConfig], OutcomeRecord] | None = (
        None
    )

    def __post_init__(self) -> None:
        """Reject missing routes and unknown edges before an incident starts."""
        _check_name(self.name, "workflow")
        if not self.version:
            raise ValueError("a workflow must declare a version for run provenance")
        if not issubclass(self.config_type, WorkflowConfig):
            raise TypeError("workflow config_type must subclass WorkflowConfig")
        if self.config_type.model_config.get("extra") != "forbid":
            raise ValueError("workflow config_type must reject unknown fields")
        object.__setattr__(self, "stages", tuple(self.stages))
        object.__setattr__(self, "transitions", MappingProxyType(dict(self.transitions)))
        names = {stage.name for stage in self.stages}
        if len(names) != len(self.stages):
            raise ValueError("workflow stage names must be unique")
        if self.entry not in names:
            raise ValueError(f"workflow entry {self.entry!r} is not a stage")
        if set(self.transitions) != names:
            raise ValueError("every workflow stage must have exactly one transition declaration")
        for stage, transition in self.transitions.items():
            unknown = {name for name in transition.destinations if name is not None} - names
            if unknown:
                raise ValueError(f"stage {stage!r} routes to unknown stages: {sorted(unknown)}")

    def stage(self, name: str) -> StageDefinition:
        """Look up a workflow position, including the entry named by a checkpoint."""
        for stage in self.stages:
            if stage.name == name:
                return stage
        raise ValueError(f"unknown stage {name!r} in workflow {self.name!r}")

    def manifest(self) -> dict[str, Any]:
        """Describe Python topology and version without serialising executable code."""
        return {
            "implementation": self.name,
            "version": self.version,
            "entry": self.entry,
            "outcome": f"{self.outcome.__module__}.{self.outcome.__qualname__}",
            "post_hooks": self.post_hooks,
            "completion": (
                f"{self.completion.__module__}.{self.completion.__qualname__}"
                if self.completion is not None
                else None
            ),
            "stages": [
                {
                    "name": stage.name,
                    "role": stage.role,
                    "output_model": f"{stage.output_model.__module__}.{stage.output_model.__qualname__}",
                    "output_schema": stage.output_model.model_json_schema(),
                    "state_slot": stage.state_slot,
                    "destinations": list(self.transitions[stage.name].destinations),
                    "router": (
                        f"{self.transitions[stage.name].decide.__module__}."
                        f"{self.transitions[stage.name].decide.__qualname__}"
                    ),
                }
                for stage in self.stages
            ],
        }


@dataclass(frozen=True, slots=True)
class ResolvedPipeline:
    """A configured workflow, with an immutable snapshot for admission and resume."""

    name: str
    workflow: WorkflowDefinition
    agents: Mapping[str, AgentDefinition]
    config_json: str
    manifest_json: str
    digest: str
    learning_agent: AgentDefinition | None = None

    def __post_init__(self) -> None:
        """Keep caller-owned binding maps out of an admitted pipeline."""
        object.__setattr__(self, "agents", MappingProxyType(dict(self.agents)))

    @property
    def config(self) -> WorkflowConfig:
        """Return fresh validated settings so nested user settings cannot mutate a run."""
        return self.workflow.config_type.model_validate_json(self.config_json)

    @property
    def entry(self) -> str:
        """The Python workflow's entry stage."""
        return self.workflow.entry

    @property
    def max_steps(self) -> int:
        """The shared execution bound, applied by both supported engines."""
        return self.config.max_steps

    def manifest(self) -> dict[str, Any]:
        """Return a fresh JSON-safe description of this exact composition."""
        result: dict[str, Any] = json.loads(self.manifest_json)
        return result

    def resolve_outcome(self, state: IncidentState) -> OutcomeRecord:
        """Resolve this workflow's result, then apply its optional completion policy.

        Persistence and post-incident hooks remain runtime responsibilities; these
        functions return a decision and do not acquire their resources.
        """
        outcome = self.workflow.outcome(state)
        if self.workflow.completion is not None:
            outcome = self.workflow.completion(state, outcome, self.config)
        if not isinstance(outcome, OutcomeRecord):
            raise TypeError("a workflow outcome resolver must return OutcomeRecord")
        return outcome

    def next_stage(self, stage: str, state: IncidentState, policy: RoutingPolicy) -> str | None:
        """Run a declared router and reject undeclared destinations immediately."""
        self.workflow.stage(stage)
        transition = self.workflow.transitions[stage]
        destination = transition.decide(state, self.config, policy)
        if destination not in transition.destinations:
            raise ValueError(
                f"workflow {self.workflow.name!r} stage {stage!r} returned undeclared "
                f"destination {destination!r}"
            )
        return destination


class PipelineRegistry:
    """Explicitly registered Python workflows; YAML cannot import executable code."""

    def __init__(self) -> None:
        """Create an independent registry for one deployment's assembly code."""
        self._workflows: dict[str, WorkflowDefinition] = {}

    def register(self, workflow: WorkflowDefinition) -> WorkflowDefinition:
        """Register a definition, refusing ambiguous replacement under the same name."""
        if workflow.name in self._workflows:
            raise ValueError(f"workflow {workflow.name!r} is already registered")
        self._workflows[workflow.name] = workflow
        return workflow

    def resolve(
        self,
        name: str,
        entry: PipelineConfig | Mapping[str, Any],
        *,
        agents: Mapping[str, AgentDefinition],
    ) -> ResolvedPipeline:
        """Validate one YAML composition against workflow and agent contracts."""
        _check_name(name, "pipeline")
        config = (
            entry if isinstance(entry, PipelineConfig) else PipelineConfig.model_validate(entry)
        )
        try:
            workflow = self._workflows[config.implementation]
        except KeyError:
            raise ValueError(f"unknown workflow implementation {config.implementation!r}") from None
        settings = workflow.config_type.model_validate(config.config)
        names = {stage.name for stage in workflow.stages}
        unknown = set(config.agents) - names
        if unknown:
            raise ValueError(f"pipeline {name!r} binds unknown stages: {sorted(unknown)}")
        bindings: dict[str, AgentDefinition] = {}
        for stage in workflow.stages:
            selected = config.agents.get(stage.name, stage.default_agent)
            if selected is None:
                raise ValueError(f"pipeline {name!r} needs an agent for stage {stage.name!r}")
            if selected not in agents:
                raise ValueError(f"pipeline {name!r} names unknown agent {selected!r}")
            agent = agents[selected]
            if agent.spec.name != stage.role:
                raise ValueError(
                    f"stage {stage.name!r} requires authority role {stage.role!r}; "
                    f"agent {selected!r} has role {agent.spec.name!r}"
                )
            if not issubclass(agent.spec.submit_model, stage.output_model):
                raise ValueError(
                    f"agent {selected!r} has incompatible output for stage {stage.name!r}: "
                    f"expected {stage.output_model.__name__}"
                )
            if agent.spec.state_slot != stage.state_slot:
                raise ValueError(
                    f"agent {selected!r} writes {agent.spec.state_slot!r}, but stage "
                    f"{stage.name!r} requires {stage.state_slot!r}"
                )
            bindings[stage.name] = agent
        learning_agent = self._learning_agent(config, workflow, agents)
        config_json = _json(settings.model_dump(mode="json"))
        manifest_json = _json(
            {
                "name": name,
                "workflow": workflow.manifest(),
                "config": json.loads(config_json),
                "agents": {stage: agent.manifest() for stage, agent in bindings.items()},
                "learning_agent": learning_agent.manifest() if learning_agent is not None else None,
            }
        )
        return ResolvedPipeline(
            name=name,
            workflow=workflow,
            agents=bindings,
            config_json=config_json,
            manifest_json=manifest_json,
            digest=sha256(manifest_json.encode("utf-8")).hexdigest(),
            learning_agent=learning_agent,
        )

    @staticmethod
    def _learning_agent(
        config: PipelineConfig,
        workflow: WorkflowDefinition,
        agents: Mapping[str, AgentDefinition],
    ) -> AgentDefinition | None:
        """Bind optional asynchronous distillation without adding a workflow stage.

        Omitting the setting inherits the configured learning agent when this
        workflow enables completion hooks. Explicit null disables distillation.
        """
        if "learning_agent" not in config.model_fields_set:
            selected = "learning" if workflow.post_hooks and "learning" in agents else None
        else:
            selected = config.learning_agent
        if selected is None:
            return None
        if selected not in agents:
            raise ValueError(f"unknown learning agent {selected!r}")
        agent = agents[selected]
        spec = agent.spec
        if (
            spec.name != "learning"
            or not issubclass(spec.submit_model, LearningResult)
            or spec.state_slot is not None
        ):
            raise ValueError(
                f"agent {selected!r} must have the learning role, LearningResult output "
                "and no workflow artifact slot"
            )
        return agent

    def resolve_many(
        self,
        entries: Mapping[str, PipelineConfig | Mapping[str, Any]],
        *,
        agents: Mapping[str, AgentDefinition],
    ) -> Mapping[str, ResolvedPipeline]:
        """Resolve all deployment entries before accepting any incidents."""
        return MappingProxyType(
            {name: self.resolve(name, entry, agents=agents) for name, entry in entries.items()}
        )
