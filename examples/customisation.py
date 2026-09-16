#!/usr/bin/env python3
"""Run an inherited agent in a custom workflow, entirely offline.

    uv run python examples/customisation.py

The scripted completion only demonstrates composition; it makes no model, telemetry,
or cluster calls. For a deployment, import the registration functions into
``maver/custom.py`` and select ``examples/customisation.yaml``.
"""

import asyncio
import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import yaml

from maver.adapters.base import RoleOverlay
from maver.agents import AgentConfig, AgentRegistry, ComponentConfig, RCAAgent
from maver.agents import builtin_registry as agent_registry
from maver.contracts.artifacts import RootCause
from maver.contracts.state import IncidentState
from maver.llm.types import ModelRequest, ModelResponse, ToolCall, Usage
from maver.memory.factory import build_storage
from maver.middleware.base import TurnContext
from maver.orchestration.routing import RoutingPolicy
from maver.orchestration.runner import Deployment, run_incident
from maver.pipelines import (
    PipelineCatalog,
    PipelineRegistry,
    PipelineSelector,
    StageDefinition,
    Transition,
    WorkflowConfig,
    WorkflowDefinition,
)
from maver.pipelines import builtin_registry as pipeline_registry
from maver.roles.base import RoleContext
from maver.settings.groups import MemorySettings, PipelineSettings, ServerSettings
from maver.settings.profiles import load_profile
from maver.settings.root import Settings


class DatabaseConfig(AgentConfig):
    """Existing agent settings plus one supported local feature."""

    schema_names: tuple[str, ...] = ("orders",)


class DatabaseAgent(RCAAgent):
    """Inherit tools, guards and execution, adding database context construction."""

    version = "1"
    config_type = DatabaseConfig
    config: DatabaseConfig

    def prepare_context(self, context: RoleContext) -> RoleContext:
        """Extend the inherited preparation without widening environment authority."""
        context = super().prepare_context(context)
        overlay = context.overlay or RoleOverlay()
        return replace(
            context,
            overlay=replace(
                overlay,
                context_blocks=(
                    *overlay.context_blocks,
                    "Schemas: " + ", ".join(self.config.schema_names),
                ),
            ),
        )


class NoteConfig(ComponentConfig):
    """The middleware exposes only a note, not arbitrary code or global settings."""

    note: str = "Inspect database availability before proposing a repair."


class OperatorNote:
    """Trigger once per invocation, retaining its trigger state across approval pauses."""

    name = "operator_note"

    def __init__(self, note: str) -> None:
        """Bind immutable configuration; durable state stays in the runtime."""
        self.note = note

    def before_model(self, context: TurnContext, request: ModelRequest) -> None:
        """Queue an explicit next-call addition while leaving metadata private."""
        run, agent = context.run_state, context.agent_state
        if run is not None and agent is not None and not agent.middleware.get(self.name):
            run.add_context(self.note, source=self.name, recipient=agent.id)
            agent.middleware[self.name] = True


def note_factory(config: ComponentConfig, context: RoleContext) -> OperatorNote:
    """Build a new hook for every agent invocation."""
    assert isinstance(config, NoteConfig)
    return OperatorNote(config.note)


def register_agents(registry: AgentRegistry) -> None:
    """Use these same registrations in maver/custom.py for a real deployment."""
    registry.register("database", DatabaseAgent)
    registry.register_middleware("operator_note", note_factory, config_type=NoteConfig, version="1")


def _review(state: IncidentState, config: WorkflowConfig, policy: RoutingPolicy) -> str:
    """Choose the second independently conversing instance in Python."""
    return "review"


def _end(state: IncidentState, config: WorkflowConfig, policy: RoutingPolicy) -> None:
    """Finish this diagnostic workflow without invoking a remediator."""
    return None


DATABASE_REVIEW = WorkflowDefinition(
    name="database_review",
    version="1",
    entry="investigate",
    stages=(
        StageDefinition("investigate", "rca", RootCause, "root_cause", "database"),
        StageDefinition("review", "rca", RootCause, "root_cause", "database"),
    ),
    transitions={
        "investigate": Transition(("review",), _review),
        "review": Transition((None,), _end),
    },
    post_hooks=False,
)


def register_pipelines(registry: PipelineRegistry) -> None:
    """Register topology, contracts and routing in Python."""
    registry.register(DATABASE_REVIEW)


class DatabaseSelector:
    """An optional selector can use private admission metadata."""

    def select(self, alert: Mapping[str, Any], metadata: Mapping[str, Any]) -> str | None:
        """Select a configured name or abstain and use the deployment default."""
        return "database_review" if metadata.get("team") == "database" else None


def pipeline_selectors() -> tuple[PipelineSelector, ...]:
    """Explicit incident selection takes precedence over this optional hook."""
    return (DatabaseSelector(),)


class _OfflineModel:
    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest, *, caller: str) -> ModelResponse:
        self.requests.append(request)
        diagnosis = RootCause(
            fault_type="example",
            affected_resource="database/orders",
            confidence=0.5,
            evidence=["Synthetic evidence; no real diagnosis was performed."],
            summary="Offline composition demonstration.",
        ).model_dump()
        return ModelResponse(
            text=None,
            tool_calls=(
                ToolCall("example", "submit_root_cause", diagnosis, json.dumps(diagnosis)),
            ),
            finish_reason="stop",
            reasoning=None,
            usage=Usage(),
            duration_ms=0,
            attempts=1,
        )


class _OfflineTransport:
    name = "offline"
    capabilities = frozenset()
    timeout_seconds = 1


async def main() -> None:
    """Exercise local registration, YAML composition and isolated conversations."""
    root = Path(__file__).resolve().parents[1]
    document = yaml.safe_load((root / "examples/customisation.yaml").read_text())
    agents = agent_registry()
    workflows = pipeline_registry()
    register_agents(agents)
    register_pipelines(workflows)
    definitions = agents.resolve_many(document["agents"])
    pipelines = workflows.resolve_many(document["pipelines"], agents=definitions)
    catalog = PipelineCatalog(pipelines, document["default_pipeline"])
    model = _OfflineModel()
    with TemporaryDirectory(prefix="maver-example-") as directory:
        settings = Settings(
            server=ServerSettings(workdir=Path(directory)),
            pipeline=PipelineSettings(auto_snapshot=False, engine="sequential"),
            memory=MemorySettings(enabled=False, learning_enabled=False),
        )
        storage = build_storage(settings)
        deployment = Deployment(
            settings=settings,
            storage=storage,
            completion=model,
            profile=load_profile("qwen3.6-35b-a3b", root / "configs/models"),
            transport_for=lambda _: _OfflineTransport(),
            pipelines=catalog,
            pipeline_selectors=pipeline_selectors(),
        )
        alert = {"alertname": "DatabaseUnavailable", "namespace": "example"}
        session = storage.sessions.create_session(alert)
        try:
            result = await run_incident(
                alert,
                session.session_id,
                deployment,
                metadata={"team": "database", "ticket_id": "PRIVATE-123"},
                fire_post_hooks=False,
            )
            state = result["runtime"]
            sent = "\n".join(str(request.messages) for request in model.requests)
            assert len(state.agents) == 2
            assert "PRIVATE-123" not in sent
            assert "Check the database before deciding" in sent
            assert "Schemas: orders, customers" in sent
            print(
                json.dumps(
                    {
                        "pipeline": state.selection["name"],
                        "conversations": len(state.agents),
                        "private_metadata_exposed": False,
                    },
                    indent=2,
                )
            )
        finally:
            storage.close()


if __name__ == "__main__":
    asyncio.run(main())
