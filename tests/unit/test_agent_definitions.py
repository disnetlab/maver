"""Agent inheritance changes real requests while keeping definitions and runs isolated."""

import asyncio
import json
from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ConfigDict, Field, ValidationError

from maver.adapters.base import RoleOverlay
from maver.agents import (
    AgentConfig,
    AgentConfigurationError,
    AgentLimits,
    ComponentConfig,
    MiddlewareConfig,
    RCAAgent,
    builtin_registry,
)
from maver.audit.journal import Journal, ListSink
from maver.contracts.artifacts import Plan
from maver.llm.types import ModelRequest, ModelResponse, ToolCall, Usage
from maver.middleware.base import MiddlewareStack, TurnContext
from maver.roles import policy, rca
from maver.roles.base import RoleContext, RoleOutcome
from maver.settings.groups import MemorySettings, PipelineSettings
from maver.settings.profiles import load_profile
from maver.settings.root import Settings
from maver.tools.allowlists import ROLE_TOOLSETS
from maver.tools.context import ToolContext
from maver.tools.toolsets import build_toolset
from maver.tools.workspace import SessionWorkspace

PROFILE = load_profile("qwen3.6-35b-a3b", Path("configs/models"))
FIXTURE = json.loads(
    (Path(__file__).resolve().parents[1] / "fixtures/prototype_role_messages.json").read_text()
)


class Recorder:
    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest, *, caller: str) -> ModelResponse:
        self.requests.append(request)
        arguments = dict(FIXTURE["root_cause"])
        return ModelResponse(
            text=None,
            tool_calls=(ToolCall("c1", "submit_root_cause", arguments, json.dumps(arguments)),),
            finish_reason="stop",
            reasoning=None,
            usage=Usage(),
            duration_ms=1,
            attempts=1,
        )


class Transport:
    name = "fake"
    capabilities = frozenset({"kubernetes", "metrics", "logs", "traces"})
    timeout_seconds = 30


def context(tmp_path: Path) -> tuple[RoleContext, Recorder]:
    model = Recorder()
    journal = Journal(ListSink(), session_id="s1", stage="rca")
    tools = ToolContext(
        transport=Transport(),
        workspace=SessionWorkspace(tmp_path),
        role="rca",
        session_id="s1",
        journal=journal,
    )
    return (
        RoleContext(
            spec=rca.RCA,
            state={"session_id": "s1", "alert": FIXTURE["alert"]},
            handback=None,
            settings=Settings(
                pipeline=PipelineSettings(auto_snapshot=False),
                memory=MemorySettings(enabled=False, learning_enabled=False),
            ),
            completion=model,
            profile=PROFILE,
            middleware=MiddlewareStack(),
            write_lock=asyncio.Lock(),
            toolset=build_toolset("rca", tools),
            tool_context=tools,
            journal=journal,
            run_id="r1",
            session_id="s1",
        ),
        model,
    )


class DatabaseConfig(AgentConfig):
    limits: AgentLimits = AgentLimits(max_turns=12)
    include_schema_context: bool = True
    schemas: list[str] = Field(default_factory=lambda: ["orders"])


class DatabaseAgent(RCAAgent):
    config_type = DatabaseConfig
    config: DatabaseConfig

    def prepare_context(self, ctx: RoleContext) -> RoleContext:
        ctx = super().prepare_context(ctx)
        if self.config.include_schema_context:
            overlay = ctx.overlay or RoleOverlay()
            ctx = replace(
                ctx,
                overlay=replace(
                    overlay,
                    context_blocks=(
                        *overlay.context_blocks,
                        "Schemas: " + ", ".join(self.config.schemas),
                    ),
                ),
            )
        return ctx


class LargeDatabaseConfig(DatabaseConfig):
    limits: AgentLimits = AgentLimits(max_turns=20)


class LargeDatabaseAgent(DatabaseAgent):
    config_type = LargeDatabaseConfig
    config: LargeDatabaseConfig


async def test_yaml_configures_inherited_behaviour_in_a_real_role_request(tmp_path: Path) -> None:
    registry = builtin_registry()
    registry.register("database", DatabaseAgent)
    registry.register("large_database", LargeDatabaseAgent)
    entries = yaml.safe_load(
        """
        large:
          implementation: large_database
          config:
            schemas: [orders, customers]
        small:
          implementation: database
          config:
            include_schema_context: false
            limits:
              max_turns: 5
        """
    )
    definitions = registry.resolve_many(entries)
    assert definitions["large"].config.limits.max_turns == 20
    assert definitions["small"].config.limits.max_turns == 5
    assert definitions["large"].spec.name == "rca"
    assert DatabaseConfig().limits.max_turns == 12

    for name, expected in (("large", True), ("small", False)):
        ctx, recorder = context(tmp_path / name)
        definition = definitions[name]
        outcome = await definition.spec.run(
            replace(ctx, spec=definition.spec, settings=definition.settings_for(ctx.settings))
        )
        assert outcome.artifact is not None
        assert (
            "Schemas: orders, customers" in recorder.requests[0].messages[0]["content"]
        ) == expected


async def test_default_wrapper_preserves_the_complete_measured_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FixedClock:
        @staticmethod
        def now(tz: Any) -> datetime:
            return datetime(2026, 9, 17, tzinfo=UTC)

    monkeypatch.setattr(policy, "datetime", FixedClock)
    definition = builtin_registry().resolve("diagnoser", {"implementation": "rca"})
    original, original_model = context(tmp_path / "original")
    configured, configured_model = context(tmp_path / "configured")
    assert definition.settings_for(original.settings) is original.settings
    before = await rca.RCA.run(original)
    after = await definition.spec.run(replace(configured, spec=definition.spec))
    assert original_model.requests == configured_model.requests
    assert before == after
    assert definition.tool_names == ROLE_TOOLSETS["rca"]


def test_ordered_lists_replace_python_defaults_without_mutating_parent() -> None:
    class RestrictedConfig(AgentConfig):
        tools: tuple[str, ...] | None = ("get_metrics", "submit_root_cause")

    class RestrictedAgent(RCAAgent):
        config_type = RestrictedConfig

    registry = builtin_registry()
    registry.register("restricted", RestrictedAgent)
    before = dict(ROLE_TOOLSETS)
    definition = registry.resolve(
        "logs",
        {"implementation": "restricted", "config": {"tools": ["submit_root_cause", "get_logs"]}},
    )
    assert definition.tool_names == ("submit_root_cause", "get_logs")
    assert RestrictedConfig().tools == ("get_metrics", "submit_root_cause")
    assert before == ROLE_TOOLSETS


@pytest.mark.parametrize(
    ("entry", "error"),
    [
        ({"implementation": "os.system"}, "unknown agent implementation"),
        ({"implementation": "rca", "config": {"invented": True}}, "Extra inputs"),
        ({"implementation": "rca", "config": {"temperature": 0.9}}, "Extra inputs"),
        ({"implementation": "rca", "config": {"tools": ["missing"]}}, "unknown tools"),
        ({"implementation": "rca", "config": {"tools": ["get_logs"]}}, "must include"),
        ({"implementation": "rca", "config": {"tools": ["get_logs", "get_logs"]}}, "unique"),
        (
            {"implementation": "rca", "config": {"middleware": [{"name": "missing"}]}},
            "unknown middleware",
        ),
        (
            {"implementation": "rca", "config": {"model_profile": "missing"}},
            "unknown model profile",
        ),
        ({"implementation": "rca", "config": {"limits": {"max_turns": 0}}}, "greater than 0"),
        ({"implementation": "rca", "authority_role": "execution"}, "Extra inputs"),
    ],
)
def test_unsupported_configuration_fails_before_an_incident(
    entry: dict[str, Any], error: str
) -> None:
    with pytest.raises((AgentConfigurationError, ValidationError), match=error):
        builtin_registry().resolve("bad", entry)


def test_output_schema_cannot_disagree_with_the_offered_submit_tool() -> None:
    class WrongOutput(RCAAgent):
        role_spec = replace(rca.RCA, submit_model=Plan)

    registry = builtin_registry()
    registry.register("wrong", WrongOutput)
    with pytest.raises(AgentConfigurationError, match="output schema differs"):
        registry.resolve("bad", {"implementation": "wrong"})


def test_inherited_config_schema_cannot_drop_the_parent_contract() -> None:
    class WrongSchema(DatabaseAgent):
        config_type = AgentConfig

    class PermissiveConfig(AgentConfig):
        model_config = ConfigDict(extra="ignore")

    class PermissiveAgent(RCAAgent):
        config_type = PermissiveConfig

    with pytest.raises(AgentConfigurationError, match="parent's schema"):
        builtin_registry().register("wrong", WrongSchema)
    with pytest.raises(AgentConfigurationError, match="reject unknown"):
        builtin_registry().register("permissive", PermissiveAgent)
    with pytest.raises(TypeError, match="requires DatabaseConfig"):
        DatabaseAgent(AgentConfig())


async def test_definitions_snapshot_mutable_custom_config_and_create_fresh_instances(
    tmp_path: Path,
) -> None:
    class CountingAgent(DatabaseAgent):
        async def run(self, ctx: RoleContext) -> RoleOutcome:
            self.config.schemas.append("only-this-invocation")
            count = getattr(self, "count", 0) + 1
            self.count = count
            return RoleOutcome(
                None, [], metadata={"count": count, "schemas": list(self.config.schemas)}
            )

    registry = builtin_registry()
    registry.register("counting", CountingAgent)
    source = {"implementation": "counting", "config": {"schemas": ["original"]}}
    definition = registry.resolve("counting", source)
    source["config"]["schemas"].append("later-edit")
    definition.config.schemas.append("edited-copy")
    with pytest.raises(FrozenInstanceError):
        definition.name = "renamed"
    ctx, _ = context(tmp_path)
    first, second = await asyncio.gather(definition.spec.run(ctx), definition.spec.run(ctx))
    assert (
        first.metadata
        == second.metadata
        == {"count": 1, "schemas": ["original", "only-this-invocation"]}
    )
    assert definition.manifest()["config"]["schemas"] == ["original"]


def test_complete_profile_is_snapshotted_and_partial_runtime_overrides_are_validated() -> None:
    registry = builtin_registry()
    profile = PROFILE.model_copy(deep=True)
    definition = registry.resolve(
        "configured",
        {
            "implementation": "rca",
            "config": {"model_profile": "qwen", "limits": {"runtime": {"no_tool_nudges": 0}}},
        },
        profiles={"qwen": profile},
    )
    profile.limits["num_retries"] = 999
    definition.profile.limits["num_retries"] = 555
    assert definition.profile.limits["num_retries"] == PROFILE.limits["num_retries"]
    assert definition.profile.sampling == PROFILE.sampling
    settings = Settings()
    modified = definition.settings_for(settings)
    assert modified.runtime.no_tool_nudges == 0
    assert modified.runtime.repeat_call_stop == settings.runtime.repeat_call_stop
    bad = registry.resolve(
        "bad",
        {"implementation": "rca", "config": {"limits": {"runtime": {"repeat_call_soft": 100}}}},
    )
    with pytest.raises(ValidationError, match="repeat_call_soft"):
        bad.settings_for(settings)


def test_agent_memory_cannot_enable_an_unavailable_deployment_store() -> None:
    definition = builtin_registry().resolve(
        "memory", {"implementation": "rca", "config": {"memory": {"enabled": True}}}
    )
    settings = Settings(memory=MemorySettings(enabled=False, learning_enabled=False))
    assert definition.settings_for(settings).memory.enabled is False


def test_explicit_turn_cap_preserves_a_stricter_environment_bound(tmp_path: Path) -> None:
    ctx, _ = context(tmp_path)
    agent = DatabaseAgent(DatabaseConfig())
    bounded = agent.prepare_context(replace(ctx, overlay=RoleOverlay(max_turns=3)))
    assert bounded.overlay.max_turns == 3
    assert agent.prepare_context(ctx).overlay.max_turns == 12


def test_middleware_defaults_are_validated_snapshotted_and_constructed_in_order(
    tmp_path: Path,
) -> None:
    class PrefixConfig(ComponentConfig):
        prefix: str = "default"

    class Prefix:
        def __init__(self, text: str, name: str) -> None:
            self.name = name
            self.text = text
            self.calls = 0

        def before_model(self, ctx: TurnContext, request: ModelRequest) -> ModelRequest:
            self.calls += 1
            return replace(request, system=self.text + request.system)

    def factory(config: ComponentConfig, ctx: RoleContext) -> Prefix:
        assert isinstance(config, PrefixConfig)
        return Prefix(config.prefix, ctx.run_id)

    class HookConfig(AgentConfig):
        middleware: tuple[MiddlewareConfig, ...] = (MiddlewareConfig(name="first"),)

    class HookAgent(RCAAgent):
        config_type = HookConfig

    registry = builtin_registry()
    registry.register("hooks", HookAgent)
    for name in ("first", "second"):
        registry.register_middleware(name, factory, config_type=PrefixConfig)
    entry = {
        "implementation": "hooks",
        "config": {
            "middleware": [{"name": "second", "config": {"prefix": "B"}}, {"name": "first"}]
        },
    }
    definition = registry.resolve("hooked", entry)
    ctx, _ = context(tmp_path)
    a, b = definition.make_middleware(ctx), definition.make_middleware(replace(ctx, run_id="r2"))
    shaped = MiddlewareStack(a).before_model(
        TurnContext("rca", "s1", 0, "rca"), ModelRequest("S", ())
    )
    assert shaped.system == "defaultBS"
    assert all(hook.calls == 1 for hook in a)
    assert all(hook.calls == 0 for hook in b)
    assert b[0].name == "r2"
    assert definition.manifest()["config"]["middleware"][1]["config"] == {"prefix": "default"}
    with pytest.raises(ValidationError, match="Extra inputs"):
        registry.resolve(
            "bad",
            {
                "implementation": "hooks",
                "config": {"middleware": [{"name": "first", "config": {"unknown": True}}]},
            },
        )


def test_prompt_override_is_validated_before_runtime(tmp_path: Path) -> None:
    roles = tmp_path / "roles"
    roles.mkdir()
    (roles / "database.txt").write_text("Inspect the database. {{NOT_SUPPORTED}}")
    entry = {"implementation": "rca", "config": {"prompt": "database"}}
    with pytest.raises(AgentConfigurationError, match="unfillable"):
        builtin_registry().resolve("database", entry, override_dir=tmp_path)
    (roles / "database.txt").write_text("Inspect the database.")
    definition = builtin_registry().resolve("database", entry, override_dir=tmp_path)
    assert definition.spec.prompt == "database"
    assert definition.spec.name == "rca"
