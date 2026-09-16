"""Subclassable agent behaviour and immutable, configured definitions.

The definition is shared. An implementation object and all mutable component
configuration are constructed anew for each invocation. Conversations belong to
the runtime's agent instance, never to either a class or a YAML declaration.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, ClassVar

from maver.adapters.base import RoleOverlay
from maver.agents.config import AgentConfig, ComponentConfig
from maver.middleware.base import AgentHook
from maver.roles.base import RoleContext, RoleOutcome, RoleSpec
from maver.settings.profiles import ModelProfile
from maver.settings.root import Settings


class Agent:
    """Python behaviour inherited by configured agents.

    Subclasses provide a role spec, extend ``config_type``, and optionally override
    ``prepare_context`` or ``run``. The role spec retains the canonical authority
    role, output contract and submission behaviour; a YAML name never changes them.
    """

    version: ClassVar[str] = "1"
    config_type: ClassVar[type[AgentConfig]] = AgentConfig
    role_spec: ClassVar[RoleSpec]

    def __init__(self, config: AgentConfig) -> None:
        """Bind this invocation to an independent, schema-checked configuration."""
        if type(config) is not self.config_type:
            raise TypeError(f"{type(self).__name__} requires {self.config_type.__name__}")
        self.config = config

    def prepare_context(self, context: RoleContext) -> RoleContext:
        """Apply configured bounds; subclasses may add input projection or guards.

        The overlay also reaches implementations with a trigger-specific turn
        budget, such as Learning. Existing environment bounds remain the ceiling.
        """
        bound = self.config.limits.max_turns
        if bound is not None:
            overlay = context.overlay or RoleOverlay()
            if overlay.max_turns is not None:
                bound = min(bound, overlay.max_turns)
            return replace(context, overlay=replace(overlay, max_turns=bound))
        return context

    async def run(self, context: RoleContext) -> RoleOutcome:
        """Run inherited behaviour using the runtime's already bound components."""
        return await self.role_spec.run(self.prepare_context(context))


MiddlewareFactory = Callable[[ComponentConfig, RoleContext], AgentHook]


@dataclass(frozen=True, slots=True)
class ResolvedMiddleware:
    """A validated factory declaration; no live hook or mutable config is shared."""

    name: str
    config_json: str
    config_type: type[ComponentConfig]
    factory: MiddlewareFactory
    version: str = "1"

    def create(self, context: RoleContext) -> AgentHook:
        """Construct a fresh hook for the runtime's current agent instance."""
        return self.factory(self.config_type.model_validate_json(self.config_json), context)


@dataclass(frozen=True, slots=True)
class AgentDefinition:
    """One immutable named agent resolved from Python plus YAML.

    JSON snapshots prevent nested lists/dicts in user-defined Pydantic schemas from
    mutating the deployment after validation. Accessors give callers fresh copies.
    """

    name: str
    implementation: str
    config_json: str
    agent_type: type[Agent]
    tool_names: tuple[str, ...]
    profile_json: str | None = None
    middleware: tuple[ResolvedMiddleware, ...] = ()
    prompt_digest: str = ""
    deployment_digest: str | None = None

    @property
    def config(self) -> AgentConfig:
        """Return an independent configuration with the implementation's exact type."""
        return self.agent_type.config_type.model_validate_json(self.config_json)

    @property
    def profile(self) -> ModelProfile | None:
        """Return the selected complete profile, or inherit the deployment profile."""
        return ModelProfile.model_validate_json(self.profile_json) if self.profile_json else None

    @property
    def spec(self) -> RoleSpec:
        """Bind inherited behaviour without changing the canonical authority role."""
        return replace(
            self.agent_type.role_spec,
            prompt=self.config.prompt or self.agent_type.role_spec.prompt,
            run=self._run,
        )

    async def _run(self, context: RoleContext) -> RoleOutcome:
        return await self.agent_type(self.config).run(context)

    def settings_for(self, settings: Settings) -> Settings:
        """Apply exposed settings before the runtime constructs tools and prompts.

        The existing deployment is returned unchanged for inherited defaults. The
        runtime still applies environment limits, store availability and authority.
        """
        config = self.config
        updates: dict[str, Any] = {}
        if config.limits.max_turns is not None:
            updates["pipeline"] = settings.pipeline.model_copy(
                update={"max_turns": config.limits.max_turns}
            )
        runtime = config.limits.runtime.model_dump(exclude_none=True)
        if runtime:
            # model_copy does not validate cross-field rules. Revalidation of the
            # merged complete group catches repeat thresholds in the wrong order.
            updates["runtime"] = type(settings.runtime).model_validate(
                settings.runtime.model_dump() | runtime
            )
        if config.memory.enabled is not None:
            updates["memory"] = settings.memory.model_copy(
                update={"enabled": settings.memory.enabled and config.memory.enabled}
            )
        return settings.model_copy(update=updates) if updates else settings

    def make_middleware(self, context: RoleContext) -> tuple[AgentHook, ...]:
        """Construct configured hooks in order, inside the runtime's mandatory gate."""
        return tuple(item.create(context) for item in self.middleware)

    def manifest(self) -> dict[str, Any]:
        """Snapshot executable identity and resolved configuration for provenance."""
        role = self.agent_type.role_spec
        return {
            "name": self.name,
            "implementation": self.implementation,
            "version": self.agent_type.version,
            "python_type": f"{self.agent_type.__module__}.{self.agent_type.__qualname__}",
            "config": json.loads(self.config_json),
            "role": role.name,
            "submit_tool": role.submit_tool,
            "submit_model": f"{role.submit_model.__module__}.{role.submit_model.__qualname__}",
            "state_slot": role.state_slot,
            "prompt": self.config.prompt or role.prompt,
            "prompt_digest": self.prompt_digest,
            "deployment_digest": self.deployment_digest,
            "tools": list(self.tool_names),
            "profile": json.loads(self.profile_json) if self.profile_json else None,
            "middleware": [
                {
                    "name": item.name,
                    "factory": f"{item.factory.__module__}.{item.factory.__qualname__}",
                    "version": item.version,
                    "config": json.loads(item.config_json),
                }
                for item in self.middleware
            ],
        }
