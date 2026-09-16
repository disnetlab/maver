"""Explicit local registration and startup validation for configured agents."""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Any

from maver.agents.base import Agent, AgentDefinition, MiddlewareFactory, ResolvedMiddleware
from maver.agents.config import AgentConfig, AgentEntry, ComponentConfig, MiddlewareConfig
from maver.prompts.loader import FILLABLE_MARKERS, markers, override_for, read_asset
from maver.settings.profiles import ModelProfile
from maver.tools.allowlists import ROLE_TOOLSETS
from maver.tools.context import KUBECTL_ROUTES
from maver.tools.registry import description_for, load_descriptions, registered
from maver.tools.schema import normalise_schema

_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_-]*\Z")


class AgentConfigurationError(ValueError):
    """An agent names unsupported behaviour or inconsistent components."""


def _check_name(name: str) -> None:
    if not _NAME.fullmatch(name):
        raise AgentConfigurationError(f"invalid component name {name!r}")


@dataclass(frozen=True, slots=True)
class _MiddlewareRegistration:
    config_type: type[ComponentConfig]
    factory: MiddlewareFactory
    version: str


class AgentRegistry:
    """Code registered by a deployment's Python composition root.

    YAML can select only these implementations and factories. No import-by-string,
    plugin discovery, or process-wide mutation is performed here.
    """

    def __init__(self) -> None:
        """Start an independent registry, including no implicit custom code."""
        self._implementations: dict[str, type[Agent]] = {}
        self._middleware: dict[str, _MiddlewareRegistration] = {}

    def register(self, name: str, implementation: type[Agent]) -> None:
        """Register an inheritable Python implementation under a stable YAML name."""
        _check_name(name)
        if name in self._implementations:
            raise AgentConfigurationError(f"agent implementation {name!r} is already registered")
        if not hasattr(implementation, "role_spec"):
            raise AgentConfigurationError(
                "an implementation must subclass Agent and define role_spec"
            )
        if not implementation.version:
            raise AgentConfigurationError("an agent must declare a version")
        schema = implementation.config_type
        if not issubclass(schema, AgentConfig):
            raise AgentConfigurationError("agent config_type must subclass AgentConfig")
        if schema.model_config.get("extra") != "forbid":
            raise AgentConfigurationError("agent config_type must reject unknown fields")
        for parent in implementation.__bases__:
            if issubclass(parent, Agent) and not issubclass(schema, parent.config_type):
                raise AgentConfigurationError("agent config_type must extend its parent's schema")
        self._implementations[name] = implementation

    def register_middleware(
        self,
        name: str,
        factory: MiddlewareFactory,
        *,
        config_type: type[ComponentConfig] = ComponentConfig,
        version: str = "1",
    ) -> None:
        """Register a typed factory which receives the current runtime context."""
        _check_name(name)
        if name in self._middleware:
            raise AgentConfigurationError(f"middleware {name!r} is already registered")
        if not issubclass(config_type, ComponentConfig):
            raise AgentConfigurationError("middleware config_type must subclass ComponentConfig")
        if config_type.model_config.get("extra") != "forbid":
            raise AgentConfigurationError("middleware config_type must reject unknown fields")
        if not version:
            raise AgentConfigurationError("middleware must declare a version")
        self._middleware[name] = _MiddlewareRegistration(config_type, factory, version)

    def resolve(
        self,
        name: str,
        entry: AgentEntry | Mapping[str, Any],
        *,
        profiles: Mapping[str, ModelProfile] | None = None,
        override_dir: Path | None = None,
    ) -> AgentDefinition:
        """Validate YAML against selected code and capture an immutable definition."""
        _check_name(name)
        declaration = entry if isinstance(entry, AgentEntry) else AgentEntry.model_validate(entry)
        implementation = self._implementations.get(declaration.implementation)
        if implementation is None:
            raise AgentConfigurationError(
                f"unknown agent implementation {declaration.implementation!r}"
            )
        config = implementation.config_type.model_validate(declaration.config)
        spec = implementation.role_spec
        tools = config.tools if config.tools is not None else ROLE_TOOLSETS.get(spec.name)
        if tools is None:
            raise AgentConfigurationError(f"agent {name!r} must declare its tools")
        self._validate_tools(name, implementation, tools)
        prompt = config.prompt or spec.prompt
        template = override_for(prompt, override_dir)
        if template is None:
            template = read_asset(f"roles/{prompt}.txt")
        unknown_markers = markers(template) - FILLABLE_MARKERS
        if unknown_markers:
            raise AgentConfigurationError(f"unfillable prompt markers: {sorted(unknown_markers)}")

        profile = None
        if config.model_profile is not None:
            profile = (profiles or {}).get(config.model_profile)
            if profile is None:
                raise AgentConfigurationError(f"unknown model profile {config.model_profile!r}")
        middleware: list[ResolvedMiddleware] = []
        for item in config.middleware:
            registration = self._middleware.get(item.name)
            if registration is None:
                raise AgentConfigurationError(f"unknown middleware {item.name!r}")
            resolved = registration.config_type.model_validate(item.config)
            middleware.append(
                ResolvedMiddleware(
                    item.name,
                    resolved.model_dump_json(),
                    registration.config_type,
                    registration.factory,
                    registration.version,
                )
            )
        config = config.model_copy(
            update={
                "middleware": tuple(
                    MiddlewareConfig(
                        name=item.name,
                        config=item.config_type.model_validate_json(item.config_json).model_dump(),
                    )
                    for item in middleware
                )
            }
        )
        return AgentDefinition(
            name=name,
            implementation=declaration.implementation,
            config_json=config.model_dump_json(),
            agent_type=implementation,
            tool_names=tuple(tools),
            profile_json=profile.model_dump_json() if profile is not None else None,
            middleware=tuple(middleware),
            prompt_digest=sha256(template.encode()).hexdigest(),
        )

    @staticmethod
    def _validate_tools(name: str, implementation: type[Agent], tools: tuple[str, ...]) -> None:
        specs = registered()
        unknown = set(tools) - specs.keys()
        if unknown:
            raise AgentConfigurationError(f"agent {name!r} names unknown tools {sorted(unknown)}")
        role = implementation.role_spec
        descriptions = load_descriptions()
        for tool_name in tools:
            spec = specs[tool_name]
            if spec.safety != "submit" and spec.dispatch is None:
                raise AgentConfigurationError(f"tool {tool_name!r} has no implementation")
            for route in KUBECTL_ROUTES:
                description_for(
                    tool_name,
                    role.name,
                    descriptions=spec.descriptions or descriptions,
                    route=route,
                )
        if role.submit_tool not in tools:
            raise AgentConfigurationError(f"agent {name!r} must include {role.submit_tool!r}")
        submit = specs[role.submit_tool]
        if submit.safety != "submit":
            raise AgentConfigurationError(f"{role.submit_tool!r} is not a submission tool")
        if submit.parameters != normalise_schema(role.submit_model.model_json_schema()):
            raise AgentConfigurationError(
                f"agent {name!r} output schema differs from its submission tool"
            )

    def required_profiles(
        self, entries: Mapping[str, AgentEntry | Mapping[str, Any]]
    ) -> frozenset[str]:
        """Find model profiles selected by YAML or inherited Python config defaults."""
        names: set[str] = set()
        for entry in entries.values():
            declaration = (
                entry if isinstance(entry, AgentEntry) else AgentEntry.model_validate(entry)
            )
            implementation = self._implementations.get(declaration.implementation)
            if implementation is None:
                raise AgentConfigurationError(
                    f"unknown agent implementation {declaration.implementation!r}"
                )
            config = implementation.config_type.model_validate(declaration.config)
            if config.model_profile is not None:
                names.add(config.model_profile)
        return frozenset(names)

    def resolve_many(
        self,
        entries: Mapping[str, AgentEntry | Mapping[str, Any]],
        *,
        profiles: Mapping[str, ModelProfile] | None = None,
        override_dir: Path | None = None,
    ) -> Mapping[str, AgentDefinition]:
        """Resolve a YAML ``agents`` mapping without sharing mutable declarations."""
        return MappingProxyType(
            {
                name: self.resolve(name, entry, profiles=profiles, override_dir=override_dir)
                for name, entry in entries.items()
            }
        )
