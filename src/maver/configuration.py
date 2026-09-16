"""Resolve the deployment's YAML using explicitly registered local Python code."""

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from maver import custom
from maver.agents import AgentDefinition, AgentEntry
from maver.agents import builtin_registry as agent_registry
from maver.llm.sampling import assert_sampling_is_pinned, resolve_sampling
from maver.pipelines import PipelineCatalog, PipelineConfig, PipelineSelector
from maver.pipelines import builtin_registry as pipeline_registry
from maver.prompts.loader import overridden_roles
from maver.settings.profiles import ModelProfile, load_profile
from maver.settings.root import Settings

_SECRET = re.compile(r"(^|_)(key|token|secret|password|dsn)(?:_literal)?$", re.IGNORECASE)


def _redact(value: Any) -> Any:
    """Keep ad hoc component credential fields out of human-readable config output."""
    if isinstance(value, dict):
        return {
            key: (
                "**********" if child is not None and _SECRET.search(str(key)) else _redact(child)
            )
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [_redact(child) for child in value]
    return value


class _Document(BaseModel):
    """The complete YAML vocabulary; Python alone owns implementations and edges."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    agents: dict[str, AgentEntry] = Field(default_factory=dict)
    pipelines: dict[str, PipelineConfig] = Field(default_factory=dict)
    default_pipeline: str | None = None


class _UniqueLoader(yaml.SafeLoader):
    """Refuse duplicate declarations instead of silently taking the last one."""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
        """Construct a safe mapping while preserving typo and duplicate detection."""
        self.flatten_mapping(node)
        result: dict[Any, Any] = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in result:
                raise ValueError(f"duplicate YAML key {key!r}")
            result[key] = self.construct_object(value_node, deep=deep)
        return result


@dataclass(frozen=True, slots=True)
class Configuration:
    """Validated configured agents, workflows, and optional admission selectors."""

    agents: Mapping[str, AgentDefinition]
    pipelines: PipelineCatalog
    selectors: tuple[PipelineSelector, ...] = ()

    def manifest(self) -> dict[str, Any]:
        """Record the concrete deployment composition without executable objects."""
        result = {
            "default_pipeline": self.pipelines.default,
            "agents": {name: agent.manifest() for name, agent in self.agents.items()},
            "pipelines": {
                name: pipeline.manifest() for name, pipeline in self.pipelines.pipelines.items()
            },
            "selectors": [
                f"{type(selector).__module__}.{type(selector).__qualname__}"
                for selector in self.selectors
            ],
        }
        redacted: dict[str, Any] = _redact(result)
        return redacted


def _read(path: Path | None) -> _Document:
    if path is None:
        return _Document()
    with path.open(encoding="utf-8") as stream:
        raw = yaml.load(stream, Loader=_UniqueLoader)
    return _Document.model_validate(raw if raw is not None else {})


def _json_value(value: Any) -> Any:
    """Canonicalise unordered settings and avoid serialising credentials in plaintext."""
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    return str(value)


def behavior_digest(settings: Settings, profile: ModelProfile) -> str:
    """Fingerprint execution behaviour independently of paths and default selection.

    Credentials are masked before hashing. Actual prompt override bytes are hashed
    through the prompt loader; source directories and the selected YAML filename do
    not determine execution identity. Ordered lists retain their declared order.
    """
    behaviour = {
        group: getattr(settings, group).model_dump()
        for group in ("pipeline", "runtime", "memory", "verify", "tools", "authority", "transport")
    }
    for field in ("mode", "engine"):
        behaviour["pipeline"].pop(field, None)
    for field in ("backend", "postgres_dsn", "postgres_schema", "learning_config_dir"):
        behaviour["memory"].pop(field, None)
    provenance = {
        "profile": profile.model_dump(),
        "sampling": resolve_sampling(profile, settings).as_record(),
        "model": {
            "endpoint_override": settings.model.endpoint_override,
        },
        "prompts": overridden_roles(settings.prompts.override_dir),
        "learning_files": {
            path.name: sha256(path.read_bytes()).hexdigest()
            for path in sorted(settings.memory.learning_config_dir.glob("*.yaml"))
        }
        if settings.memory.learning_enabled
        else {},
        "behaviour": behaviour,
    }
    return sha256(json.dumps(provenance, sort_keys=True, default=_json_value).encode()).hexdigest()


def load_configuration(
    settings: Settings, *, default_profile: ModelProfile | None = None
) -> Configuration:
    """Load and validate all configured behaviour before accepting any incident.

    Shipped agents and pipelines remain available unless a YAML entry explicitly
    replaces their configuration. Python extensions are registered in ``custom.py``;
    YAML never names modules to import or expressions to evaluate.
    """
    document = _read(settings.customisation)
    agents = agent_registry()
    workflows = pipeline_registry()
    custom.register_agents(agents)
    custom.register_pipelines(workflows)
    entries = {
        name: AgentEntry(implementation=name)
        for name in ("rca", "planning", "execution", "verify", "learning", "single")
    } | document.agents
    profiles = {
        name: load_profile(name, settings.model.profile_dir)
        for name in agents.required_profiles(entries)
    }
    resolved = agents.resolve_many(
        entries, profiles=profiles, override_dir=settings.prompts.override_dir
    )
    default_profile = default_profile or load_profile(
        settings.model.profile, settings.model.profile_dir
    )
    stamped: dict[str, AgentDefinition] = {}
    for name, agent in resolved.items():
        effective = agent.settings_for(settings)
        profile = agent.profile or default_profile
        assert_sampling_is_pinned(profile, resolve_sampling(profile, effective))
        digest = behavior_digest(effective, profile)
        stamped[name] = replace(agent, deployment_digest=digest)
    resolved = MappingProxyType(stamped)
    pipeline_entries = {
        name: PipelineConfig(implementation=name) for name in ("multi", "single", "diagnose")
    } | document.pipelines
    pipelines = workflows.resolve_many(pipeline_entries, agents=resolved)
    selectors = tuple(custom.pipeline_selectors())
    if any(not callable(getattr(selector, "select", None)) for selector in selectors):
        raise TypeError("a pipeline selector must implement select(alert, metadata)")
    return Configuration(
        resolved,
        PipelineCatalog(pipelines, document.default_pipeline or settings.pipeline.mode),
        selectors,
    )
