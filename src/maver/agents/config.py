"""The configuration surface Python implementations expose to YAML.

Lists replace inherited defaults in full. Configuration contains names and values,
never import paths or executable expressions; the composition root registers code.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ComponentConfig(BaseModel):
    """Base schema for an explicitly registered configurable component."""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


class RuntimeOverrides(ComponentConfig):
    """Optional loop controls; unset fields retain the deployment's values."""

    no_tool_nudges: int | None = Field(default=None, ge=0)
    submit_guard_rejections: int | None = Field(default=None, ge=0)
    submit_validation_rejections: int | None = Field(default=None, ge=0)
    repeat_call_soft: int | None = Field(default=None, ge=0)
    repeat_call_hard: int | None = Field(default=None, ge=0)
    repeat_call_stop: int | None = Field(default=None, ge=0)
    max_tool_calls_per_turn: int | None = Field(default=None, gt=0)
    keep_reasoning_max_chars: int | None = Field(default=None, ge=0)
    finalizer_rescues: int | None = Field(default=None, ge=0)
    finalizer_min_content_chars: int | None = Field(default=None, ge=0)
    finalizer_tool_results: int | None = Field(default=None, ge=0)
    finalizer_tool_result_chars: int | None = Field(default=None, ge=0)
    finalizer_assistant_chars: int | None = Field(default=None, ge=0)


class AgentLimits(ComponentConfig):
    """Per-agent loop configuration; environment overlays may still narrow it."""

    max_turns: int | None = Field(default=None, gt=0)
    runtime: RuntimeOverrides = Field(default_factory=RuntimeOverrides)


class AgentMemory(ComponentConfig):
    """Whether this agent may use the deployment's configured experience store."""

    enabled: bool | None = None


class MiddlewareConfig(ComponentConfig):
    """One registered middleware factory and its typed configuration values."""

    name: str = Field(min_length=1)
    config: dict[str, Any] = Field(default_factory=dict)


class AgentConfig(ComponentConfig):
    """Common settings inherited by every agent implementation's config schema.

    An implementation subclasses this model to add supported settings or change
    defaults. Sampling remains one complete model profile, never per-agent knobs.
    """

    model_profile: str | None = Field(default=None, min_length=1)
    prompt: str | None = Field(default=None, min_length=1)
    tools: tuple[str, ...] | None = None
    middleware: tuple[MiddlewareConfig, ...] = ()
    limits: AgentLimits = Field(default_factory=AgentLimits)
    memory: AgentMemory = Field(default_factory=AgentMemory)

    @model_validator(mode="after")
    def _unique_components(self) -> "AgentConfig":
        """Reject ambiguous duplicate tools or middleware state namespaces."""
        for label, names in (
            ("tools", self.tools or ()),
            ("middleware", tuple(item.name for item in self.middleware)),
        ):
            if any(not name.strip() for name in names) or len(set(names)) != len(names):
                raise ValueError(f"{label} names must be non-empty and unique")
        return self


class AgentEntry(ComponentConfig):
    """A final named YAML agent, before its Python schema is known."""

    implementation: str = Field(min_length=1)
    config: dict[str, Any] = Field(default_factory=dict)
