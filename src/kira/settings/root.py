"""The root settings object, its validation, and the run's configuration snapshot."""

import os
from collections.abc import Iterator, Mapping
from difflib import get_close_matches
from functools import lru_cache
from typing import Any, NamedTuple

from pydantic import AliasChoices, Field, model_validator
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings

from kira.settings.authority import AuthoritySettings
from kira.settings.base import ENV_PREFIX, SettingsGroup
from kira.settings.groups import (
    IntakeSettings,
    MemorySettings,
    ModelSettings,
    ObservabilitySettings,
    PipelineSettings,
    RuntimeSettings,
    SecondPassSettings,
    ServerSettings,
    ToolSettings,
    TransportSettings,
    VerifySettings,
)

ADAPTER_ENV_NAME = f"{ENV_PREFIX}ADAPTERS"
"""Where an adapter's configuration slice is read from."""

_ADAPTER_SLICE_PREFIX = f"{ADAPTER_ENV_NAME}__"

__all__ = [
    "ADAPTER_ENV_NAME",
    "AliasedField",
    "Settings",
    "get_settings",
    "known_env_names",
    "legacy_names_in_use",
    "load_settings",
    "reset_settings",
    "unknown_prefixed_names",
]


class Settings(SettingsGroup):
    """Everything the process is configured with.

    Groups are independent ``BaseSettings`` instances, so each reads its own
    environment names. The object is frozen: settings are injected into the components
    that need them, never re-read, so a run's behaviour cannot change underneath it.
    """

    pipeline: PipelineSettings = Field(default_factory=PipelineSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    verify: VerifySettings = Field(default_factory=VerifySettings)
    secondpass: SecondPassSettings = Field(default_factory=SecondPassSettings)
    authority: AuthoritySettings = Field(default_factory=AuthoritySettings)
    intake: IntakeSettings = Field(default_factory=IntakeSettings)
    tools: ToolSettings = Field(default_factory=ToolSettings)
    transport: TransportSettings = Field(default_factory=TransportSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)
    server: ServerSettings = Field(default_factory=ServerSettings)
    adapters: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        validation_alias=AliasChoices(ADAPTER_ENV_NAME),
        description=(
            "Per-adapter configuration, keyed by registered adapter name and read from "
            "KIRA_ADAPTERS__<NAME>__<FIELD>. Untyped here on purpose: an adapter "
            "validates its own slice with its own model, which is what keeps the core "
            "free of the field names of environments it must not know about."
        ),
    )

    @model_validator(mode="after")
    def _cross_group_invariants(self) -> "Settings":
        """Reject combinations that cannot work, at startup rather than mid-incident.

        Returns:
            The validated settings.

        Raises:
            ValueError: When two groups are configured inconsistently.
        """
        if (
            self.secondpass.enabled
            and self.secondpass.delivery != "none"
            and not self.memory.enabled
        ):
            raise ValueError(
                "secondpass.enabled with delivery != 'none' requires memory.enabled: the "
                "second pass has no precedent to deliver. Use delivery='none' for the "
                "resampling control, which is the configuration that deliberately uses "
                "no memory."
            )
        if self.memory.learning_enabled and not self.memory.enabled:
            raise ValueError(
                "memory.learning_enabled requires memory.enabled: distilling records "
                "that nothing can retrieve is work with no consumer."
            )
        if not self.intake.http_enabled and not self.intake.kafka.enabled:
            raise ValueError(
                "no intake is enabled: set intake.http_enabled or intake.kafka.enabled, "
                "otherwise the service starts and can never receive an incident."
            )
        return self

    def sampling_overrides(self) -> dict[str, Any]:
        """Return any per-knob sampling overrides that displace the model profile.

        Returns:
            Knob name to overriding value, empty when the profile is used as written.
        """
        return self.model.overrides.active()

    def snapshot(self) -> dict[str, Any]:
        """Produce the redacted configuration record written at the start of a run.

        This is what makes a campaign reconstructible. Its absence is why a change to
        one sampling knob went unnoticed across weeks of runs whose results were, from
        then on, not comparable with what came before.

        Returns:
            A JSON-safe mapping with every secret rendered as a mask.
        """
        return {
            "settings": self.model_dump(mode="json"),
            "sampling_overrides": sorted(self.sampling_overrides()),
            "legacy_env_names": sorted(legacy_names_in_use()),
        }


class AliasedField(NamedTuple):
    """One configurable field and every environment name that reaches it."""

    modern: str
    legacy: tuple[str, ...]
    path: tuple[str, ...]
    info: FieldInfo


def _iter_aliases(
    model: type[BaseSettings], prefix: tuple[str, ...] = ()
) -> Iterator[AliasedField]:
    """Walk a settings model, yielding every field that declares environment names.

    Args:
        model: A settings class.
        prefix: Attribute path accumulated from enclosing groups.

    Yields:
        One record per aliased field, deepest groups included.
    """
    for name, info in model.model_fields.items():
        annotation = info.annotation
        path = (*prefix, name)
        if isinstance(annotation, type) and issubclass(annotation, BaseSettings):
            yield from _iter_aliases(annotation, path)
            continue
        names = _alias_names(info)
        if names:
            yield AliasedField(names[0], tuple(names[1:]), path, info)


def known_env_names() -> set[str]:
    """Every environment variable the settings model recognises.

    Returns:
        Modern and legacy names together.
    """
    names: set[str] = set()
    for field in _iter_aliases(Settings):
        names.add(field.modern)
        names.update(field.legacy)
    return names


def unknown_prefixed_names(env: Mapping[str, str] | None = None) -> list[str]:
    """Find KIRA_-prefixed variables the settings model does not recognise.

    Pydantic only looks up the names it knows, so a misspelled variable is otherwise
    indistinguishable from an unset one: the process starts on defaults and behaves
    differently from what the operator asked for, silently. This is the check that
    turns that into a startup failure.

    Args:
        env: Environment to inspect. Defaults to the process environment.

    Returns:
        Unrecognised prefixed names, sorted.
    """
    environ = os.environ if env is None else env
    known = known_env_names()
    return sorted(
        n
        for n in environ
        # An adapter's own slice is checked by the adapter, at startup, against its own
        # model. Checking it here would mean the settings layer holding the field names
        # of every environment it exists not to know about.
        if n.startswith(ENV_PREFIX) and n not in known and not n.startswith(_ADAPTER_SLICE_PREFIX)
    )


def _alias_names(info: FieldInfo) -> list[str]:
    """Extract the accepted environment names from a field.

    Args:
        info: The field's metadata.

    Returns:
        Accepted names in precedence order, empty when the field declares none.
    """
    choices = info.validation_alias
    if isinstance(choices, AliasChoices):
        return [c for c in choices.choices if isinstance(c, str)]
    if isinstance(choices, str):
        return [choices]
    return []


def legacy_names_in_use(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Report which prototype environment names are set, and what replaced them.

    The map is derived from the field aliases themselves rather than maintained by
    hand, so it cannot drift from what the code actually accepts.

    Args:
        env: Environment to inspect. Defaults to the process environment.

    Returns:
        Legacy name to the modern name that supersedes it.
    """
    environ = os.environ if env is None else env
    found: dict[str, str] = {}
    for field in _iter_aliases(Settings):
        for name in field.legacy:
            if name in environ and not name.startswith(ENV_PREFIX):
                found[name] = field.modern
    return found


def load_settings() -> Settings:
    """Build settings from the environment, validating everything.

    Returns:
        A frozen settings object.

    Raises:
        pydantic.ValidationError: When any value is missing, malformed, or
            inconsistent with another group.
    """
    if unknown := unknown_prefixed_names():
        known = known_env_names()
        hints = {n: _closest(n, known) for n in unknown}
        detail = "; ".join(
            f"{n}" + (f" (did you mean {hint}?)" if hint else "") for n, hint in hints.items()
        )
        raise ValueError(
            f"unrecognised {ENV_PREFIX}-prefixed environment variable(s): {detail}. "
            "A misspelled name is indistinguishable from an unset one, so this is a "
            "startup failure rather than a silent default."
        )
    return Settings()


def _closest(name: str, known: set[str]) -> str | None:
    """Suggest the recognised name most similar to a misspelling.

    Args:
        name: The unrecognised variable.
        known: Every recognised name.

    Returns:
        The nearest match, or None when nothing is close.
    """
    matches = get_close_matches(name, sorted(known), n=1, cutoff=0.85)
    return matches[0] if matches else None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings, building them on first use.

    Returns:
        The frozen settings object.
    """
    return load_settings()


def reset_settings() -> None:
    """Discard the cached settings so the next call rebuilds them.

    For tests. Production code receives settings by injection and never resets them.
    """
    get_settings.cache_clear()
