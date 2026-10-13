"""The root settings object, its validation, and the run's configuration snapshot."""

import logging
import os
import re
from collections.abc import Iterator, Mapping
from difflib import get_close_matches
from functools import lru_cache
from pathlib import Path
from typing import Any, NamedTuple

from pydantic import AliasChoices, Field, model_validator
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

from maver.prompts.loader import overridden_roles
from maver.settings.arms import Arm, arm_digest, load_arm
from maver.settings.authority import AuthoritySettings
from maver.settings.base import ENV_PREFIX, SettingsGroup, setting
from maver.settings.groups import (
    IntakeSettings,
    MemorySettings,
    ModelSettings,
    ObservabilitySettings,
    PipelineSettings,
    PromptSettings,
    RuntimeSettings,
    ServerSettings,
    ToolSettings,
    TransportSettings,
    VerifySettings,
)

ADAPTER_ENV_NAME = f"{ENV_PREFIX}ADAPTERS"
"""Where an adapter's configuration slice is read from."""

_ADAPTER_SLICE_PREFIX = f"{ADAPTER_ENV_NAME}__"

_SECRET_FIELD_NAME = re.compile(r"(^|_)(key|token|secret|password|dsn)(_|$)", re.IGNORECASE)
"""What a credential is called, for the one configuration slice with no type to read.

An adapter's slice is untyped on purpose, so nothing there is a ``SecretStr`` and the
settings layer has only the field's name to go on. Judging by the name is what an
adapter's own model would do one layer later, and this record is written into every run
and echoed by a command whose contract is that it is safe to paste.
"""


_DIRECT_ENV_NAMES = frozenset({"REVISION"})
"""Prefixed names read straight from the environment rather than through a field.

The image writes the commit it was built from into ``MAVER_REVISION`` so that
``/healthz`` can report it and a deploy can tell the container it just built from a
stale one nobody recreated. It answers to no setting, so the sweep below would read it
as a misspelling and refuse to start the very container that sets it.
"""

__all__ = [
    "ADAPTER_ENV_NAME",
    "AliasedField",
    "ArmOverrides",
    "ArmYamlSource",
    "Settings",
    "arm_overrides",
    "get_settings",
    "known_env_names",
    "legacy_names_in_use",
    "load_settings",
    "reset_settings",
    "resolved_arm",
    "unknown_prefixed_names",
]


_log = logging.getLogger(__name__)


class Settings(SettingsGroup):
    """Everything the process is configured with.

    Groups are independent ``BaseSettings`` instances, so each reads its own
    environment names. The object is frozen: settings are injected into the components
    that need them, never re-read, so a run's behaviour cannot change underneath it.
    """

    # The prefix is declared here rather than inherited, because without it this class
    # answers to the bare group names as well: a plain ``MODEL`` in the environment
    # aborted startup while pydantic tried to parse it as a group, and ``MEMORY__TOP_K``
    # was a second live vocabulary that the misspelling sweep, which only looks at
    # prefixed names, could not see and the run's record never mentioned.
    model_config = SettingsGroup.model_config | SettingsConfigDict(env_prefix=ENV_PREFIX)

    arm: str | None = setting(
        None,
        f"{ENV_PREFIX}ARM",
        description=(
            "The experiment arm this process runs, read from <arm_dir>/<name>.yaml. "
            "Unset means no overlay: field defaults plus whatever the environment says."
        ),
    )
    arm_dir: Path = setting(
        Path("configs/arms"),
        f"{ENV_PREFIX}ARM_DIR",
        description="Where arm files live. Baked into the image alongside the profiles.",
    )

    customisation: Path | None = setting(
        None,
        f"{ENV_PREFIX}CUSTOMISATION",
        description=(
            "Optional YAML file configuring named agents, pipelines and the default pipeline. "
            "Python implementations are registered in maver/custom.py."
        ),
    )

    pipeline: PipelineSettings = Field(default_factory=PipelineSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    verify: VerifySettings = Field(default_factory=VerifySettings)
    authority: AuthoritySettings = Field(default_factory=AuthoritySettings)
    intake: IntakeSettings = Field(default_factory=IntakeSettings)
    prompts: PromptSettings = Field(default_factory=PromptSettings)
    tools: ToolSettings = Field(default_factory=ToolSettings)
    transport: TransportSettings = Field(default_factory=TransportSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)
    server: ServerSettings = Field(default_factory=ServerSettings)
    adapters: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        validation_alias=ADAPTER_ENV_NAME,
        description=(
            "Per-adapter configuration, keyed by registered adapter name and read from "
            "MAVER_ADAPTERS__<NAME>__<FIELD>. Untyped here on purpose: an adapter "
            "validates its own slice with its own model, which is what keeps the core "
            "free of the field names of environments it must not know about."
        ),
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Put the arm file below every source an operator can reach.

        Args:
            settings_cls: The class being built.
            init_settings: Values passed to the constructor.
            env_settings: The modern prefixed names.
            dotenv_settings: A developer's ``.env``.
            file_secret_settings: The secrets directory.

        Returns:
            The sources, highest precedence first.
        """
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            file_secret_settings,
            ArmYamlSource(settings_cls),
        )

    @model_validator(mode="after")
    def _cross_group_invariants(self) -> "Settings":
        """Reject combinations that cannot work, at startup rather than mid-incident.

        Returns:
            The validated settings.

        Raises:
            ValueError: When two groups are configured inconsistently.
        """
        if not self.intake.http_enabled and not self.intake.kafka.enabled:
            raise ValueError("no intake is enabled: enable HTTP or Kafka intake")
        if self.server.environment == "production" and self.authority.mode == "auto":
            # The escape hatch is a rule, not a flag, and that is the point. A
            # deployment that wants routine scaling to proceed unattended can say so;
            # what it cannot do is hand a model unreviewed write access to a production
            # cluster by setting one global value. Refused rather than warned because
            # the whole approval mechanism is inert underneath it, and a warning about
            # an inert safety control is a thing an operator learns to scroll past.
            raise ValueError(
                "server.environment='production' with authority.mode='auto' would let "
                "every cluster write proceed unreviewed. Set authority.mode to 'prompt' "
                "or 'deny', and name what may proceed unattended with an authority rule "
                "(MAVER_AUTHORITY__RULES) scoped to those tools, stages or selectors."
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
        dumped = self.model_dump(mode="json")
        dumped["adapters"] = {
            adapter: {
                field: ("**********" if _SECRET_FIELD_NAME.search(field) else value)
                for field, value in slice_.items()
            }
            for adapter, slice_ in dumped["adapters"].items()
        }
        return {
            "arm_digest": self._arm_digest(),
            "settings": dumped,
            "sampling_overrides": sorted(self.sampling_overrides()),
            # Which prompts this deployment replaced, and what was in them. The
            # directory path alone does not say, and a result produced under a prompt
            # nobody recorded is not attributable.
            "prompt_overrides": overridden_roles(self.prompts.override_dir),
            "legacy_env_names": sorted(legacy_names_in_use()),
        }

    def _arm_digest(self) -> str | None:
        """Hash the arm file this configuration was resolved from.

        Returns:
            The file's SHA-256, or None when no arm is named or the file has since
            become unreadable. Recorded as absent rather than raised: the digest is
            provenance beside a name the dump already carries, and an incident must not
            fail to start over a file the process has already read.
        """
        if not self.arm:
            return None
        try:
            return arm_digest(self.arm, self.arm_dir)
        except OSError:
            return None


class ArmYamlSource(PydanticBaseSettingsSource):
    """The named arm's settings, pruned to what no higher-precedence source supplied.

    Placing a source last is not enough on its own. Every settings group is an
    independent ``BaseSettings`` that reads its own environment, so a nested mapping
    handed down by the parent arrives as that group's init keyword arguments — and init
    outranks the group's own environment source. An arm file would therefore beat
    ``MAVER_MEMORY__ENABLED`` and ``MEMORY_ENABLED`` alike, which is the one thing a
    deployment whose control surface is a ConfigMap cannot have, and the one thing a
    campaign whose run scripts export flat names cannot have either.

    So the overlay is pruned before it is returned. Each group the arm mentions is built
    once with no arguments, which runs that group's own chain, and every field it
    resolved for itself is dropped from the overlay. What is left is only the settings
    nothing above the arm had an opinion about.
    """

    def get_field_value(self, field: FieldInfo, field_name: str) -> tuple[Any, str, bool]:
        """Unused: the overlay is resolved as a whole, not field by field.

        Args:
            field: The field being resolved.
            field_name: Its name.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError

    def __call__(self) -> dict[str, Any]:
        """Read the arm named by the sources above this one.

        Returns:
            The pruned overlay, empty when no arm is named.

        Raises:
            FileNotFoundError: When the named arm does not exist.
            ValueError: When the arm names a setting nothing reads.
        """
        state = _by_field_name(self.settings_cls, self.current_state)
        name = state.get("arm")
        if not name:
            return {}
        directory = state.get("arm_dir") or self.settings_cls.model_fields["arm_dir"].default
        arm = load_arm(str(name), Path(str(directory)))
        return _prune(self.settings_cls, arm.overlay(), state, set(state))


def _by_field_name(model: type[BaseSettings], state: Mapping[str, Any]) -> dict[str, Any]:
    """Re-key what the sources above produced, from environment names to field names.

    A source reports an aliased field under whichever of its accepted names it matched,
    so the raw state is a mixture of the two vocabularies and neither one alone answers
    "did anything above set this?".

    Args:
        model: The settings class the state belongs to.
        state: What the higher-precedence sources produced.

    Returns:
        Field name to value, for every field something above supplied.
    """
    lowered = {name.lower(): value for name, value in state.items()}
    found: dict[str, Any] = {}
    for name, info in model.model_fields.items():
        for candidate in (name, *_alias_names(info)):
            if candidate.lower() in lowered:
                found[name] = lowered[candidate.lower()]
                break
    return found


def _prune(
    model: type[BaseSettings],
    overlay: Mapping[str, Any],
    supplied: Mapping[str, Any],
    claimed: set[str],
    path: str = "",
) -> dict[str, Any]:
    """Drop the overlay values something above the arm already decided.

    Args:
        model: The settings class this level of the overlay describes.
        overlay: The arm's values for it.
        supplied: What higher-precedence sources produced at this level.
        claimed: Field names those sources set.
        path: Dotted path accumulated from enclosing levels, for error messages.

    Returns:
        The surviving overlay.

    Raises:
        ValueError: When the arm names a field the model does not declare.
    """
    kept: dict[str, Any] = {}
    for key, value in overlay.items():
        info = model.model_fields.get(key)
        if info is None:
            raise ValueError(
                f"arm sets {path}{key}, which is not a setting. An arm file is what a "
                "published number is attributed to, so a key nothing reads is a claim "
                "about a configuration that never ran."
            )
        annotation = info.annotation
        if (
            isinstance(annotation, type)
            and issubclass(annotation, BaseSettings)
            and isinstance(value, Mapping)
        ):
            raw = supplied.get(key)
            # A source reports a nested name in the environment's own casing, so a
            # case-sensitive membership test below would miss what it supplied and the
            # arm would displace it.
            below = (
                {
                    str(supplied_name).lower(): below_value
                    for supplied_name, below_value in raw.items()
                }
                if isinstance(raw, Mapping)
                else {}
            )
            # Building the group runs its own chain: the modern name, the legacy names
            # it aliases, .env and the secrets directory. What it resolved for itself
            # is precisely what the arm must not displace.
            nested = _prune(
                annotation,
                value,
                below,
                set(annotation().model_fields_set) | set(below),
                f"{path}{key}.",
            )
            if nested:
                kept[key] = nested
        elif key not in claimed:
            kept[key] = value
    return kept


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
    """Find MAVER_-prefixed variables the settings model does not recognise.

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
        if n.startswith(ENV_PREFIX)
        and n not in known
        and n.removeprefix(ENV_PREFIX) not in _DIRECT_ENV_NAMES
        and not n.startswith(_ADAPTER_SLICE_PREFIX)
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


class ArmOverrides(NamedTuple):
    """Arm values the resolved configuration displaced, split by whether that is legal."""

    allowed: dict[str, tuple[Any, Any]]
    """Paths the environment may override. Reported, never refused."""
    broken: dict[str, tuple[Any, Any]]
    """Paths the arm claims to hold constant, which makes that claim false."""


def resolved_arm(settings: Settings) -> Arm | None:
    """Load the arm a configuration was resolved under.

    Args:
        settings: The resolved configuration.

    Returns:
        The arm, or None when none is named.

    Raises:
        FileNotFoundError: When the named arm has since disappeared.
    """
    return load_arm(settings.arm, settings.arm_dir) if settings.arm else None


def arm_overrides(settings: Settings) -> ArmOverrides:
    """Find where the resolved configuration departs from the arm it names.

    Args:
        settings: The resolved configuration.

    Returns:
        The departures, empty on both counts when no arm is named.
    """
    arm = resolved_arm(settings)
    if arm is None:
        return ArmOverrides({}, {})
    resolved = settings.model_dump(mode="json")
    # An arm's own name and directory are how it was selected, not settings it varies.
    baseline = _unconfigured_tree() | {k: resolved[k] for k in ("arm", "arm_dir")}
    found = arm.contradicted_by(resolved, baseline)
    broken = {path: pair for path, pair in found.items() if arm.is_held_constant(path)}

    # A sampling override displaces the profile without touching any path the arm
    # declares, so the comparison above cannot see it: an arm holding the model profile
    # constant would run at a different temperature and report no contradiction. That is
    # the exact shape of a knob that moved once here and went unnoticed for twelve days,
    # leaving the campaigns either side of it not comparable.
    if arm.is_held_constant("model_profile"):
        for knob, value in sorted(settings.sampling_overrides().items()):
            broken[f"model.overrides.{knob}"] = (f"the {arm.name} profile", value)
    return ArmOverrides({p: v for p, v in found.items() if p not in broken}, broken)


def _unconfigured_tree() -> dict[str, Any]:
    """Render the settings tree with nothing supplied, as the arm comparison's baseline.

    Built from the declared defaults rather than by constructing the groups, because
    constructing one runs that group's own environment chain and what this needs is
    precisely the tree an empty environment would produce.

    Returns:
        A JSON-safe mapping of every setting at its default.
    """

    def build(model: type[BaseSettings]) -> BaseSettings:
        groups: dict[str, Any] = {
            name: build(info.annotation)
            for name, info in model.model_fields.items()
            if isinstance(info.annotation, type) and issubclass(info.annotation, BaseSettings)
        }
        return model.model_construct(**groups)

    return build(Settings).model_dump(mode="json")


def load_settings() -> Settings:
    """Build settings from the environment, validating everything.

    Returns:
        A frozen settings object.

    Raises:
        ValueError: When a prefixed name is unrecognised, or when the environment
            overrides something the named arm claims to hold constant.
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
    settings = Settings()
    if broken := arm_overrides(settings).broken:
        detail = "; ".join(
            f"{path}: arm says {declared!r}, resolved to {actual!r}"
            for path, (declared, actual) in sorted(broken.items())
        )
        raise ValueError(
            f"the environment contradicts what arm {settings.arm!r} holds constant: "
            f"{detail}. Held constant is what licenses reading a gap between two arms "
            "as single-variable, so this is a startup failure rather than a run whose "
            "comparison quietly means something else."
        )
    _warn_if_no_node_can_be_recovered(settings)
    return settings


def _warn_if_no_node_can_be_recovered(settings: Settings) -> None:
    """Say out loud that this deployment cannot repair anything on a node.

    A warning rather than a refusal: most deployments have no worker they can reach over
    SSH, and for them declining node-level recovery is the correct configuration. The
    resolved value is in the run's configuration record either way, which is what lets a
    later reader tell one of those campaigns from the other.

    The tool is still offered, because every measured campaign offered it on an empty
    allowlist and what a role is shown is an input to the result. So the warning says
    what an operator will otherwise infer from a transcript: the calls are made and each
    one is refused.
    """
    if settings.tools.host_recovery_nodes:
        return
    _log.warning(
        "no host-recovery node is configured. ssh_exec is still offered to the "
        "remediating role, and every call it makes will be refused until nodes are "
        "configured, so a fault that can only be repaired on the node itself will end "
        "the incident unfixed. Set MAVER_TOOLS__HOST_RECOVERY_NODES to the SSH aliases "
        "of the nodes this deployment may act on."
    )


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
