"""Shared configuration machinery for the settings groups.

Every group is its own ``BaseSettings`` subclass with an explicit prefix, rather than
plain models nested under one root. That costs some verbosity and buys predictability:
a field's environment name is written out in full at its declaration, which matters
because ``pydantic-settings`` stops applying ``env_prefix`` to any field that declares
a ``validation_alias``. Groups need aliases to keep the prototype's flat names working,
so the prefix would silently stop applying to exactly the fields most likely to be
misread.

Settings are frozen once loaded. The prototype re-read ``os.environ`` on every access,
which made a running process's behaviour depend on when a value was read. Nothing used
that, a pod restarts on a ConfigMap change anyway, and for a system whose results have
to be reconstructible it is a liability rather than a feature.
"""

import json
from typing import Annotated, Any

from pydantic import AliasChoices, BeforeValidator, Field
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

__all__ = [
    "ENV_PREFIX",
    "NameSet",
    "SettingsGroup",
    "alias",
    "group_config",
    "setting",
]


def _split_names(value: object) -> object:
    """Parse a set of names from either a comma-separated string or a JSON array.

    The prototype spelled these as comma lists and existing deployments still export
    them that way. Pydantic-settings JSON-decodes collection-typed values before
    validation, which rejects a comma list outright, so these fields carry ``NoDecode``
    and parse here instead. Without it the process dies at startup on a value that
    worked for a year.

    Args:
        value: A raw environment value.

    Returns:
        A list of names, or the value unchanged when it is not a string.

    Raises:
        ValueError: When a bracketed value is not a valid JSON array of strings.
    """
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return []
    if text.startswith("["):
        parsed = json.loads(text)
        if not isinstance(parsed, list):
            raise ValueError(f"expected a JSON array of names, got {type(parsed).__name__}")
        return [str(item).strip() for item in parsed]
    return [part.strip() for part in text.split(",") if part.strip()]


NameSet = Annotated[frozenset[str], NoDecode, BeforeValidator(_split_names)]
"""A set of names accepting either a comma-separated string or a JSON array."""

ENV_PREFIX = "KIRA_"
"""Namespace for every modern setting name. Legacy flat names have no prefix."""

_BASE = SettingsConfigDict(
    env_nested_delimiter="__",
    env_file=(".env",),
    env_file_encoding="utf-8",
    # Nested groups share a prefix hierarchy, so a parent would see a child's
    # variables as extra keys. Typo protection instead comes from an explicit sweep of
    # every KIRA_-prefixed name against the known set, which is stronger anyway:
    # pydantic only looks up names it recognises, so "forbid" never saw a misspelling
    # of a name no field declares.
    extra="ignore",
    case_sensitive=False,
    frozen=True,
    validate_default=True,
)


def group_config(name: str) -> SettingsConfigDict:
    """Build the model config for one settings group.

    Args:
        name: The group's name, lowercase, e.g. ``memory``.

    Returns:
        A settings config whose environment prefix is ``KIRA_<NAME>__``.
    """
    return _BASE | SettingsConfigDict(env_prefix=f"{ENV_PREFIX}{name.upper()}__")


class SettingsGroup(BaseSettings):
    """Base for a group of related settings.

    Subclasses set ``model_config = group_config("<name>")`` and declare fields with a
    ``validation_alias`` naming the modern environment variable first and any legacy
    name after it.
    """

    model_config = _BASE

    def redacted(self) -> dict[str, Any]:
        """Dump the group with secret values replaced by a marker.

        Used by the startup config snapshot and by ``kira config show``, both of which
        are read by people and written to logs.

        Returns:
            A JSON-safe mapping with secrets rendered as ``"**********"``.
        """
        return self.model_dump(mode="json")


def alias(modern: str, *legacy: str) -> list[str]:
    """Spell out a field's accepted environment names, modern first.

    The modern name must be written in full: once a field declares a validation alias,
    ``pydantic-settings`` no longer applies the group's ``env_prefix`` to it, so
    omitting it here silently removes the documented name. A parametrised test asserts
    that both spellings resolve for every aliased field.

    Args:
        modern: The prefixed name, e.g. ``KIRA_MEMORY__ENABLED``.
        *legacy: Flat names the prototype used, still honoured.

    Returns:
        The accepted names in precedence order.
    """
    return [modern, *legacy]


def setting(default: Any, modern: str, *legacy: str, description: str) -> Any:
    """Declare a field, naming the environment variables it answers to.

    Args:
        default: The value used when nothing is set.
        modern: The prefixed environment name.
        *legacy: Flat names the prototype used.
        description: What the setting controls.

    Returns:
        A configured ``Field``.
    """
    return Field(
        default=default,
        validation_alias=AliasChoices(*alias(modern, *legacy)),
        description=description,
    )
