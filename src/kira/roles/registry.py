"""Where roles declare themselves, and where the pipeline's wiring is checked.

Same shape as the tool registry, for the same reason: every shipped role is findable in
one place, and a third party adds one through an entry point without forking. And the
same discipline about when things fail — a role whose submit tool is missing from its own
allowlist cannot terminate, which surfaces as a run that burns its whole turn budget and
reports a timeout. That is a startup error here instead.
"""

import logging
from collections.abc import Mapping
from importlib.metadata import entry_points
from pathlib import Path

from kira.prompts.loader import (
    FILLABLE_MARKERS,
    PromptError,
    markers,
    override_for,
    read_asset,
)
from kira.roles.base import RoleSpec
from kira.tools.allowlists import ROLE_TOOLSETS
from kira.tools.registry import registered

__all__ = [
    "RoleRegistryError",
    "get_role",
    "load_plugin_roles",
    "registered_roles",
    "role",
    "validate_roles",
]

_log = logging.getLogger(__name__)

_ROLES: dict[str, RoleSpec] = {}


class RoleRegistryError(RuntimeError):
    """A role declaration is inconsistent with the tools or prompts it names."""


def role(spec: RoleSpec) -> RoleSpec:
    """Register a role.

    Args:
        spec: The declaration.

    Returns:
        The same spec, so a module can assign it.

    Raises:
        RoleRegistryError: When the name is already taken. A silent overwrite would make
            an audit record name a role that did not run.
    """
    if spec.name in _ROLES:
        raise RoleRegistryError(f"role {spec.name!r} is already registered")
    _ROLES[spec.name] = spec
    return spec


def get_role(name: str) -> RoleSpec:
    """Look up one role.

    Args:
        name: The role name.

    Returns:
        Its declaration.

    Raises:
        RoleRegistryError: When no such role is registered.
    """
    try:
        return _ROLES[name]
    except KeyError:
        raise RoleRegistryError(f"no role registered as {name!r}") from None


def registered_roles() -> Mapping[str, RoleSpec]:
    """Every role declared so far.

    Returns:
        A read-only view of the registry.
    """
    return dict(_ROLES)


def clear_roles() -> None:
    """Empty the registry. For tests that register throwaway roles."""
    _ROLES.clear()


def load_plugin_roles(group: str = "kira.roles") -> list[str]:
    """Import every distribution that registers roles.

    Args:
        group: The entry-point group to load.

    Returns:
        Names of the distributions that loaded.

    Raises:
        RoleRegistryError: When an entry point re-registers an existing name.
    """
    loaded: list[str] = []
    for point in entry_points(group=group):
        try:
            point.load()
        except RoleRegistryError:
            raise
        except Exception as exc:  # one bad package must not stop startup
            _log.error("role entry point %s failed to import and was skipped: %s", point.name, exc)
            continue
        loaded.append(point.name)
    return loaded


def validate_roles(
    *, roles: Mapping[str, RoleSpec] | None = None, override_dir: Path | None = None
) -> None:
    """Check every way a role's wiring can be inconsistent with its tools or prompt.

    Args:
        roles: The roles to check. Defaults to the live registry.
        override_dir: A deployment's own prompts, checked here rather than at the
            incident that first uses one. An override declaring a marker nothing fills
            is the same failure as a packaged prompt doing so, and it must fail in the
            same place.

    Raises:
        RoleRegistryError: When a role cannot terminate, cannot be prompted, or names a
            submission schema its submit tool does not register.
    """
    specs = dict(roles if roles is not None else _ROLES)
    tools = registered()

    for name, spec in specs.items():
        submit = tools.get(spec.submit_tool)
        if submit is None:
            raise RoleRegistryError(
                f"role {name!r} names unregistered submit tool {spec.submit_tool!r}"
            )
        if submit.safety != "submit":
            raise RoleRegistryError(
                f"role {name!r} names {spec.submit_tool!r} as its submit tool, "
                f"but that tool is {submit.safety!r}"
            )

        allowed = ROLE_TOOLSETS.get(name)
        if allowed is None:
            raise RoleRegistryError(f"role {name!r} has no tool allowlist")
        if spec.submit_tool not in allowed:
            # The loop ends only through this tool. Absent from the allowlist it is
            # never offered, so the run cannot terminate: it exhausts its turns and
            # reports a timeout that reads as the model's failure rather than ours.
            raise RoleRegistryError(
                f"role {name!r} cannot terminate: its submit tool {spec.submit_tool!r} "
                "is not in its own allowlist"
            )

        try:
            override = override_for(spec.prompt, override_dir)
            template = read_asset(f"roles/{spec.prompt}.txt") if override is None else override
        except PromptError as exc:
            raise RoleRegistryError(f"role {name!r} has no usable prompt: {exc}") from exc
        unfillable = markers(template) - FILLABLE_MARKERS
        if unfillable:
            raise RoleRegistryError(
                f"role {name!r} prompt declares markers nothing fills: {sorted(unfillable)}"
            )
