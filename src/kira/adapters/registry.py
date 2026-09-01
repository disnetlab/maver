"""How an environment adapter is found, and how one is chosen.

Core-importable, and it names no adapter. An adapter is a factory taking the resolved
settings and returning a fresh :class:`~kira.adapters.base.EnvironmentAdapter` per
episode. In-tree adapters register with the decorator below; out-of-tree ones declare an
entry point in the ``kira.adapters`` group and need no fork and no core diff to be
selectable.

The factory takes settings rather than reaching for them. An adapter validates its own
slice of ``Settings.adapters`` with its own model, which is what keeps the field names
of an environment out of the core and still makes a typo inside that slice a startup
failure rather than a default nobody asked for.

Discovery and selection are kept apart on purpose. Importing an adapter's package must
never turn it on, or a package installed for one experiment would silently change what
every other run does; enabling one is a settings decision, and an adapter that cannot be
loaded is reported rather than quietly absent.
"""

import logging
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib.metadata import entry_points

from kira.adapters.base import EnvironmentAdapter
from kira.settings.root import Settings

__all__ = [
    "AdapterFactory",
    "adapter",
    "available_adapters",
    "clear_adapters",
    "get_adapter",
    "legacy_selection",
    "load_entry_point_adapters",
    "registered_adapters",
]

_log = logging.getLogger(__name__)

AdapterFactory = Callable[[Settings], EnvironmentAdapter]

_READY = "ready"


@dataclass(frozen=True, slots=True)
class _Registration:
    """One adapter, as the registry holds it."""

    name: str
    factory: AdapterFactory
    legacy_switch: str | None


_REGISTRY: dict[str, _Registration] = {}
_UNAVAILABLE: dict[str, str] = {}


def adapter(
    name: str, *, legacy_switch: str | None = None
) -> Callable[[AdapterFactory], AdapterFactory]:
    """Register a factory under a name.

    Args:
        name: The name a deployment enables the adapter by.
        legacy_switch: A prototype environment variable that used to mean "this process
            runs under this environment". Declared by the adapter rather than known to
            the settings layer, so a deployment file written for the prototype keeps
            working without the core naming an environment.

    Returns:
        A decorator that registers and returns the factory.

    Raises:
        ValueError: When the name is already taken. A silent overwrite would make a run
            report one adapter and use another.
    """

    def register(factory: AdapterFactory) -> AdapterFactory:
        if name in _REGISTRY:
            raise ValueError(f"environment adapter {name!r} is already registered")
        _REGISTRY[name] = _Registration(name=name, factory=factory, legacy_switch=legacy_switch)
        return factory

    return register


def load_entry_point_adapters(group: str = "kira.adapters") -> None:
    """Import every distribution registering in the adapter entry-point group.

    Importing the module runs its registration. A module that raises is recorded with
    the exception text and skipped: a missing optional extra, or one broken third-party
    package, must not stop an incident service from starting. Selecting an adapter that
    could not be loaded is fatal, and says why.

    Args:
        group: The entry-point group to load.
    """
    for point in entry_points(group=group):
        try:
            point.load()
        except Exception as exc:
            _UNAVAILABLE[point.name] = f"{type(exc).__name__}: {exc}"
            _log.warning("environment adapter %s could not be loaded: %s", point.name, exc)


def get_adapter(name: str) -> AdapterFactory:
    """Look up one adapter's factory.

    Args:
        name: The registered name.

    Returns:
        The factory.

    Raises:
        ValueError: When no such adapter is available, naming the import failure when
            there was one. A deployment that asked for an environment and did not get
            it would otherwise run against the default transport and grade nothing.
    """
    registration = _REGISTRY.get(name)
    if registration is not None:
        return registration.factory
    reason = _UNAVAILABLE.get(name)
    if reason is not None:
        raise ValueError(f"environment adapter {name!r} is installed but failed to load: {reason}")
    known = ", ".join(sorted(_REGISTRY)) or "none"
    raise ValueError(f"unknown environment adapter {name!r}; registered: {known}")


def registered_adapters() -> Mapping[str, AdapterFactory]:
    """Every adapter that loaded.

    Returns:
        Name to factory.
    """
    return {name: reg.factory for name, reg in _REGISTRY.items()}


def available_adapters() -> Mapping[str, str]:
    """Report what an operator can enable, and why anything else cannot be.

    Returns:
        Name to ``ready`` or the reason it could not be loaded.
    """
    return {**dict.fromkeys(_REGISTRY, _READY), **_UNAVAILABLE}


def legacy_selection(env: Mapping[str, str] | None = None) -> frozenset[str]:
    """Read the adapters a prototype-era deployment file switches on.

    Consulted only when the variable is actually present, so an unset name never
    enables anything: the prototype's switch defaulted to on, and inheriting that
    default would turn a plain install into a benchmark client.

    Args:
        env: Environment to inspect. Defaults to the process environment.

    Returns:
        The adapter names those legacy switches select.
    """
    environ = os.environ if env is None else env
    selected: set[str] = set()
    for registration in _REGISTRY.values():
        switch = registration.legacy_switch
        if switch is None or switch not in environ:
            continue
        if environ[switch].strip().lower() in {"1", "true", "yes", "on"}:
            selected.add(registration.name)
            _log.info(
                "legacy name in use: %s -> KIRA_INTAKE__ADAPTERS=%s", switch, registration.name
            )
    return frozenset(selected)


def clear_adapters() -> None:
    """Empty the registry. For tests that register throwaway adapters."""
    _REGISTRY.clear()
    _UNAVAILABLE.clear()
