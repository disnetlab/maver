"""Where a tool declares itself, and where the rules about tools are enforced.

The prototype kept a 600-line literal dict of schemas in one module and the functions
that implement them in eleven others, so adding a tool meant editing two files hundreds
of lines apart and the pairing was checked only at import. Here a tool is declared where
it is written, with a decorator, and grepping for the decorator across ``src/kira/tools``
lists every tool in the system.

Everything a misconfiguration could do quietly is made loud at import instead. A tool
with no description, a description written for a role that is not allowed the tool, a
requirement naming a capability no transport can offer: each of these leaves the model
either uninstructed or short a tool, and none of them raises anything at incident time.
"""

import logging
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from importlib.resources import files
from typing import Any, Literal

from kira.tools.context import ToolContext
from kira.tools.policy import policy_for
from kira.tools.schema import RATIONALE_ARG, strip_rationale, with_rationale
from kira.tools.transport.base import CAPABILITIES

__all__ = [
    "RETRIEVAL_PROVENANCE",
    "Authority",
    "BoundTool",
    "ToolOutput",
    "ToolRegistryError",
    "ToolSafety",
    "ToolSpec",
    "load_entry_point_tools",
    "registered",
    "tool",
    "validate_registry",
]

_log = logging.getLogger(__name__)

ToolSafety = Literal["read_only", "local_exec", "cluster_write", "submit"]
"""What a tool can do, independent of who may call it.

``read_only`` queries telemetry, cluster state or the experience store.
``local_exec`` runs code in the session sandbox and holds no cluster credentials.
``cluster_write`` can mutate a cluster or a host.
``submit`` is a terminal validated submission and is never dispatched.
"""

Authority = Literal["read", "write", "exec", "blocked"]
"""What one specific call asks for, computed from its arguments.

The approval gate decides on this, not on :data:`ToolSafety`, because a single kubectl
tool serves both ``get pods`` and ``delete deployment``. ``exec`` is its own class rather
than a kind of write: diagnostic roles use ``kubectl exec`` for read-only inspection, and
folding it into ``write`` would change behaviour every published measurement rests on.
"""

RETRIEVAL_PROVENANCE = "retrieval"
"""The key a retrieval tool's provenance travels under on ``ToolResult.metadata``.

One name for one concept, all the way through: ``ToolOutput.retrieval`` is what the tool
attaches, this is where the loop puts it, and ``ToolCallEvent.retrieval`` is where it
lands in the log.
"""


@dataclass(frozen=True, slots=True)
class ToolOutput:
    """A tool's result, and what the audit log should know about where it came from.

    A tool returns text, and text is all the model may see. Which stored records a
    retrieval matched, and under which handle each was shown, is not for the model to
    read back — it is what an analysis of what a role was actually served needs, months
    later, and it is otherwise lost the moment the call returns.

    Returning both keeps the audit half off the transcript without a context variable set
    beside the call, which is how the prototype did it and why a tool could not be called
    in a test without a fixture. ``str()`` on this is the text, so a caller that wants
    only the result is unaffected by a tool that started attaching provenance.
    """

    text: str
    retrieval: tuple[dict[str, Any], ...] = ()
    """Per record: its ``record_id``, the ``handle`` shown for it, and its ``score``."""

    def __str__(self) -> str:
        """Render the model-facing half.

        Returns:
            The text the model reads.
        """
        return self.text


class ToolRegistryError(RuntimeError):
    """A tool declaration or its description is inconsistent."""


@dataclass(frozen=True)
class ToolSpec:
    """One registered tool."""

    name: str
    parameters: Mapping[str, Any]
    safety: ToolSafety
    dispatch: Any = None
    """The implementation. None only for ``safety="submit"``."""
    requires: frozenset[str] = field(default_factory=frozenset)
    """Transport capabilities without which this tool is not offered at all.

    A gap is a typed fact recorded before the run, not an error the model discovers
    mid-incident and tries to work around.
    """
    descriptions: Mapping[str, Any] | None = None
    """Descriptions for an out-of-tree tool. Built-ins use the shipped YAML instead."""


_REGISTRY: dict[str, ToolSpec] = {}


def tool(
    *,
    name: str,
    safety: ToolSafety,
    parameters: Mapping[str, Any],
    requires: Iterable[str] = (),
    descriptions: Mapping[str, Any] | None = None,
) -> Callable[[Any], Any]:
    """Register a tool, returning the function unchanged.

    The function stays an ordinary coroutine so it can be unit-tested without the
    registry, the transport, or a role.

    Args:
        name: The name the model calls.
        safety: What the tool can do.
        parameters: Its JSON Schema.
        requires: Transport capabilities the tool needs.
        descriptions: Only for tools shipped outside this package.

    Returns:
        A decorator that registers and returns the function.

    Raises:
        ToolRegistryError: When the name is already taken.
    """

    def register(fn: Any) -> Any:
        if name in _REGISTRY:
            raise ToolRegistryError(f"tool {name!r} is already registered")
        _REGISTRY[name] = ToolSpec(
            name=name,
            parameters=parameters,
            safety=safety,
            dispatch=fn,
            requires=frozenset(requires),
            descriptions=descriptions,
        )
        return fn

    return register


def register_submit_tool(
    *, name: str, parameters: Mapping[str, Any], descriptions: Mapping[str, Any] | None = None
) -> ToolSpec:
    """Register a terminal submission schema, which has no implementation.

    Args:
        name: The name the model calls.
        parameters: The normalised schema.
        descriptions: Only for tools shipped outside this package.

    Returns:
        The registered spec.

    Raises:
        ToolRegistryError: When the name is already taken.
    """
    if name in _REGISTRY:
        raise ToolRegistryError(f"tool {name!r} is already registered")
    spec = ToolSpec(
        name=name, parameters=parameters, safety="submit", dispatch=None, descriptions=descriptions
    )
    _REGISTRY[name] = spec
    return spec


def load_descriptions() -> dict[str, Any]:
    """Read the shipped description catalogue.

    Loaded through the package rather than by filesystem path, so it keeps working from
    an installed wheel and from a zip import.

    Returns:
        Tool name to description entry.

    Raises:
        ToolRegistryError: When the file is not a mapping.
    """
    import yaml  # noqa: PLC0415 - kept off the import path of every module

    text = files("kira.tools").joinpath("descriptions.yaml").read_text(encoding="utf-8")
    loaded = yaml.safe_load(text)
    if not isinstance(loaded, dict):
        raise ToolRegistryError("descriptions.yaml must be a mapping of tool name to entry")
    return loaded


def registered() -> Mapping[str, ToolSpec]:
    """Every tool declared so far.

    Returns:
        A read-only view of the registry.
    """
    return dict(_REGISTRY)


def clear_registry() -> None:
    """Empty the registry. For tests that register throwaway tools."""
    _REGISTRY.clear()


def load_entry_point_tools(group: str = "kira.tools") -> list[str]:
    """Import every distribution that registers tools, running their decorators.

    A name that collides with a built-in is a hard error: a third party silently
    replacing a tool would make an audit record say one thing and mean another. A module
    that fails to import is logged and skipped, because one broken optional package must
    not stop an incident pipeline from starting.

    Args:
        group: The entry-point group to load.

    Returns:
        Names of the distributions that loaded successfully.

    Raises:
        ToolRegistryError: When an entry point re-registers an existing name.
    """
    loaded: list[str] = []
    for point in entry_points(group=group):
        try:
            point.load()
        except ToolRegistryError:
            raise
        except Exception as exc:  # one bad package must not stop startup
            _log.error("tool entry point %s failed to import and was skipped: %s", point.name, exc)
            continue
        loaded.append(point.name)
    return loaded


@dataclass(frozen=True)
class BoundTool:
    """A registered tool bound to one run's context."""

    spec: ToolSpec
    ctx: ToolContext

    async def __call__(self, **arguments: Any) -> str | ToolOutput:
        """Dispatch the call.

        Args:
            **arguments: The model's arguments, rationale included.

        Returns:
            What the tool returned: its text, or its text with the provenance behind it.

        Raises:
            ToolRegistryError: When the spec has no implementation.
        """
        if self.spec.dispatch is None:
            raise ToolRegistryError(f"tool {self.spec.name!r} has no implementation")
        returned = await self.spec.dispatch(self.ctx, **strip_rationale(arguments))
        return returned if isinstance(returned, ToolOutput) else str(returned)


def definition_for(spec: ToolSpec) -> dict[str, Any]:
    """Build the model-facing parameter schema for a tool.

    Args:
        spec: The registered tool.

    Returns:
        Its schema, with the rationale prepended unless it is a submission.
    """
    if spec.safety == "submit":
        return dict(spec.parameters)
    return with_rationale(spec.parameters)


def validate_registry(
    *,
    descriptions: Mapping[str, Any],
    allowlists: Mapping[str, tuple[str, ...]],
    registry: Mapping[str, ToolSpec] | None = None,
) -> None:
    """Check every rule that would otherwise fail quietly during an incident.

    Args:
        descriptions: The description catalogue, keyed by tool name.
        allowlists: Tool names each role may call, keyed by role.
        registry: The registry to check. Defaults to the live one.

    Raises:
        ToolRegistryError: When any rule is violated, naming what and where.
    """
    specs = dict(registry if registry is not None else _REGISTRY)

    missing = sorted(set(specs) - set(descriptions))
    undeclared = sorted(set(descriptions) - set(specs))
    if missing or undeclared:
        raise ToolRegistryError(
            "registry and descriptions must match exactly; "
            f"undescribed tools: {missing}; described but unregistered: {undeclared}"
        )

    for name, entry in descriptions.items():
        if not isinstance(entry, dict):
            raise ToolRegistryError(f"description for {name!r} must be a mapping")
        # A typo'd key is why this rule exists: the prototype wrote a role name as a
        # sibling of `default` instead of nesting it under `agents`, nothing rejected
        # it, and that role was served the generic text for the life of the system.
        unknown = sorted(set(entry) - {"default", "agents"})
        if unknown:
            raise ToolRegistryError(
                f"description for {name!r} has unknown keys {unknown}; "
                "a per-role description belongs under 'agents'"
            )
        agents = entry.get("agents") or {}
        if not isinstance(agents, dict):
            raise ToolRegistryError(f"tool {name!r} agents descriptions must be a mapping")
        allowed_roles = {role for role, names in allowlists.items() if name in names}
        for role in agents:
            if role not in allowed_roles:
                raise ToolRegistryError(
                    f"tool {name!r} has a description for role {role!r}, "
                    "which is not allowed to call it"
                )

    for role, names in allowlists.items():
        for name in names:
            spec = specs.get(name)
            if spec is None:
                raise ToolRegistryError(f"role {role!r} allows unregistered tool {name!r}")
            if spec.safety != "submit" and spec.dispatch is None:
                raise ToolRegistryError(f"tool {name!r} is allowlisted but has no implementation")
            description_for(name, role, descriptions=descriptions)

    for name, spec in specs.items():
        unknown_caps = sorted(spec.requires - CAPABILITIES)
        if unknown_caps:
            raise ToolRegistryError(
                f"tool {name!r} requires unknown transport capabilities {unknown_caps}"
            )
        # The gate reads the policy registry, not the spec, so this is the registration
        # that has to exist. A writer with no policy is treated as a write by every
        # stage and permitted to none — refused at incident time, in the stage that was
        # supposed to be allowed, having passed every check at import.
        if spec.safety == "cluster_write":
            policy = policy_for(name)
            if policy is None or not policy.write_stages:
                raise ToolRegistryError(
                    f"tool {name!r} can write and must register a policy "
                    "naming the stages that may issue one"
                )
        if spec.safety == "submit" and spec.dispatch is not None:
            raise ToolRegistryError(f"submit tool {name!r} must not have an implementation")
        if spec.safety != "submit" and RATIONALE_ARG in (spec.parameters.get("properties") or {}):
            raise ToolRegistryError(
                f"tool {name!r} declares {RATIONALE_ARG!r} itself; it is injected"
            )


def description_for(name: str, role: str, *, descriptions: Mapping[str, Any]) -> str:
    """Resolve the description one role sees for one tool.

    A per-role entry replaces the default outright rather than extending it, which is
    easy to misread and worth stating: a role-specific description must be complete.

    Args:
        name: The tool.
        role: The role about to be offered it.
        descriptions: The description catalogue.

    Returns:
        The description to send.

    Raises:
        ToolRegistryError: When no usable description exists.
    """
    raw = descriptions.get(name)
    if not isinstance(raw, dict):
        raise ToolRegistryError(f"missing description entry for tool {name!r}")
    agents = raw.get("agents") or {}
    text = agents.get(role) or raw.get("default")
    if not isinstance(text, str) or not text.strip():
        raise ToolRegistryError(
            f"missing non-empty description for tool {name!r} and role {role!r}"
        )
    return text.strip()
