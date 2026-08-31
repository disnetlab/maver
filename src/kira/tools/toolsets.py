"""Deciding which tools a role is actually offered, for one run.

The allowlist says what a role may call. What it *can* call also depends on what the
bound transport serves and what this configuration has: an environment with no log
backend cannot offer a log tool, and a configuration with no experience store cannot
offer retrieval.

Those tools are dropped **before the run**, with the reason recorded, rather than left in
place to fail when called. The difference matters to the model: a tool that errors looks
like a fault to work around, and it will try again, differently, and again. A tool that
was never offered costs nothing.

Recording what was dropped closes a real gap. In the prototype the resolved toolset was
implicit in a constant plus five environment reads, so reconstructing what a past run
actually offered the model meant reconstructing its environment. It is one field.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from kira.llm.types import ToolSpec as ModelToolSpec
from kira.tools.allowlists import ROLE_TOOLSETS
from kira.tools.context import ToolContext
from kira.tools.registry import (
    BoundTool,
    ToolRegistryError,
    definition_for,
    description_for,
    load_descriptions,
    registered,
)

__all__ = ["ResolvedToolset", "build_toolset"]

_MEMORY_TOOLS = frozenset({"memory_search", "open_memory", "session_trace_summary"})


@dataclass(frozen=True)
class ResolvedToolset:
    """What one role is offered in one run, and what it is not."""

    definitions: list[ModelToolSpec]
    """What the model sees, in the allowlist's hand-tuned order."""
    dispatch: dict[str, BoundTool]
    dropped: dict[str, str] = field(default_factory=dict)
    """Tool name to the reason it was withheld, written to the run's audit record."""


def build_toolset(
    role: str,
    ctx: ToolContext,
    *,
    descriptions: Mapping[str, object] | None = None,
    capabilities: Sequence[str] | None = None,
) -> ResolvedToolset:
    """Resolve a role's tools against this run's environment and configuration.

    Args:
        role: The role being run.
        ctx: The run's context, bound into every dispatch.
        descriptions: The description catalogue. Defaults to the shipped one.
        capabilities: What the transport serves. Defaults to asking the bound transport.

    Returns:
        The model-facing definitions, the dispatch map, and what was dropped.

    Raises:
        ToolRegistryError: When the role has no allowlist.
    """
    names = ROLE_TOOLSETS.get(role)
    if names is None:
        raise ToolRegistryError(f"no toolset defined for role {role!r}")

    catalogue = descriptions if descriptions is not None else load_descriptions()
    served = frozenset(capabilities if capabilities is not None else ctx.transport.capabilities)
    specs = registered()

    definitions: list[ModelToolSpec] = []
    dispatch: dict[str, BoundTool] = {}
    dropped: dict[str, str] = {}

    for name in names:
        spec = specs[name]

        # A submit tool is never dropped. The loop terminates only by calling it, so a
        # run without one cannot end — it burns its whole budget and reports a timeout.
        if spec.safety != "submit":
            missing = sorted(spec.requires - served)
            if missing:
                dropped[name] = f"transport lacks {', '.join(missing)}"
                continue
            if name in _MEMORY_TOOLS and ctx.memory is None:
                dropped[name] = "no experience store in this configuration"
                continue

        definitions.append(
            ModelToolSpec(
                name=name,
                description=description_for(name, role, descriptions=catalogue),
                parameters=definition_for(spec),
            )
        )
        if spec.dispatch is not None:
            dispatch[name] = BoundTool(spec=spec, ctx=ctx)

    return ResolvedToolset(definitions=definitions, dispatch=dispatch, dropped=dropped)
