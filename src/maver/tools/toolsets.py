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

from maver.adapters.base import AdapterTool
from maver.llm.types import ToolSpec as ModelToolSpec
from maver.tools.allowlists import RETRIEVAL_TOOLS, ROLE_TOOLSETS
from maver.tools.context import ToolContext, kubectl_route
from maver.tools.registry import (
    BoundTool,
    ToolRegistryError,
    ToolSpec,
    definition_for,
    description_for,
    load_descriptions,
    registered,
)
from maver.tools.sandbox import fs_sandbox_available

__all__ = ["ResolvedToolset", "build_toolset"]


def _sandboxed() -> bool:
    """Read through the module so a test can answer for a kernel it is not running on."""
    return fs_sandbox_available()


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
    extras: Sequence[AdapterTool] = (),
    drop: frozenset[str] = frozenset(),
) -> ResolvedToolset:
    """Resolve a role's tools against this run's environment and configuration.

    Args:
        role: The role being run.
        ctx: The run's context, bound into every dispatch.
        descriptions: The description catalogue. Defaults to the shipped one.
        capabilities: What the transport serves. Defaults to asking the bound transport.
        extras: Tools an environment adapter supplies, each carrying the roles allowed
            to see it. Unioned after the allowlist rather than named in it, because the
            core's allowlists must not carry the name of an environment's tool.
        drop: Tools the environment withholds from this role, from its overlay. Dropped
            before the run with the rest, so the model is never offered a tool the
            environment will refuse: a tool that errors looks like a fault to work
            around, and the model will try it again.

    Returns:
        The model-facing definitions, the dispatch map, and what was dropped.

    Raises:
        ToolRegistryError: When the role has no allowlist, or when an adapter's tool
            collides with a registered name. A silent override would make the audit
            record say one thing and mean another.
    """
    names = ROLE_TOOLSETS.get(role)
    if names is None:
        raise ToolRegistryError(f"no toolset defined for role {role!r}")

    catalogue = descriptions if descriptions is not None else load_descriptions()
    served = frozenset(capabilities if capabilities is not None else ctx.transport.capabilities)
    # Asked of the transport, like the capabilities above and for the same reason: what
    # a tool can do here is a property of the environment this run was handed, and the
    # two sandbox tools do opposite things on the two routes.
    route = kubectl_route(ctx.transport)
    specs = registered()

    definitions: list[ModelToolSpec] = []
    dispatch: dict[str, BoundTool] = {}
    dropped: dict[str, str] = {}

    offered = _offer(role, [specs[name] for name in names], extras, specs)

    for spec in offered:
        name = spec.name

        # A submit tool is never dropped. The loop terminates only by calling it, so a
        # run without one cannot end — it burns its whole budget and reports a timeout.
        if spec.safety != "submit":
            if name in drop:
                dropped[name] = "withheld by the environment"
                continue
            if name in ctx.settings.disabled:
                # A kill switch, and the reason submissions are exempt from it is the
                # same reason they are exempt from everything else: a run with no way
                # to finish does not fail, it burns its whole budget and reports a
                # timeout that reads as the model's fault.
                dropped[name] = "withheld by configuration"
                continue
            missing = sorted(spec.requires - served)
            if missing:
                dropped[name] = f"transport lacks {', '.join(missing)}"
                continue
            if name in RETRIEVAL_TOOLS and ctx.memory is None:
                dropped[name] = "no experience store in this configuration"
                continue
            if spec.safety == "local_exec" and ctx.settings.fs_sandbox and not _sandboxed():
                # The one place this rule was not applied. The sandbox is fail-closed on
                # Linux, so on a kernel with no usable Landlock — the documented amd64
                # image under emulation is one — these two returned `error:` on every
                # call for a whole incident, which in the results is indistinguishable
                # from a model that could not use them.
                dropped[name] = "no filesystem sandbox on this kernel"
                continue

        definitions.append(
            ModelToolSpec(
                name=name,
                description=description_for(
                    name, role, descriptions=spec.descriptions or catalogue, route=route
                ),
                parameters=definition_for(spec),
            )
        )
        if spec.dispatch is not None:
            dispatch[name] = BoundTool(spec=spec, ctx=ctx)

    return ResolvedToolset(definitions=definitions, dispatch=dispatch, dropped=dropped)


def _offer(
    role: str,
    allowed: list[ToolSpec],
    extras: Sequence[AdapterTool],
    registry: Mapping[str, ToolSpec],
) -> list[ToolSpec]:
    """Splice an adapter's tools into the order the model is shown them in.

    Ahead of the submission tool rather than after it. The order within a role's list
    was hand-tuned and every published measurement was taken with it, and in the
    configuration these tools came from they sat immediately before the submission.

    Args:
        role: The role being run.
        allowed: The role's own tools, in the allowlist's order.
        extras: What an adapter contributes.
        registry: The registered tools, for the collision check.

    Returns:
        Every tool this role is offered, in order.

    Raises:
        ToolRegistryError: When an adapter's tool collides with a registered name.
    """
    added: list[ToolSpec] = []
    for extra in extras:
        if role not in extra.roles:
            continue
        if extra.spec.name in registry:
            raise ToolRegistryError(
                f"adapter tool {extra.spec.name!r} collides with a registered tool name"
            )
        added.append(extra.spec)
    if not added:
        return allowed
    cut = next((i for i, spec in enumerate(allowed) if spec.safety == "submit"), len(allowed))
    return [*allowed[:cut], *added, *allowed[cut:]]
