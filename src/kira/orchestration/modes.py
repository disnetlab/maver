"""What "running an incident" means, in the three shapes it can take.

A registry rather than a chain of conditionals, because the choice between them is the
single decision the runner makes and it should be readable in one place. The precedence
is explicit for the same reason: a diagnosis-only task wins over the configured mode,
and in the prototype it did so only because its branch happened to be written first.

Diagnosis-only exists because some environments grade detection, localisation and
analysis independently. Running the planning or remediating roles for those tasks would
mutate the environment before the read-only answer had been graded, which invalidates
the comparison the task exists to make.
"""

import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, cast

from kira.contracts.state import IncidentState, TargetStage
from kira.orchestration.engine import Engine
from kira.orchestration.nodes import IncidentDeps, NodeFn, make_node
from kira.roles.registry import get_role
from kira.settings.root import Settings

__all__ = ["MODES", "ModeContext", "mode", "resolve_mode"]

_log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ModeContext:
    """What every mode needs, whether or not it uses all of it."""

    deps: IncidentDeps
    engine: Engine
    nodes: Mapping[str, NodeFn] = field(default_factory=dict)
    """The composed roles, as the graph would run them.

    Carried rather than rebuilt, because a mode that built its own would run a role
    without whatever wraps it — a diagnosis-only task under an arm that second-guesses
    its diagnosis would then be a different arm, silently.
    """
    start: str | None = None
    """Where a continuation re-enters, or None for a run that starts at the beginning.

    Only the graph has a part-way entry to name. A mode that runs one role re-enters
    that role, which is the only place it could resume."""


ModeFn = Callable[[IncidentState, ModeContext], Awaitable[IncidentState]]

MODES: dict[str, ModeFn] = {}


def mode(name: str) -> Callable[[ModeFn], ModeFn]:
    """Register a mode under a name.

    Args:
        name: The mode name.

    Returns:
        The decorator.
    """

    def register(fn: ModeFn) -> ModeFn:
        MODES[name] = fn
        return fn

    return register


def resolve_mode(*, target_stage: TargetStage, settings: Settings) -> str:
    """Choose the mode for this run.

    Args:
        target_stage: How far the caller asked the pipeline to go.
        settings: The run's settings.

    Returns:
        The mode name.
    """
    # Stated rather than implied by the order of two branches: a diagnosis-only task is
    # a request about what may be done to the environment, and no configured mode
    # overrides it.
    if target_stage == "diagnose":
        return "diagnose"
    return settings.pipeline.mode


@mode("multi")
async def run_multi(state: IncidentState, ctx: ModeContext) -> IncidentState:
    """Run the graph: diagnosis, planning, remediation, and verification when enabled.

    Args:
        state: The seeded incident state.
        ctx: The run's shared resources.

    Returns:
        The terminal state.
    """
    if ctx.start is None:
        return await ctx.engine.run(state)
    return await ctx.engine.run(state, start=ctx.start)


@mode("single")
async def run_single(state: IncidentState, ctx: ModeContext) -> IncidentState:
    """Run the one-agent baseline.

    Its two phases write into the same diagnosis and remediation state slots the
    pipeline uses, which is what lets one environment driver grade either configuration
    without knowing which it is looking at.

    Args:
        state: The seeded incident state.
        ctx: The run's shared resources.

    Returns:
        The terminal state.
    """
    return await _run_one("single", state, ctx)


@mode("diagnose")
async def run_diagnose(state: IncidentState, ctx: ModeContext) -> IncidentState:
    """Run the diagnosing role once and stop, mutating nothing.

    Args:
        state: The seeded incident state.
        ctx: The run's shared resources.

    Returns:
        The terminal state.
    """
    return await _run_one("rca", state, ctx)


async def _run_one(role: str, state: IncidentState, ctx: ModeContext) -> IncidentState:
    """Run one role through the same wrapper the graph uses, and merge its delta.

    Composed if the incident's own composition names it, and wrapped plainly otherwise:
    a role outside the graph, like the single-agent configuration, has nothing composed
    around it to look up.
    """
    node = ctx.nodes.get(role) or make_node(get_role(role), ctx.deps)
    delta = await node(state)
    return cast(IncidentState, {**state, **delta})


def registered_modes() -> Mapping[str, Any]:
    """Every mode registered so far.

    Returns:
        A read-only view of the registry.
    """
    return dict(MODES)
