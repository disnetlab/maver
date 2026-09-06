"""Running the routing to completion, with and without the framework.

Two implementations of one protocol. :class:`LangGraphEngine` drives the compiled graph;
:class:`SequentialEngine` walks the same predicates in an explicit loop and imports
nothing. Both are supported configurations, selected by settings, so the framework-free
path is a deployment option rather than a fixture that only tests use.

That is what makes the containment claim checkable. A parametrised test runs both engines
over the same fake roles and asserts the same terminal state and the same sequence of
stage-hook calls, which is the cheapest available evidence that if the dependency
disappeared, deleting ``graph.py`` and defaulting to the sequential engine would be the
whole migration.

Note that neither class imports LangGraph. The graph arrives already compiled, from the
one module permitted to build it, and this file only calls two methods on it.
"""

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, cast

from kira.contracts.routing import Route
from kira.contracts.state import IncidentState
from kira.orchestration.nodes import NodeFn
from kira.orchestration.routing import (
    RoutingPolicy,
    after_execution,
    after_planning,
    after_rca,
    after_verify,
)

__all__ = ["Engine", "LangGraphEngine", "SequentialEngine", "StepLimitExceeded"]

_log = logging.getLogger(__name__)

STEP_LIMIT = 25
"""How many role runs one incident may take before it is treated as a runaway.

Ours, and applied to both engines rather than inherited from either. The graph
executor's own default is 10007, which is not a bound on anything a routing bug can do:
each of those runs is a model call and, in the remediating role, a cluster mutation. So
the sequential engine counts to this and the graph engine is handed it as its recursion
limit, and a configuration able to cycle forever is stopped at the same count in both.

The longest legitimate path through the edge table is well inside it; a run that reaches
this bound has a routing bug, not an unusually hard incident.
"""

_RUN_CONFIG = {"recursion_limit": STEP_LIMIT}
"""What the graph engine hands its invocation. An argument to the call, not a
checkpointer or an interrupt: see the containment rule in ``graph.py``."""

_ROUTERS: Mapping[str, Callable[[IncidentState, RoutingPolicy], Route]] = {
    "rca": after_rca,
    "planning": after_planning,
    "execution": after_execution,
    "verify": after_verify,
}

ENTRY = "rca"
"""Where an incident starts. The same edge ``graph.py`` draws from the graph's start."""


class StepLimitExceeded(RuntimeError):
    """An incident took more role runs than :data:`STEP_LIMIT` allows."""


class Engine(Protocol):
    """Runs the routing to completion."""

    async def run(self, state: IncidentState, *, start: str = ENTRY) -> IncidentState:
        """Run the incident from its seeded state to a terminal one.

        Args:
            state: The incident state to run from.
            start: Which role to run first. Anything but the entry point continues an
                incident that stopped part-way.

        Returns:
            The terminal state.
        """
        ...


@dataclass(frozen=True, slots=True)
class LangGraphEngine:
    """Drives the compiled graph."""

    graph: Any
    """The object ``build_graph`` returned. Held as Any so this module stays free of the
    framework's types along with its imports."""

    async def run(self, state: IncidentState, *, start: str = ENTRY) -> IncidentState:
        """Run the graph to a terminal state.

        Args:
            state: The incident state to run from.
            start: Which role to run first. Only the entry point is supported.

        Returns:
            The terminal state.

        Raises:
            NotImplementedError: When asked to start part-way. Entering a compiled graph
                at an arbitrary node is what a checkpointer is for, and keeping
                persistence and resume in our own code is the reason there is not one.
                A continuation runs on the sequential engine, over the same predicates.
            StepLimitExceeded: The incident took more role runs than the limit allows.
        """
        if start != ENTRY:
            raise NotImplementedError(
                "the graph executor cannot start part-way; resume on the sequential engine"
            )
        # The executor's overflow subclasses the builtin RecursionError, which is how it
        # is translated here without importing the framework this file is kept free of.
        try:
            return cast(IncidentState, await self.graph.ainvoke(state, config=_RUN_CONFIG))
        except RecursionError as exc:
            # Only the executor's own overflow, which subclasses the builtin. A node
            # that genuinely recursed away its stack raises the same class, and
            # relabelling that as a routing loop sends whoever reads the incident back
            # to the edge table for a bug that is not there.
            if type(exc).__name__ != "GraphRecursionError":
                raise
            _log.error("incident %s exceeded %d role runs", state.get("session_id", ""), STEP_LIMIT)
            raise StepLimitExceeded(f"incident exceeded {STEP_LIMIT} role runs") from exc


@dataclass(frozen=True, slots=True)
class SequentialEngine:
    """Walks the same edge table by hand.

    Sixty lines and no dependency. The routing tests run against this, which means the
    edge table is exercised without the framework in the loop and a failure there points
    at a predicate rather than at an executor.
    """

    nodes: Mapping[str, NodeFn]
    policy: RoutingPolicy

    async def run(self, state: IncidentState, *, start: str = ENTRY) -> IncidentState:
        """Run each role and follow its predicate until one routes to the end.

        Args:
            state: The incident state to run from.
            start: Which role to run first. A continuation names the role that was
                interrupted; the routing from there is the routing every run takes.

        Returns:
            The terminal state.

        Raises:
            StepLimitExceeded: The incident took more role runs than the limit allows.
        """
        accumulated = state
        current = start
        for _ in range(STEP_LIMIT):
            delta = await self.nodes[current](accumulated)
            accumulated = cast(IncidentState, {**accumulated, **delta})
            route = _ROUTERS[current](accumulated, self.policy)
            if route is Route.END:
                return accumulated
            current = route.value
        _log.error("incident %s exceeded %d role runs", state.get("session_id", ""), STEP_LIMIT)
        raise StepLimitExceeded(f"incident exceeded {STEP_LIMIT} role runs")
