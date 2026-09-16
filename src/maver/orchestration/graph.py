"""The compiled incident graph. The only module in the package that imports LangGraph.

Four operations are used and no more: add a node, add an edge, compile, invoke. No
checkpointer, no saver, no interrupt, no built-in human-in-the-loop, no prebuilt agents.
Persistence, resume, approval and lifecycle are ours, in our own modules, because they
are load-bearing for what the system claims about auditability and must not depend on a
framework's roadmap. The target is that if this dependency disappeared, replacing it
would take days.

Containment is enforced twice: a banned-import rule everywhere except this file, and a
test that reads this file for the forbidden names — because the import exemption that
makes this module possible would otherwise also let the forbidden surface in here.
"""

from collections.abc import Callable, Mapping
from typing import Any, cast

from langgraph.graph import END, START, StateGraph

from maver.contracts.state import IncidentState
from maver.orchestration.nodes import NodeFn
from maver.orchestration.routing import (
    RoutingPolicy,
    after_execution,
    after_planning,
    after_rca,
    after_verify,
)

__all__ = ["build_graph"]

Predicate = Callable[[IncidentState, RoutingPolicy], Any]


def _router(predicate: Predicate, policy: RoutingPolicy) -> Callable[[IncidentState], str]:
    """Bind a routing predicate to this incident's policy.

    The predicate is framework-free and lives in ``routing.py``; this only adapts its
    return type to the string the graph library wants.
    """
    return lambda state: str(predicate(state, policy).value)


def build_graph(nodes: Mapping[str, NodeFn], policy: RoutingPolicy) -> Any:
    """Wire and compile the incident graph.

    The verification node is registered whether or not verification is enabled. What
    turns it off is the predicate, not the topology, so one compiled shape serves every
    configuration and the graph a reader inspects is the graph that ran.

    Args:
        nodes: Role name to the node function that runs it.
        policy: The routing policy for this incident.

    Returns:
        The compiled graph.
    """
    graph = StateGraph(IncidentState)
    for name, node in nodes.items():
        # The executor types a node more loosely than we do; ours returns a partial
        # state, which is the contract every node in this package keeps.
        graph.add_node(name, cast(Any, node))

    graph.add_edge(START, "rca")
    graph.add_conditional_edges(
        "rca", _router(after_rca, policy), {"planning": "planning", "end": END}
    )
    graph.add_conditional_edges(
        "planning", _router(after_planning, policy), {"execution": "execution", "end": END}
    )
    graph.add_conditional_edges(
        "execution",
        _router(after_execution, policy),
        {"verify": "verify", "planning": "planning", "end": END},
    )
    graph.add_conditional_edges(
        "verify",
        _router(after_verify, policy),
        {"rca": "rca", "planning": "planning", "verify": "verify", "end": END},
    )
    return graph.compile()
