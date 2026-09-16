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
from maver.orchestration.engine import run_pipeline_node
from maver.orchestration.nodes import NodeFn
from maver.orchestration.routing import (
    RoutingPolicy,
    after_execution,
    after_planning,
    after_rca,
    after_verify,
)
from maver.pipelines.definitions import ResolvedPipeline

__all__ = ["build_graph"]

Predicate = Callable[[IncidentState, RoutingPolicy], Any]


def _router(predicate: Predicate, policy: RoutingPolicy) -> Callable[[IncidentState], str]:
    """Bind a routing predicate to this incident's policy.

    The predicate is framework-free and lives in ``routing.py``; this only adapts its
    return type to the string the graph library wants.
    """
    return lambda state: str(predicate(state, policy).value)


def build_graph(
    nodes: Mapping[str, NodeFn], policy: RoutingPolicy, pipeline: ResolvedPipeline | None = None
) -> Any:
    """Wire and compile the incident graph.

    The verification node is registered whether or not verification is enabled. What
    turns it off is the predicate, not the topology, so one compiled shape serves every
    configuration and the graph a reader inspects is the graph that ran.

    Args:
        nodes: Role name to the node function that runs it.
        policy: The routing policy for this incident.
        pipeline: A resolved Python workflow, or None for the historical graph.

    Returns:
        The compiled graph.
    """
    if pipeline is not None:
        return _configured_graph(nodes, policy, pipeline)
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


def _configured_graph(
    nodes: Mapping[str, NodeFn], policy: RoutingPolicy, pipeline: ResolvedPipeline
) -> Any:
    """Compile Python-declared topology using the same durable bound as the loop."""
    missing = {stage.name for stage in pipeline.workflow.stages} - nodes.keys()
    if missing:
        raise ValueError(f"pipeline has no runtime nodes for stages: {sorted(missing)}")
    graph = StateGraph(IncidentState)
    for stage in pipeline.workflow.stages:
        graph.add_node(
            stage.name, cast(Any, _bounded_node(stage.name, nodes[stage.name], pipeline))
        )
        graph.add_conditional_edges(
            stage.name,
            _configured_router(stage.name, pipeline, policy),
            {
                destination if destination is not None else END: destination
                if destination is not None
                else END
                for destination in pipeline.workflow.transitions[stage.name].destinations
            },
        )
    graph.add_edge(START, pipeline.entry)
    return graph.compile()


def _bounded_node(stage: str, node: NodeFn, pipeline: ResolvedPipeline) -> NodeFn:
    """Bind a workflow position without sharing closure variables across nodes."""

    async def bounded(state: IncidentState) -> Any:
        return await run_pipeline_node(stage, node, state, pipeline.max_steps)

    return bounded


def _configured_router(
    stage: str, pipeline: ResolvedPipeline, policy: RoutingPolicy
) -> Callable[[IncidentState], str]:
    """Adapt the common checked router to the framework's terminal sentinel."""

    def route(state: IncidentState) -> str:
        destination = pipeline.next_stage(stage, state, policy)
        return destination if destination is not None else str(END)

    return route
