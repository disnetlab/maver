"""Distributed-trace tools.

``get_services`` and ``get_operations`` name what is reporting spans, ``get_traces``
returns recent spans for one service, and ``get_dependency_graph`` returns the
service-to-service edges observed in a window.

What a trace backend calls a service is a name with no Kubernetes namespace attached, so
a backend shared by several applications answers every one of these queries for all of
them at once. Deciding which of those names belong to the incident is therefore work this
module has to do: the allowed set is derived from one cluster read of the incident
namespace's own workloads, cached on the incident scope for the run, and every result is
narrowed to it before the model sees it. Unnarrowed, an agent can read another
application's traces and cite them as evidence for this incident.
"""

import ast
import json
from typing import Any

from kira.tools.context import ToolContext
from kira.tools.errors import OUTPUT_LIMIT
from kira.tools.kubernetes import read_kubectl
from kira.tools.policy import read_only
from kira.tools.registry import tool

__all__ = ["get_dependency_graph", "get_operations", "get_services", "get_traces"]

# Labels a chart uses to name the workload behind a pod, and therefore the names a trace
# backend is likely to know it by. A chart routinely names its Deployment one thing and
# reports spans under another, so the object names alone are not the allowed set.
_APP_LABELS = ("app", "app.kubernetes.io/name", "io.kompose.service")

_MAX_LISTED_CANDIDATES = 30

_SAFE_PASSTHROUGH_PREFIXES = (
    "error:",
    "[ob_mcp]",
    "[loki_mcp]",
    "[prom_mcp]",
    "Query failed:",
    "No dependency data available",
)
"""Payload heads that are already a failure or a notice, and so carry nothing to leak.

The bracketed ones are artefacts of the MCP transport. A direct transport never produces
them, and the tuple stays because the benchmark configuration still can.
"""

_PASSTHROUGH_LIMIT = 400

_SERVICES_PARAMETERS: dict[str, Any] = {"type": "object", "properties": {}, "required": []}

_OPERATIONS_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "service": {
            "type": "string",
            "description": "Service name whose operations to list.",
        },
    },
    "required": ["service"],
}

_TRACES_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "service": {
            "type": "string",
            "description": "Jaeger service name (e.g. 'frontend').",
        },
        "last_n_minutes": {
            "type": "integer",
            "description": "Lookback window in minutes.",
        },
    },
    "required": ["service", "last_n_minutes"],
}

_DEPENDENCY_GRAPH_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "last_n_minutes": {
            "type": "integer",
            "description": "Lookback window in minutes (default 30).",
        },
    },
    "required": [],
}


def _parse_structured(raw: str) -> Any:
    """Parse the JSON or Python-literal response shape a trace backend may return.

    None means the payload was neither, and no reading of it may be invented.
    """
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(raw)
        except (TypeError, ValueError, SyntaxError):
            pass
    return None


def _candidate_names_from_inventory(raw: str) -> frozenset[str] | None:
    """Collect the names one namespace's workloads could report traces under.

    None means the inventory could not be read, which is not the same as a namespace
    that holds nothing.
    """
    if not raw or raw.startswith("error:"):
        return None
    data = _parse_structured(raw)
    if not isinstance(data, dict) or not isinstance(data.get("items"), list):
        return None
    names: set[str] = set()
    for item in data["items"]:
        if not isinstance(item, dict):
            continue
        metadata = item.get("metadata") or {}
        if not isinstance(metadata, dict):
            continue
        name = metadata.get("name")
        if isinstance(name, str) and name:
            names.add(name)
        labels = metadata.get("labels") or {}
        if isinstance(labels, dict):
            for key in _APP_LABELS:
                value = labels.get(key)
                if isinstance(value, str) and value:
                    names.add(value)
    return frozenset(names)


async def _scoped_service_candidates(
    ctx: ToolContext,
) -> tuple[str | None, frozenset[str] | None, str | None]:
    """Return ``(namespace, candidates, error)`` for this incident.

    Failure is deliberately closed. Returning Jaeger's global service catalogue after an
    inventory error would reintroduce cross-application evidence. That is the opposite
    reading from the health gate, which falls open when it cannot look, and both are
    right: a gate that could not look has found no evidence of harm and must not invent
    an incident, whereas a scope that could not be computed has not been enforced, and an
    unenforced scope is a leak rather than a gap.
    """
    namespace = ctx.scope.namespace
    if not namespace:
        return None, None, None
    if ctx.scope.trace_service_candidates is not None:
        return namespace, ctx.scope.trace_service_candidates, None

    raw = await read_kubectl(
        ctx,
        f"kubectl get services,deployments,statefulsets,daemonsets -n {namespace} -o json",
    )
    candidates = _candidate_names_from_inventory(raw)
    if not candidates:
        return (
            namespace,
            None,
            "error: unable to enforce Jaeger namespace scope for "
            f"'{namespace}': Kubernetes service/workload inventory is unavailable",
        )
    ctx.scope.trace_service_candidates = candidates
    return namespace, candidates, None


def _unparseable_scope_error(what: str, raw: str) -> str:
    """Report that scoping could not read the payload, unless the payload says why.

    A known-safe error or notice string has no cross-namespace telemetry in it, so it is
    passed through. Masking one made the agent read a failed upstream query as a
    namespace problem and spend its turns guessing at the wrong thing.
    """
    head = raw.strip()[:_PASSTHROUGH_LIMIT]
    if head.startswith(_SAFE_PASSTHROUGH_PREFIXES):
        return head
    return f"error: unable to enforce Jaeger namespace scope: Jaeger returned {what}"


def _outside_scope_error(service: str, namespace: str, candidates: frozenset[str]) -> str:
    """Refuse a service the incident namespace does not own, and name the ones it does."""
    available = ", ".join(sorted(candidates)[:_MAX_LISTED_CANDIDATES])
    return (
        f"error: Jaeger service '{service}' is outside incident namespace '{namespace}'. "
        f"Allowed Kubernetes service/workload names: {available}"
    )


def _filter_traces_to_candidates(raw: str, candidates: frozenset[str]) -> str | None:
    """Drop the spans and processes the backend included from another application.

    None means the payload did not have the shape this knows how to filter, which the
    caller reports rather than letting an unfiltered result through.
    """
    if raw.strip() in {"None", "[]"}:
        return raw.strip()
    traces = _parse_structured(raw)
    if not isinstance(traces, list):
        return None
    filtered: list[dict[str, Any]] = []
    for trace in traces:
        if not isinstance(trace, dict):
            continue
        processes = trace.get("processes")
        spans = trace.get("spans")
        if not isinstance(processes, dict) or not isinstance(spans, list):
            return None
        kept_processes = {
            process_id: process
            for process_id, process in processes.items()
            if isinstance(process, dict) and process.get("serviceName") in candidates
        }
        kept_spans = [
            span
            for span in spans
            if isinstance(span, dict) and span.get("processID") in kept_processes
        ]
        if not kept_spans:
            continue
        item = dict(trace)
        item["processes"] = kept_processes
        item["spans"] = kept_spans
        filtered.append(item)
    return str(filtered)


@tool(name="get_services", safety="read_only", requires={"traces"}, parameters=_SERVICES_PARAMETERS)
async def get_services(ctx: ToolContext) -> str:
    """List the services *currently reporting traces*, restricted to the alert namespace.

    This is a liveness signal, not an inventory, and the difference matters. The list is
    the backend's view intersected with the namespace's workloads, so a service that has
    stopped serving simply disappears from it. Returning the bare intersection let agents
    read ``['recommendation']`` for an eight-service namespace as "this namespace has one
    service" and clear the incident, when the seven missing services were the failure.
    Reporting what is absent alongside what is present makes a collapsed list read as the
    symptom it is.

    Args:
        ctx: The run's tool context.

    Returns:
        The reporting and the silent sets, saved to the session workspace when they
        exceed the limit.
    """
    namespace, candidates, scope_error = await _scoped_service_candidates(ctx)
    if scope_error:
        return scope_error
    raw = await ctx.transport.query_traces("services")
    if namespace is None or candidates is None:
        return ctx.workspace.save_tool_output(raw, "jaeger_services", OUTPUT_LIMIT)
    services = [] if raw.strip() == "None" else _parse_structured(raw)
    if not isinstance(services, list) or not all(isinstance(s, str) for s in services):
        return _unparseable_scope_error("an unparseable service catalogue", raw)
    reporting = sorted(s for s in services if s in candidates)
    silent = sorted(c for c in candidates if c not in set(services))
    body = f"Services reporting traces in namespace {namespace!r}: {reporting!r}"
    if silent:
        body += (
            f"\nWorkloads in {namespace!r} with NO traces in this window: {silent!r}. "
            "This is not an inventory of what exists — these workloads are deployed but "
            "silent. A service that stopped serving stops reporting, so treat silence as "
            "a symptom to investigate, not as absence."
        )
    return ctx.workspace.save_tool_output(body, "jaeger_services", OUTPUT_LIMIT)


@tool(
    name="get_operations",
    safety="read_only",
    requires={"traces"},
    parameters=_OPERATIONS_PARAMETERS,
)
async def get_operations(ctx: ToolContext, service: str) -> str:
    """List the operations one service reports.

    Args:
        ctx: The run's tool context.
        service: The service name, which must belong to the incident namespace.

    Returns:
        The operations, saved to the session workspace when they exceed the limit.
    """
    namespace, candidates, scope_error = await _scoped_service_candidates(ctx)
    if scope_error:
        return scope_error
    if namespace is not None and candidates is not None and service not in candidates:
        return _outside_scope_error(service, namespace, candidates)
    raw = await ctx.transport.query_traces("operations", service=service)
    return ctx.workspace.save_tool_output(raw, "jaeger_operations", OUTPUT_LIMIT)


@tool(name="get_traces", safety="read_only", requires={"traces"}, parameters=_TRACES_PARAMETERS)
async def get_traces(ctx: ToolContext, service: str, last_n_minutes: int) -> str:
    """Fetch recent traces for one service.

    A trace crosses application boundaries, so scoping the service it was asked for is
    not enough on its own. The spans inside each returned trace are narrowed to the
    namespace as well, and a trace left with none of them is dropped.

    Args:
        ctx: The run's tool context.
        service: The service name, which must belong to the incident namespace.
        last_n_minutes: How far back to look.

    Returns:
        The traces, saved to the session workspace when they exceed the limit.
    """
    namespace, candidates, scope_error = await _scoped_service_candidates(ctx)
    if scope_error:
        return scope_error
    if namespace is not None and candidates is not None and service not in candidates:
        return _outside_scope_error(service, namespace, candidates)
    raw = await ctx.transport.query_traces("traces", service=service, last_n_minutes=last_n_minutes)
    if namespace is not None and candidates is not None:
        filtered = _filter_traces_to_candidates(raw, candidates)
        if filtered is None:
            return _unparseable_scope_error("unparseable traces", raw)
        raw = filtered
    return ctx.workspace.save_tool_output(raw, "jaeger_traces", OUTPUT_LIMIT)


@tool(
    name="get_dependency_graph",
    safety="read_only",
    requires={"traces"},
    parameters=_DEPENDENCY_GRAPH_PARAMETERS,
)
async def get_dependency_graph(ctx: ToolContext, last_n_minutes: int = 30) -> str:
    """Fetch the service dependency edges observed in a window.

    Only edges with both ends inside the incident namespace survive. Keeping a half-inside
    edge would be worse than dropping it: it names a service the model has no scoped way
    to look at, so the only thing it can do with the name is follow it out of the
    incident.

    Args:
        ctx: The run's tool context.
        last_n_minutes: How far back to look.

    Returns:
        The graph, saved to the session workspace when it exceeds the limit.
    """
    namespace, candidates, scope_error = await _scoped_service_candidates(ctx)
    if scope_error:
        return scope_error
    raw = await ctx.transport.query_traces("dependencies", last_n_minutes=last_n_minutes)
    if namespace is not None and candidates is not None:
        graph = _parse_structured(raw)
        edges = graph.get("data") if isinstance(graph, dict) else None
        if not isinstance(edges, list):
            return _unparseable_scope_error("an unparseable dependency graph", raw)
        graph["data"] = [
            edge
            for edge in edges
            if isinstance(edge, dict)
            and edge.get("parent") in candidates
            and edge.get("child") in candidates
        ]
        raw = str(graph)
    return ctx.workspace.save_tool_output(raw, "jaeger_deps", OUTPUT_LIMIT)


read_only("get_services")
read_only("get_operations")
read_only("get_traces")
read_only("get_dependency_graph")
