"""Distributed-trace tools.

``get_services`` and ``get_operations`` name what is reporting spans, ``get_traces``
returns recent spans for one service, and ``get_dependency_graph`` returns the
service-to-service edges observed in a window.

All four are one transport call each. What a trace backend calls a service is a name
with no Kubernetes namespace attached, so deciding which of those names belong to the
incident is a scoping question rather than a transport one, and it is answered above
this module.
"""

from typing import Any

from kira.tools.context import ToolContext
from kira.tools.errors import OUTPUT_LIMIT
from kira.tools.policy import read_only
from kira.tools.registry import tool

__all__ = ["get_dependency_graph", "get_operations", "get_services", "get_traces"]

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


@tool(name="get_services", safety="read_only", requires={"traces"}, parameters=_SERVICES_PARAMETERS)
async def get_services(ctx: ToolContext) -> str:
    """List the services currently reporting traces.

    Args:
        ctx: The run's tool context.

    Returns:
        The service catalogue, saved to the session workspace when it exceeds the limit.
    """
    raw = await ctx.transport.query_traces("services")
    return ctx.workspace.save_tool_output(raw, "jaeger_services", OUTPUT_LIMIT)


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
        service: The service name.

    Returns:
        The operations, saved to the session workspace when they exceed the limit.
    """
    raw = await ctx.transport.query_traces("operations", service=service)
    return ctx.workspace.save_tool_output(raw, "jaeger_operations", OUTPUT_LIMIT)


@tool(name="get_traces", safety="read_only", requires={"traces"}, parameters=_TRACES_PARAMETERS)
async def get_traces(ctx: ToolContext, service: str, last_n_minutes: int) -> str:
    """Fetch recent traces for one service.

    Args:
        ctx: The run's tool context.
        service: The service name.
        last_n_minutes: How far back to look.

    Returns:
        The traces, saved to the session workspace when they exceed the limit.
    """
    raw = await ctx.transport.query_traces("traces", service=service, last_n_minutes=last_n_minutes)
    return ctx.workspace.save_tool_output(raw, "jaeger_traces", OUTPUT_LIMIT)


@tool(
    name="get_dependency_graph",
    safety="read_only",
    requires={"traces"},
    parameters=_DEPENDENCY_GRAPH_PARAMETERS,
)
async def get_dependency_graph(ctx: ToolContext, last_n_minutes: int = 30) -> str:
    """Fetch the service dependency edges observed in a window.

    Args:
        ctx: The run's tool context.
        last_n_minutes: How far back to look.

    Returns:
        The graph, saved to the session workspace when it exceeds the limit.
    """
    raw = await ctx.transport.query_traces("dependencies", last_n_minutes=last_n_minutes)
    return ctx.workspace.save_tool_output(raw, "jaeger_deps", OUTPUT_LIMIT)


read_only("get_services")
read_only("get_operations")
read_only("get_traces")
read_only("get_dependency_graph")
