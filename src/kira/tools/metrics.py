"""Metrics tools.

``get_metrics`` runs a PromQL query; ``get_alerts`` lists the alerts currently firing.
Neither passes a time range. The prototype could not: its metrics backend windowed
server-side and accepted no range, so every prompt and every published measurement was
written against a window the agent does not choose. A transport that *can* take a range
therefore has to pick one itself rather than being handed one from here.
"""

from typing import Any

from kira.tools.context import ToolContext
from kira.tools.errors import OUTPUT_LIMIT
from kira.tools.registry import tool

__all__ = ["get_alerts", "get_metrics"]

_METRICS_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "PromQL expression."},
    },
    "required": ["query"],
}

_ALERTS_PARAMETERS: dict[str, Any] = {"type": "object", "properties": {}, "required": []}


@tool(
    name="get_metrics",
    safety="read_only",
    requires={"metrics"},
    parameters=_METRICS_PARAMETERS,
)
async def get_metrics(ctx: ToolContext, query: str) -> str:
    """Run a PromQL query and return the matching series.

    Args:
        ctx: The run's tool context.
        query: The PromQL expression the model wrote.

    Returns:
        The result, saved to the session workspace when it exceeds the output limit.
    """
    raw = await ctx.transport.query_metrics(query)
    return ctx.workspace.save_tool_output(raw, "prometheus_metrics", OUTPUT_LIMIT)


@tool(
    name="get_alerts",
    safety="read_only",
    requires={"metrics"},
    parameters=_ALERTS_PARAMETERS,
)
async def get_alerts(ctx: ToolContext) -> str:
    """Return every currently firing alert with its labels and annotations.

    Args:
        ctx: The run's tool context.

    Returns:
        The firing alerts, saved to the session workspace when they exceed the limit.
    """
    raw = await ctx.transport.query_alerts()
    return ctx.workspace.save_tool_output(raw, "prometheus_alerts", OUTPUT_LIMIT)
