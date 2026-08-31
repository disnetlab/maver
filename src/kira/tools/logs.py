"""Log tools.

``get_logs`` runs a LogQL query over the last N minutes; ``get_labels`` and
``get_label_values`` exist so the model can build one against labels that are really
there instead of guessing.

The query passes through the evidence policy before it leaves the process, not after the
results come back. That placement is the point: an environment hiding a namespace has to
stop the question, because a model that can *ask* about the hidden namespace learns it
exists from the shape of the answer.
"""

from typing import Any

from kira.tools.context import ToolContext
from kira.tools.errors import OUTPUT_LIMIT
from kira.tools.registry import tool

__all__ = ["get_label_values", "get_labels", "get_logs"]

_LOGS_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "Valid LogQL expression."},
        "last_n_minutes": {
            "type": "integer",
            "description": "Lookback window in minutes (default 15).",
        },
    },
    "required": ["query"],
}

_LABELS_PARAMETERS: dict[str, Any] = {"type": "object", "properties": {}, "required": []}

_LABEL_VALUES_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "label": {
            "type": "string",
            "description": "Label name to list values for (e.g. 'namespace', 'app', 'pod').",
        },
    },
    "required": ["label"],
}


@tool(name="get_logs", safety="read_only", requires={"logs"}, parameters=_LOGS_PARAMETERS)
async def get_logs(ctx: ToolContext, query: str, last_n_minutes: int = 15) -> str:
    """Run a LogQL query over a recent window.

    Args:
        ctx: The run's tool context.
        query: The LogQL expression the model wrote.
        last_n_minutes: How far back to look.

    Returns:
        Matching log lines, saved to the session workspace when they exceed the limit.
    """
    raw = await ctx.transport.query_logs(
        ctx.evidence.filter_query(query), last_n_minutes=last_n_minutes
    )
    return ctx.workspace.save_tool_output(raw, "loki_logs", OUTPUT_LIMIT)


@tool(name="get_labels", safety="read_only", requires={"logs"}, parameters=_LABELS_PARAMETERS)
async def get_labels(ctx: ToolContext) -> str:
    """List every available log label name.

    Args:
        ctx: The run's tool context.

    Returns:
        The label names, saved to the session workspace when they exceed the limit.
    """
    raw = await ctx.transport.log_labels()
    return ctx.workspace.save_tool_output(raw, "loki_labels", OUTPUT_LIMIT)


@tool(
    name="get_label_values",
    safety="read_only",
    requires={"logs"},
    parameters=_LABEL_VALUES_PARAMETERS,
)
async def get_label_values(ctx: ToolContext, label: str) -> str:
    """List every value one log label takes.

    Args:
        ctx: The run's tool context.
        label: The label name.

    Returns:
        The values, saved to the session workspace when they exceed the limit.
    """
    raw = await ctx.transport.log_label_values(label)
    return ctx.workspace.save_tool_output(raw, "loki_label_values", OUTPUT_LIMIT)
