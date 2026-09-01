"""Metrics tools.

``get_metrics`` runs a PromQL query; ``get_alerts`` lists the alerts currently firing.
Neither passes a time range. The prototype could not: its metrics backend windowed
server-side and accepted no range, so every prompt and every published measurement was
written against a window the agent does not choose. A transport that *can* take a range
therefore has to pick one itself rather than being handed one from here.

``get_alerts`` is not a pass-through, and the two stages it adds are the reason. A
metrics backend answers for the whole cluster, while an incident belongs to one
namespace, so an unpartitioned list invites a model to adopt another application's
firing alert as evidence for this one. And a benchmark that recreates its environment
between problems leaves alerts behind whose namespace is already gone, which the first
run of the next arm inherits and reads as live. Both are labelled rather than removed:
a foreign alert can be a genuine dependency signal, and a model told what it is looking
at can decide, where a model handed a filtered list cannot.
"""

import ast
from collections.abc import Iterable, Mapping
from typing import Any

from kira.tools.context import ToolContext
from kira.tools.errors import OUTPUT_LIMIT
from kira.tools.kubernetes import read_kubectl
from kira.tools.policy import read_only
from kira.tools.registry import tool

__all__ = ["get_alerts", "get_metrics", "partition_incident_alerts"]

_STALE_NAMESPACE_MESSAGE = (
    "namespace {namespace!r} is absent from the Kubernetes API; this firing alert "
    "is stale and must not be treated as an active incident"
)

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
    if ctx.settings.annotate_stale_namespaces:
        raw = await _annotate_stale_namespaces(ctx, raw)
    raw = partition_incident_alerts(ctx.scope.namespace, raw)
    return ctx.workspace.save_tool_output(raw, "prometheus_alerts", OUTPUT_LIMIT)


def _parse_alerts(raw: str) -> list[dict[str, Any]] | None:
    """Read an alert payload, without weakening what a failure to read it means.

    None means the payload was not the expected list of dictionaries and must be
    returned byte for byte; an empty list is a valid parsed response and not the same
    thing. That distinction is what keeps a backend's format change visible: a parser
    that fell back to an empty list would hide every firing alert and report a quiet
    cluster.
    """
    try:
        alerts = ast.literal_eval(raw)
    except (SyntaxError, ValueError, TypeError, MemoryError, RecursionError):
        return None
    if not isinstance(alerts, list) or not all(isinstance(alert, dict) for alert in alerts):
        return None
    return alerts


def _namespace_of(alert: Mapping[str, Any]) -> str | None:
    labels = alert.get("labels")
    namespace = labels.get("namespace") if isinstance(labels, Mapping) else None
    return namespace if isinstance(namespace, str) and namespace.strip() else None


def _alert_namespaces(alerts: Iterable[Mapping[str, Any]]) -> set[str]:
    return {ns.strip() for alert in alerts if (ns := _namespace_of(alert)) is not None}


def _namespace_is_missing(output: str) -> bool:
    """Only an explicit NotFound counts.

    A timeout or an RBAC refusal says nothing about whether the namespace exists, and
    treating either as proof would mark a live incident's own alerts stale.
    """
    normalised = output.lower()
    return "notfound" in normalised or "not found" in normalised


async def _annotate_stale_namespaces(ctx: ToolContext, raw: str) -> str:
    """Mark alerts whose namespace is gone from the cluster.

    An environment that tears its namespaces down between problems leaves alerts firing
    for namespaces that no longer exist. The first run after a teardown inherits them
    and, with nothing to say otherwise, reads them as live evidence for its own
    incident.
    """
    alerts = _parse_alerts(raw)
    if alerts is None:
        return raw

    missing = set()
    for candidate in _alert_namespaces(alerts):
        response = await read_kubectl(ctx, f"kubectl get namespace {candidate} -o name")
        if _namespace_is_missing(response):
            missing.add(candidate)
    if not missing:
        return raw

    annotated = []
    for alert in alerts:
        namespace = _namespace_of(alert)
        if namespace is not None and namespace.strip() in missing:
            # A copy: the payload belongs to the transport, not to this tool.
            alert = dict(alert)
            alert["stale_namespace"] = _STALE_NAMESPACE_MESSAGE.format(namespace=namespace.strip())
        annotated.append(alert)
    return repr(annotated)


def partition_incident_alerts(namespace: str | None, raw: str) -> str:
    """Put this incident's alerts first, and label every other alert for what it is.

    Nothing is dropped. A firing alert in another namespace can be a real dependency
    signal, so removing it would hide evidence; presenting it unlabelled invites the
    model to adopt it as its own. Labelling is the only option that does neither.

    Args:
        namespace: The incident's namespace, or None when the alert named none.
        raw: The payload from the metrics backend.

    Returns:
        The rendering the model sees, or the payload unchanged when there is no
        namespace to partition by or nothing parseable to partition.
    """
    if not namespace:
        return raw
    alerts = _parse_alerts(raw)
    if alerts is None:
        return raw

    scoped: list[dict[str, Any]] = []
    external: list[dict[str, Any]] = []
    unscoped: list[dict[str, Any]] = []
    stale: list[dict[str, Any]] = []
    for alert in alerts:
        # Order matters: a stale alert in this very namespace is still stale, so that
        # test comes before the namespace match.
        alert_namespace = _namespace_of(alert)
        if alert.get("stale_namespace"):
            stale.append(alert)
        elif alert_namespace == namespace:
            scoped.append(alert)
        elif alert_namespace is not None:
            external.append(alert)
        else:
            unscoped.append(alert)

    if not external and not unscoped and not stale:
        return repr(scoped) if scoped else "No firing alerts"

    sections = [
        f"INCIDENT-SCOPED firing alerts for namespace {namespace!r}: {scoped!r}"
        if scoped
        else f"No firing alerts scoped to incident namespace {namespace!r}."
    ]
    if external:
        sections.append(
            "OUT-OF-SCOPE firing alerts from other namespaces (not evidence for "
            "this incident unless live dependency evidence links them): " + repr(external)
        )
    if unscoped:
        sections.append(
            "UNSCOPED firing alerts (do not assign them to this incident without "
            "live resource/dependency evidence): " + repr(unscoped)
        )
    if stale:
        sections.append(
            "STALE firing alerts whose namespace no longer exists (never active "
            "incident evidence): " + repr(stale)
        )
    return "\n".join(sections)


read_only("get_metrics")
read_only("get_alerts")
