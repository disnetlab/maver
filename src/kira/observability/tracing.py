"""Tracing, configured once and cheap when nobody is collecting.

Instrumentation is unconditional; only export is configured. That asymmetry is the point
of the setting's wording: a deployment with no collector still runs the same code paths,
so turning collection on later does not change what is measured. The alternative — code
that only runs when a collector is present — is code that is never exercised in the runs
that matter.

Spans carry the incident, the stage and the outcome, and nothing a span attribute should
not hold. Telemetry leaves the cluster in a way the audit log does not, so the payloads
stay here and only their shape travels.
"""

import logging
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Status, StatusCode

from kira.settings.root import Settings

__all__ = ["configure_tracing", "span"]

_log = logging.getLogger(__name__)

_TRACER_NAME = "kira"
_CONFIGURED = False


def configure_tracing(settings: Settings) -> None:
    """Install the tracer provider, exporting only when a collector is configured.

    Idempotent, because a restarted process configures again and a second provider
    would silently orphan every span the first one held.

    Args:
        settings: The deployment's settings.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True

    attributes: dict[str, Any] = {"service.name": "kira"}
    if settings.observability.deployment_id:
        attributes["deployment.id"] = settings.observability.deployment_id
    attributes["kira.model_profile"] = settings.model.profile
    attributes["kira.pipeline_mode"] = settings.pipeline.mode

    provider = TracerProvider(resource=Resource.create(attributes))
    endpoint = settings.observability.otlp_endpoint
    if endpoint:
        provider.add_span_processor(
            BatchSpanProcessor(
                OTLPSpanExporter(endpoint=endpoint, insecure=settings.observability.otlp_insecure)
            )
        )
        _log.info("exporting traces to %s", endpoint)
    else:
        # No processor at all rather than a console one: an unconfigured deployment
        # should cost nothing per span, and a log full of spans is not observability.
        _log.info("tracing is instrumented but not exported; no collector configured")
    trace.set_tracer_provider(provider)


@contextmanager
def span(name: str, attributes: Mapping[str, Any] | None = None) -> Iterator[Any]:
    """Open a span, and mark it failed if the block raises.

    Args:
        name: The span name.
        attributes: What to record on it. Values that are None are dropped, since an
            absent attribute and one explicitly set to nothing read the same downstream
            and only one of them is honest.

    Yields:
        The span, so a caller can add what it learns as it goes.
    """
    tracer = trace.get_tracer(_TRACER_NAME)
    with tracer.start_as_current_span(name) as current:
        for key, value in (attributes or {}).items():
            if value is not None:
                current.set_attribute(key, value)
        try:
            yield current
        except Exception as exc:
            current.set_status(Status(StatusCode.ERROR, type(exc).__name__))
            raise
