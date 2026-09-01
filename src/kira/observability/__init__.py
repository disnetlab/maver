"""Spans, and making sure every model call is counted."""

from kira.observability.tracing import configure_tracing, span
from kira.observability.usage import RecordingCompletion

__all__ = ["RecordingCompletion", "configure_tracing", "span"]
