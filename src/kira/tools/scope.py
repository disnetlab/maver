"""What the incident itself says, as opposed to what the model asked for.

The namespace and the observation window arrive with the alert. They are trusted
context, not tool arguments, and the difference is the whole point: telemetry wrappers
use the namespace to keep one incident's evidence out of another's, so a model that
could set it could also read its way out of its own incident.

The prototype held this in a context variable bound per task, and never released the
token — only the offline learning pass reset it — so the last incident's namespace
leaked into whatever ran next on that task. Under one incident per process that was
harmless, which is why it survived. Here it is a value on the tool context, so unbinding
is structural: when the run ends, the object goes with it.

It also carries one short-lived cache, for a lookup that would otherwise be repeated by
every tool call that needs it and that cannot change within a run.
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

__all__ = ["IncidentScope"]

_NAMESPACE_RE = re.compile(r"^[a-z0-9][a-z0-9.-]{0,251}$")


@dataclass
class IncidentScope:
    """One incident's trusted context, and the discovery it caches."""

    namespace: str | None = None
    observation_window_start: str | None = None
    """When the incident began, ISO-8601, as the intake reported it."""
    trace_service_candidates: frozenset[str] | None = None
    """The workload names this namespace can legitimately report traces under.

    Derived from one cluster read and cached for the run. Mutable, unlike everything
    else here, because it is a cache rather than a fact about the incident; None means
    it has not been looked up yet, and an empty set means the namespace genuinely has
    no workloads.
    """

    @classmethod
    def from_alert(cls, alert: Mapping[str, Any] | None) -> "IncidentScope":
        """Read the trusted context out of an alert payload.

        Both fields are validated rather than taken as given. An unparseable value
        yields None, which every reader treats as "no scope" and therefore as a reason
        to withhold rather than to widen — a malformed namespace must not become a
        wildcard.

        Args:
            alert: The incident's alert payload.

        Returns:
            The scope for this incident.
        """
        return cls(
            namespace=_namespace(alert),
            observation_window_start=_window_start(alert),
        )


def _namespace(alert: Mapping[str, Any] | None) -> str | None:
    if not isinstance(alert, Mapping):
        return None
    raw_labels = alert.get("labels")
    labels: Mapping[str, Any] = raw_labels if isinstance(raw_labels, Mapping) else {}
    raw = alert.get("namespace") or labels.get("namespace") or labels.get("exported_namespace")
    namespace = str(raw or "").strip()
    return namespace if _NAMESPACE_RE.fullmatch(namespace) else None


def _window_start(alert: Mapping[str, Any] | None) -> str | None:
    if not isinstance(alert, Mapping):
        return None
    raw = alert.get("observation_window_start")
    if not isinstance(raw, str) or not raw.strip():
        return None
    value = raw.strip()
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return value
