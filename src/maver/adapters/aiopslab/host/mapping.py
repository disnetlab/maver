"""Pure encoders: incident artifacts in, one AIOpsLab answer out. No I/O.

These functions decide what AIOpsLab grades, they encode weeks of tuning against a
small local model, and they are carried from the prototype unchanged. Keeping them
free of the orchestrator, the bridge and the cluster is what lets every one of them be
pinned by a unit test on a laptop.

The one deliberate change is that :func:`build_alert` no longer attaches Verify's
connectivity contract: under MAVER the agent-side adapter supplies it from its own
configuration, so the credential-holding half has one fewer thing to know.
"""

import json
import re
from typing import Any

__all__ = ["build_alert", "submission_action"]

_TASKS = {"detection", "localization", "analysis", "mitigation"}

_TERMINAL_RUN_STATUSES = {"completed", "failed", "cancelled"}
_TERMINAL_LIFECYCLE_STATES = {"invalidated", "halted"}

_ANALYSIS_LEVELS = (
    "Hardware",
    "Operating System",
    "Virtualization",
    "Application",
)
_ANALYSIS_FAULTS = (
    "Misconfiguration",
    "Code Defect",
    "Authentication Issue",
    "Network/Storage Issue",
    "Operation Error",
    "Dependency Problem",
)


def _bounded_int(value: Any, *, default: int, low: int, high: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return min(high, max(low, parsed))


def _root_cause(record: dict[str, Any]) -> dict[str, Any] | None:
    value = record.get("root_cause")
    return value if isinstance(value, dict) else None


def _is_terminal_record(record: dict[str, Any]) -> bool:
    """Return true once the agent cannot make further progress on the run.

    Both clauses are load-bearing, and the second one looks like a bug against the
    SREGym driver's rule that only ``run_status`` may end a poll. It is not the same
    situation: AIOpsLab has no post-pipeline grading handshake to race, and a crash that
    lands between the lifecycle write and the run-status write leaves a session that is
    lifecycle-terminal and ``running`` forever. ``awaiting_human`` is in neither set, so
    a run parked on an approval keeps polling rather than being read as finished.
    """
    return (
        record.get("run_status") in _TERMINAL_RUN_STATUSES
        or record.get("status") in _TERMINAL_LIFECYCLE_STATES
    )


def build_alert(
    task_type: str,
    namespace: str,
    description: str,
    *,
    observation_window_start: str | None = None,
) -> dict[str, Any]:
    """Build blinded incident context; the problem ID never reaches the model.

    Args:
        task_type: The AIOpsLab task being graded.
        namespace: The active problem namespace.
        description: AIOpsLab's own problem description.
        observation_window_start: When fault injection began, if it was observed.

    Returns:
        The alert payload to post to the incident API.

    Raises:
        ValueError: If the task type is not one AIOpsLab grades.
    """
    if task_type not in _TASKS:
        raise ValueError(f"unsupported AIOpsLab task type: {task_type!r}")
    alert = {
        "alertname": "ServiceHealthDegraded",
        "signal_role": "generic_incident_trigger",
        "signal_instruction": (
            "This is an AIOpsLab task trigger, not a fault label. Discover the "
            "actual state from live telemetry; do not infer a cause from the trigger."
        ),
        "namespace": namespace,
        "benchmark": "aiopslab",
        "benchmark_task": task_type,
        "description": description,
    }
    if observation_window_start:
        alert["observation_window_start"] = observation_window_start
    return alert


def _is_no_anomaly(rc: dict[str, Any] | None) -> bool:
    if not rc:
        return False
    text = " ".join(
        str(rc.get(key) or "")
        for key in ("fault_type", "affected_resource", "summary", "halt_reason")
    ).lower()
    explicit_none = (
        str(rc.get("fault_type") or "").strip().lower() == "none"
        and str(rc.get("affected_resource") or "").strip().lower() == "none"
    )
    return (bool(rc.get("halt")) or explicit_none) and any(
        phrase in text for phrase in ("no anomaly", "no fault", "healthy", "normal", "none")
    )


def _component_from_root_cause(rc: dict[str, Any] | None) -> list[str]:
    if not rc or _is_no_anomaly(rc):
        return []
    value = str(rc.get("affected_resource") or "").strip().strip("/`")
    if not value or value.lower() in {"none", "unknown", "n/a"}:
        return []
    component = value.rsplit("/", 1)[-1].strip()
    component = re.split(r"\s+in\s+namespace\s+", component, maxsplit=1, flags=re.I)[0]
    # Prefer the owning stable workload when a generated pod name slipped into
    # the artefact despite the RootCause contract.
    component = re.sub(r"-[0-9a-f]{8,10}-[a-z0-9]{5}$", "", component)
    return [component] if component else []


def _explicit_analysis(text: str) -> tuple[str, str] | None:
    match = re.search(
        r"AIOpsLab classification:\s*system_level\s*=\s*([^;\n.]+)\s*;\s*"
        r"fault_type\s*=\s*([^\n.]+)",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    level_raw, fault_raw = (part.strip(" `") for part in match.groups())
    level = next((v for v in _ANALYSIS_LEVELS if v.lower() == level_raw.lower()), None)
    fault = next((v for v in _ANALYSIS_FAULTS if v.lower() == fault_raw.lower()), None)
    return (level, fault) if level and fault else None


def _analysis_from_root_cause(rc: dict[str, Any] | None) -> dict[str, str | None]:
    """Translate a diagnosis into AIOpsLab's taxonomy, and say who decided it.

    ``system_level`` and ``fault_type`` are the graded answer. ``source`` and ``rule``
    are the accounting the prototype lacked: two hand-written rules below override the
    model's own classification, so a reported analysis accuracy that does not separate
    ``model`` from ``normalized`` and ``fallback`` is reporting the rules' accuracy as
    the agent's.
    """
    text = " ".join(
        str((rc or {}).get(key) or "")
        for key in ("fault_type", "affected_resource", "summary", "halt_reason")
    )
    lowered = text.lower()
    resource = str((rc or {}).get("affected_resource") or "").lower()

    # AIOpsLab defines an operator mutation to an orchestrator-managed workload
    # (scale/delete/kill/restart) as a Virtualization-level Operation Error. This
    # is derivable from the diagnosed action/resource and intentionally takes
    # precedence over a model labelling every bad field "misconfiguration".
    kubernetes_workload = any(
        kind in resource
        for kind in ("deployment/", "statefulset/", "daemonset/", "pod/", "replicaset/")
    )
    replica_zero = bool(re.search(r"replicas?[^.;\n]{0,40}(?:\b0\b|\bzero\b)", lowered))
    operator_mutation = replica_zero or any(
        phrase in lowered
        for phrase in (
            "replicas: 0",
            "replicas set to 0",
            "scaled to 0",
            "scaled down to zero",
            "scale_pods_to_zero",
            "container kill",
            "pod kill",
            "deleted pod",
            "operator action",
        )
    )
    if kubernetes_workload and operator_mutation:
        return _answer("Virtualization", "Operation Error", "normalized", "operator_mutation")

    # Database users and their application roles are application-level state,
    # even though the database itself happens to run in a Kubernetes workload.
    # Normalize this before accepting an explicit model classification because
    # smaller models routinely mistake the hosting layer for the fault layer.
    database_auth = any(term in lowered for term in ("mongodb", "database", "admin user")) and any(
        term in lowered
        for term in ("auth", "privilege", "credential", "user removed", "unregistered")
    )
    if database_auth:
        return _answer("Application", "Authentication Issue", "normalized", "database_auth")

    if explicit := _explicit_analysis(text):
        return _answer(explicit[0], explicit[1], "model", None)

    if any(k in lowered for k in ("disk wore", "hardware", "device failure")):
        level = "Hardware"
    elif any(k in lowered for k in ("operating system", "kernel", "process limit")):
        level = "Operating System"
    elif any(
        k in lowered
        for k in (
            "kubernetes",
            "deployment/",
            "statefulset/",
            "daemonset/",
            "pod/",
            "container",
            "replica",
            "resource limit",
        )
    ):
        level = "Virtualization"
    else:
        level = "Application"

    if any(k in lowered for k in ("auth", "credential", "permission denied")):
        fault = "Authentication Issue"
    elif any(k in lowered for k in ("misconfig", "wrong ", "invalid ", "config", "image", "port")):
        fault = "Misconfiguration"
    elif any(k in lowered for k in ("code defect", "bug", "exception", "panic")):
        fault = "Code Defect"
    elif any(k in lowered for k in ("network", "storage", "disk", "latency", "packet loss", "i/o")):
        fault = "Network/Storage Issue"
    elif any(
        k in lowered for k in ("operation", "scaled", "replica", "deleted", "killed", "restart")
    ):
        fault = "Operation Error"
    else:
        fault = "Dependency Problem"
    return _answer(level, fault, "fallback", "keyword_ladder")


def _answer(level: str, fault: str, source: str, rule: str | None) -> dict[str, str | None]:
    return {"system_level": level, "fault_type": fault, "source": source, "rule": rule}


def _graded(answer: dict[str, str | None]) -> dict[str, str | None]:
    """Strip the accounting: AIOpsLab grades exactly these two keys."""
    return {"system_level": answer["system_level"], "fault_type": answer["fault_type"]}


def submission_action(task_type: str, record: dict[str, Any]) -> str:
    """Map persisted agent artefacts to exactly one AIOpsLab action.

    Args:
        task_type: The AIOpsLab task being graded.
        record: The terminal incident record read from the agent API.

    Returns:
        The fenced ``submit(...)`` call AIOpsLab parses as the agent's action.

    Raises:
        RuntimeError: If a detection episode produced no ``RootCause``. Guessing
            ``"Yes"`` on a crashed episode would score about half the detection tasks
            correct for free, so this stays a runner error the sweep may retry.
        ValueError: If the task type is not one AIOpsLab grades.
    """
    rc = _root_cause(record)
    if task_type == "detection":
        if rc is None:
            raise RuntimeError(
                "agent terminated without a RootCause; refusing to fabricate a detection answer"
            )
        payload: Any = "No" if _is_no_anomaly(rc) else "Yes"
        call = f"submit({json.dumps(payload)})"
    elif task_type == "localization":
        call = f"submit({json.dumps(_component_from_root_cause(rc))})"
    elif task_type == "analysis":
        call = f"submit({json.dumps(_graded(_analysis_from_root_cause(rc)))})"
    elif task_type == "mitigation":
        call = "submit()"
    else:
        raise ValueError(f"unsupported AIOpsLab task type: {task_type!r}")
    return f"```\n{call}\n```"
