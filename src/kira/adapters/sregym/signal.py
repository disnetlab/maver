"""Turning the harness's own metadata into an incident signal, and back into an answer.

Both directions are pure functions over plain dictionaries, so the mapping that decides
what an agent is told and what a judge is shown can be exercised without a conductor.
Both are carried from the prototype and byte-compared against it in the tests, because
the first decides whether an arm is blind and the second is the text every published
diagnosis score was computed on.
"""

from typing import Any

__all__ = ["build_diagnosis_solution", "build_signal", "problem_id"]

# A real monitoring alert names a *symptom*, never the injected fault. The harness's
# problem id IS the ground-truth fault label (e.g. "k8s_target_port-misconfig"), so it
# must never reach the agent prompt: feeding it lets the diagnosis be read off the
# question and invalidates any comparison with a blind baseline, including the
# benchmark's own reference agents, which see only the app name, namespace and
# description.
_BLIND_ALERTNAME = "ServiceHealthDegraded"
_BLIND_SIGNAL_ROLE = "generic_incident_trigger"
_BLIND_SIGNAL_INSTRUCTION = (
    "This is a generic intake placeholder, not a real monitoring alert. "
    "Discover the concrete firing alert(s) and degraded SLI from live telemetry; "
    "do not use disappearance of this placeholder as recovery evidence."
)

_MAX_EVIDENCE = 5
_NO_DIAGNOSIS = "No diagnosis produced."


def problem_id(problem: Any) -> str | None:
    """Take an episode identifier out of whatever shape the conductor answered with.

    Args:
        problem: The conductor's problem blob, or None.

    Returns:
        The identifier, or None.
    """
    if isinstance(problem, dict):
        for key in ("problem_id", "id", "problem", "name", "alertname"):
            value = problem.get(key)
            if value:
                return str(value)
    if isinstance(problem, str):
        return problem.strip() or None
    return None


def build_signal(problem: Any, app: Any) -> dict[str, Any]:
    """Build the blinded intake trigger from app metadata alone.

    Nothing problem-derived reaches it except the namespace fallback, which is a
    scoping hint rather than a fault label. Three leak paths existed before this was
    made one function: the identifier lifted into the alert name, the whole problem blob
    serialised into the description, and the blob kept alongside the alert, with the
    diagnosing and planning roles rendering all of it into their prompts.

    Args:
        problem: The conductor's problem blob, or None.
        app: The conductor's app metadata.

    Returns:
        The signal, with absent fields dropped rather than carried as nulls.
    """
    p = problem if isinstance(problem, dict) else {}
    a = app if isinstance(app, dict) else {}

    app_name = a.get("app_name") or a.get("name")
    # Namespace is a scoping hint, not a fault label, so falling back to the problem's
    # namespace is fair; prefer the app's so we never lift a more specific problem field.
    namespace = a.get("namespace") or a.get("default_namespace") or p.get("namespace")

    if a.get("descriptions"):
        description = f"App {app_name or ''}: {a['descriptions']}".strip()
    elif app_name:
        description = f"App {app_name} is reporting degraded health."
    else:
        description = None

    alert = {
        "alertname": _BLIND_ALERTNAME,
        "signal_role": _BLIND_SIGNAL_ROLE,
        "signal_instruction": _BLIND_SIGNAL_INSTRUCTION,
        "namespace": namespace,
        "service": app_name,
        "app_name": app_name,
        "description": description,
    }
    return {k: v for k, v in alert.items() if v is not None}


def build_diagnosis_solution(root_cause: Any) -> str:
    """Render a diagnosis as the free text the harness's judge grades.

    It duck-types the single-agent configuration's unstructured diagnosis, whose text
    is passed through rather than reframed, exactly as the benchmark's own reference
    agents submit it. And it never returns empty: a halted or absent diagnosis is a
    *failed* diagnosis and must still advance the stage machine, because one that does
    not leaves no result row at all and the problem vanishes from the denominator,
    biasing the rate upward.

    Args:
        root_cause: The diagnosis artifact, its dictionary form, or None.

    Returns:
        The text to submit, never empty.
    """
    rc = root_cause.model_dump(mode="json") if hasattr(root_cause, "model_dump") else root_cause
    if not isinstance(rc, dict):
        return str(rc) if rc else _NO_DIAGNOSIS
    if "solution" in rc and "fault_type" not in rc:
        return str(rc.get("solution") or "").strip() or _NO_DIAGNOSIS
    fault = rc.get("fault_type") or "unknown"
    resource = rc.get("affected_resource") or "unknown"
    summary = rc.get("summary") or ""
    evidence = rc.get("evidence") or []
    parts = [f"Fault type: {fault}.", f"Affected resource: {resource}."]
    if summary:
        parts.append(summary)
    if isinstance(evidence, list) and evidence:
        parts.append("Evidence: " + "; ".join(str(e) for e in evidence[:_MAX_EVIDENCE]))
    return " ".join(parts).strip() or _NO_DIAGNOSIS
