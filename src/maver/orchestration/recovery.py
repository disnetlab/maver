"""Safety checks for opt-in recovery of interrupted active incidents.

A checkpoint says where execution began, not whether an external call completed before
the process died. Replaying execution could repeat a cluster mutation, so those runs are
left for operator reconciliation. The same rule applies to adapter-owned episodes: a
fresh adapter cannot recreate the old episode's transport or grading lifecycle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from maver.memory.ports import SessionRecord
from maver.pipelines import restore_selection

if TYPE_CHECKING:
    from maver.orchestration.runner import Deployment


_LOCAL_EXEC = frozenset({"bash_exec", "code_exec"})
_REPLAY_TOOLS = {
    "rca": frozenset(
        {
            "namespace_snapshot",
            "get_metrics",
            "get_alerts",
            "get_services",
            "get_operations",
            "get_traces",
            "get_dependency_graph",
            "get_logs",
            "get_labels",
            "get_label_values",
            "exec_kubectl_cmd_safely",
            "memory_search",
            "open_memory",
            "session_trace_summary",
            "submit_root_cause",
        }
    ),
    "planning": frozenset(
        {
            "memory_search",
            "open_memory",
            "session_trace_summary",
            "exec_kubectl_cmd_safely",
            "submit_plan",
        }
    ),
    "verify": frozenset(
        {
            "get_metrics",
            "get_alerts",
            "get_services",
            "get_traces",
            "get_logs",
            "exec_kubectl_cmd_safely",
            "submit_verify_result",
        }
    ),
}


def recovery_tool_drop(role: str, enabled: bool) -> frozenset[str]:
    """Remove local command tools from stages that may be replayed after a crash."""
    return _LOCAL_EXEC if enabled and role in _REPLAY_TOOLS else frozenset()


def safe_stage(record: SessionRecord, deployment: Deployment) -> str | None:
    """Return a replayable stage, or None when recovery must fail closed."""
    checkpoint = record.checkpoint or {}
    if checkpoint.get("kind") != "active_stage" or checkpoint.get("version") != 1:
        return None
    if checkpoint.get("approval_origin"):
        return None
    if checkpoint.get("adapter") or deployment.adapter is not None:
        return None
    state = checkpoint.get("state")
    if not isinstance(state, dict) or state.get("session_id") != record.session_id:
        return None
    runtime = state.get("runtime")
    if not isinstance(runtime, dict):
        return None
    if runtime.get("deployment_digest") != deployment.runtime_digest:
        return None
    if deployment.settings.authority.additional_write_stages:
        return None
    if deployment.tool_extras:
        return None
    if deployment.pipelines is None:
        return None
    try:
        pipeline = restore_selection(deployment.pipelines, runtime["selection"]).pipeline
        stage = str(checkpoint["stage"])
        role = pipeline.workflow.stage(stage).role
        tools = set(pipeline.agents[stage].tool_names)
    except (KeyError, TypeError, ValueError):
        return None
    # The runner withholds local command execution under this flag. A custom tool not
    # on the vetted read-only list has unknown side effects and cannot be replayed.
    allowed = _REPLAY_TOOLS.get(role)
    return stage if allowed is not None and tools <= allowed | _LOCAL_EXEC else None
