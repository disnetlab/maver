"""Which tools each role may call.

The allowlist is deliberately not the whole story for write authority.
``exec_kubectl_cmd_safely`` is on every role's list, and the read/write split is enforced
a second time inside the approval gate, keyed on the stage. That duplication is on
purpose: a single misconfigured allowlist should not be enough to let a diagnostic role
mutate a cluster.

Order within a tuple is load-bearing. It is the order the model sees the tools declared
in, it was hand-tuned (namespace snapshot first for RCA, submit last everywhere), and
every published measurement was taken with it.
"""

__all__ = ["RETRIEVAL_TOOLS", "ROLE_TOOLSETS"]

RETRIEVAL_TOOLS = frozenset({"memory_search", "open_memory", "session_trace_summary"})
"""The tools that read the experience store.

Named here rather than inside toolset resolution because two independent places need
the answer to "which tools are the retrieval tools": the resolver, which withholds them
from a configuration that has no store or that keeps roles retrieval-free, and the
second pass's selector, which must never see a precedent in the live evidence it judges
two candidates on.
"""

ROLE_TOOLSETS: dict[str, tuple[str, ...]] = {
    "rca": (
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
        "bash_exec",
        "code_exec",
        "memory_search",
        "open_memory",
        "session_trace_summary",
        "submit_root_cause",
    ),
    "planning": (
        "memory_search",
        "open_memory",
        "session_trace_summary",
        "exec_kubectl_cmd_safely",
        "bash_exec",
        "code_exec",
        "submit_plan",
    ),
    "execution": (
        "exec_kubectl_cmd_safely",
        "bash_exec",
        "code_exec",
        "ssh_exec",
        "submit_execution_result",
    ),
    # Verify is read-only end-to-end: re-evaluate the alert/SLI, exercise the
    # request path (traces, logs), and judge system-level recovery. No
    # write/rollback tools.
    "verify": (
        "get_metrics",
        "get_alerts",
        "get_services",
        "get_traces",
        "get_logs",
        "exec_kubectl_cmd_safely",
        "bash_exec",
        "code_exec",
        "submit_verify_result",
    ),
    "learning": (
        "exec_kubectl_cmd_safely",
        "get_metrics",
        "get_logs",
        "get_labels",
        "get_label_values",
        "get_services",
        "memory_search",
        "open_memory",
        "session_trace_summary",
        "submit_learning_result",
    ),
    # Single-agent baseline: the full read+write kit in one toolset plus the
    # two-phase finalizers. Deliberately NO memory tools; this arm runs without
    # memory augmentation.
    "single": (
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
        "bash_exec",
        "code_exec",
        "submit_diagnosis",
        "submit_mitigation",
    ),
}
