"""Canonical tool registry and per-agent toolset builder."""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Awaitable, Callable, Literal

import yaml
from pydantic import BaseModel

from agent.llm import ToolDefinition
from agent.state import (
    ExecutionResult,
    LearningResult,
    Plan,
    RootCause,
    SingleDiagnosis,
    SingleResult,
    VerifyResult,
)
from agent.tools.aiopslab_recovery import restore_mongodb_access
from agent.tools.aiopslab_probe import probe_service
from agent.tools.code_exec import code_exec
from agent.tools.host_recovery import ssh_exec
from agent.tools.jaeger import get_dependency_graph, get_operations, get_services, get_traces
from agent.tools.kubectl import (
    exec_kubectl_cmd_safely,
    get_previous_rollbackable_cmd,
    rollback_command,
)
from agent.tools.loki import get_label_values, get_labels, get_logs
from agent.tools.memory import (
    memory_grep,
    memory_search,
    open_memory,
    session_trace_summary,
)
from agent.tools.playbook import playbook_append, playbook_read, playbook_reconfirm, playbook_remove
from agent.tools.prometheus import get_alerts, get_metrics
from agent.tools.shell import bash_exec
from agent.tools.topology import namespace_snapshot

ToolFn = Callable[..., Awaitable[str]]
_DESCRIPTIONS_PATH = Path(__file__).with_name("descriptions.yaml")


class ToolRegistryError(ValueError):
    """Raised when an agent requests an invalid toolset."""


def _load_description_config() -> dict:
    try:
        data = yaml.safe_load(_DESCRIPTIONS_PATH.read_text()) or {}
    except OSError as exc:
        raise ToolRegistryError(f"failed to read {_DESCRIPTIONS_PATH}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ToolRegistryError(f"failed to parse {_DESCRIPTIONS_PATH}: {exc}") from exc
    if not isinstance(data, dict):
        raise ToolRegistryError("tool descriptions must be a mapping")
    return data


_DESCRIPTION_CONFIG = _load_description_config()


def _normalise_schema(schema: dict) -> dict:
    """Make a Pydantic-generated JSON Schema safe across function-calling providers.

    Gemini's validator returns INTERNAL 500 on schemas with `$ref`/`$defs` and on
    `anyOf` wrappers used for nullable Optional[X] fields. We inline refs and
    collapse `anyOf: [{type: X}, {type: null}]` to a bare `{type: X}`. Optional
    fields remain optional because Pydantic already omits them from `required`,
    so the model can simply skip them when they don't apply (e.g. halt_reason).
    OpenAI and Anthropic both accept the simpler shape unchanged.
    """
    defs = schema.get("$defs", {})

    def walk(node):
        if isinstance(node, list):
            return [walk(x) for x in node]
        if not isinstance(node, dict):
            return node

        if "$ref" in node:
            ref = node["$ref"]
            if isinstance(ref, str) and ref.startswith("#/$defs/"):
                target = defs.get(ref[len("#/$defs/") :])
                if isinstance(target, dict):
                    merged = walk(target)
                    for k, v in node.items():
                        if k != "$ref":
                            merged[k] = walk(v)
                    return merged

        if isinstance(node.get("anyOf"), list):
            non_null = [
                b for b in node["anyOf"]
                if not (isinstance(b, dict) and b.get("type") == "null")
            ]
            if len(non_null) == 1 and len(non_null) != len(node["anyOf"]):
                merged = walk(non_null[0])
                for k, v in node.items():
                    if k == "anyOf":
                        continue
                    if k == "default" and v is None:
                        continue  # default=None is implied by the field being optional
                    merged[k] = walk(v) if k != "default" else v
                return merged

        return {k: walk(v) for k, v in node.items()}

    out = walk(schema)
    if isinstance(out, dict):
        out.pop("$defs", None)
    return out


def _model_schema(model: type[BaseModel]) -> dict:
    return _normalise_schema(model.model_json_schema())


def get_tool_descriptions(tool_name: str) -> dict[str, object]:
    """Return the full description block for a tool (default + per-agent variants).

    Used by the tool tester / catalog UI. Agents themselves continue to read
    their slice via `_description_for(tool, agent)`.
    """
    raw = _DESCRIPTION_CONFIG.get(tool_name)
    if not isinstance(raw, dict):
        return {"default": "", "agents": {}}
    default = raw.get("default")
    agents_raw = raw.get("agents") or {}
    agents = {
        k: v.strip()
        for k, v in agents_raw.items()
        if isinstance(v, str) and v.strip()
    } if isinstance(agents_raw, dict) else {}
    return {
        "default": default.strip() if isinstance(default, str) else "",
        "agents": agents,
    }


def _description_for(tool_name: str, agent: str) -> str:
    raw = _DESCRIPTION_CONFIG.get(tool_name)
    if not isinstance(raw, dict):
        raise ToolRegistryError(f"missing description entry for tool '{tool_name}'")

    agents = raw.get("agents", {})
    if agents is not None and not isinstance(agents, dict):
        raise ToolRegistryError(f"tool '{tool_name}' agents descriptions must be a mapping")

    description = (agents or {}).get(agent) or raw.get("default")
    if not isinstance(description, str) or not description.strip():
        raise ToolRegistryError(
            f"missing non-empty description for tool '{tool_name}' and agent '{agent}'"
        )
    return description.strip()


# Classification used by the tool tester / catalog UI to decide what may be
# invoked from outside an agent loop. Adding a new tool requires picking one
# of these explicitly — keeps the registry the single source of truth.
ToolSafety = Literal[
    "read_only",              # queries telemetry / memory / playbooks; safe to invoke ad hoc
    "cluster_write",          # can mutate the cluster or run arbitrary code
    "shared_state_write",     # mutates shared persistent state (playbooks, memory)
    "agent_loop_finalizer",   # submit_* terminal tools; only meaningful inside an agent loop
]


@dataclass(frozen=True)
class ToolSpec:
    definition: ToolDefinition
    safety: ToolSafety
    dispatch: ToolFn | None = None

    def definition_for(self, agent: str) -> ToolDefinition:
        if self.definition.name not in AGENT_TOOLSETS.get(agent, ()):
            raise ToolRegistryError(
                f"tool '{self.definition.name}' is not allowed for agent '{agent}'"
            )
        return replace(
            self.definition,
            description=_description_for(self.definition.name, agent),
        )


TOOL_REGISTRY: dict[str, ToolSpec] = {
    "bash_exec": ToolSpec(
        definition=ToolDefinition(
            name="bash_exec",
            description="",
            parameters={
                "type": "object",
                "properties": {
                    "cmd": {"type": "string", "description": "The bash command to run."}
                },
                "required": ["cmd"],
            },
        ),
        safety="cluster_write",
        dispatch=bash_exec,
        # Learning is a read-only distiller with dedicated tools for everything it
        # needs (playbook_*, memory_*, get_*, exec_kubectl_cmd_safely). General
        # shell/python is not needed and let small models wander off-task — e.g.
        # `bash_exec cat /workspace/playbooks/rca.md` instead of `playbook_read`
        # (playbooks are not on disk in postgres mode), spinning until rescue.
        # Verify gets it to actively probe the request path (kubectl exec curl in
        # proxy_exec mode); the proxy shim blocks writes for the verify stage.
    ),
    "code_exec": ToolSpec(
        definition=ToolDefinition(
            name="code_exec",
            description="",
            parameters={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python source code to execute.",
                    },
                    "language": {
                        "type": "string",
                        "description": "Programming language. Currently only 'python' is supported.",
                    },
                },
                "required": ["code", "language"],
            },
        ),
        safety="cluster_write",
        dispatch=code_exec,
        # See bash_exec note: learning does not get general shell/python.
    ),
    "ssh_exec": ToolSpec(
        definition=ToolDefinition(
            name="ssh_exec",
            description="",
            parameters={
                "type": "object",
                "properties": {
                    "node": {
                        "type": "string",
                        "description": "Configured SSH host alias for the worker node.",
                    },
                    "command": {
                        "type": "string",
                        "description": "Command to run on the allowlisted worker host.",
                    },
                },
                "required": ["node", "command"],
            },
        ),
        safety="cluster_write",
        dispatch=ssh_exec,
    ),
    "exec_kubectl_cmd_safely": ToolSpec(
        definition=ToolDefinition(
            name="exec_kubectl_cmd_safely",
            description="",
            parameters={
                "type": "object",
                "properties": {
                    "cmd": {
                        "type": "string",
                        "description": (
                            "Full kubectl command, e.g. \"kubectl get pods -n shop\". "
                            "Must start with \"kubectl\". All-namespace (-A), API discovery, "
                            "and credential/context override flags are blocked."
                        ),
                    },
                    "save_as": {
                        "type": "string",
                        "description": (
                            "Optional filename (e.g. \"pods.json\"). Use only when the "
                            "output may be long or you need to reuse/post-process the full "
                            "result. When set, the full output is written to "
                            "tool_outputs/<save_as> in the session workspace and a short "
                            "head + path is returned. Read that file with code_exec or "
                            "bash_exec to filter/parse/compute over it — never run kubectl "
                            "inside code_exec/bash_exec (they have no cluster credentials)."
                        ),
                    },
                },
                "required": ["cmd"],
            },
        ),
        safety="cluster_write",
        dispatch=exec_kubectl_cmd_safely,
    ),
    "rollback_command": ToolSpec(
        definition=ToolDefinition(
            name="rollback_command",
            description="",
            parameters={"type": "object", "properties": {}, "required": []},
        ),
        safety="cluster_write",
        dispatch=rollback_command,
    ),
    "restore_mongodb_access": ToolSpec(
        definition=ToolDefinition(
            name="restore_mongodb_access",
            description="",
            parameters={
                "type": "object",
                "properties": {
                    "service": {
                        "type": "string",
                        "enum": ["mongodb-rate", "mongodb-geo"],
                        "description": "MongoDB service whose application access is broken.",
                    },
                    "access": {
                        "type": "string",
                        "enum": ["privileges", "user"],
                        "description": (
                            "Use 'privileges' for a revoked admin role and 'user' "
                            "for a removed/unregistered admin user."
                        ),
                    },
                },
                "required": ["service", "access"],
            },
        ),
        safety="cluster_write",
        dispatch=restore_mongodb_access,
    ),
    "probe_service": ToolSpec(
        definition=ToolDefinition(
            name="probe_service",
            description="",
            parameters={
                "type": "object",
                "properties": {
                    "service": {
                        "type": "string",
                        "description": "Service name in the active incident namespace.",
                    },
                    "path": {
                        "type": "string",
                        "description": "Absolute HTTP path to request, for example '/'.",
                    },
                    "port": {
                        "type": "integer",
                        "description": "Optional declared Service port; omit to use the bridge's HTTP-oriented default.",
                    },
                },
                "required": ["service", "path"],
            },
        ),
        safety="read_only",
        dispatch=probe_service,
    ),
    "get_previous_rollbackable_cmd": ToolSpec(
        definition=ToolDefinition(
            name="get_previous_rollbackable_cmd",
            description="",
            parameters={"type": "object", "properties": {}, "required": []},
        ),
        safety="read_only",
        dispatch=get_previous_rollbackable_cmd,
    ),
    "namespace_snapshot": ToolSpec(
        definition=ToolDefinition(
            name="namespace_snapshot",
            description="",
            parameters={
                "type": "object",
                "properties": {
                    "namespace": {
                        "type": "string",
                        "description": "Namespace to snapshot, e.g. 'shop'.",
                    },
                    "since_minutes": {
                        "type": "integer",
                        "description": "Lookback window for recent changes/events (default 30).",
                    },
                },
                "required": ["namespace"],
            },
        ),
        safety="read_only",
        dispatch=namespace_snapshot,
    ),
    "get_metrics": ToolSpec(
        definition=ToolDefinition(
            name="get_metrics",
            description="",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "PromQL expression."},
                },
                "required": ["query"],
            },
        ),
        safety="read_only",
        dispatch=get_metrics,
    ),
    "get_alerts": ToolSpec(
        definition=ToolDefinition(
            name="get_alerts",
            description="",
            parameters={"type": "object", "properties": {}, "required": []},
        ),
        safety="read_only",
        dispatch=get_alerts,
    ),
    "get_services": ToolSpec(
        definition=ToolDefinition(
            name="get_services",
            description="",
            parameters={"type": "object", "properties": {}, "required": []},
        ),
        safety="read_only",
        dispatch=get_services,
    ),
    "get_operations": ToolSpec(
        definition=ToolDefinition(
            name="get_operations",
            description="",
            parameters={
                "type": "object",
                "properties": {
                    "service": {
                        "type": "string",
                        "description": "Service name whose operations to list.",
                    },
                },
                "required": ["service"],
            },
        ),
        safety="read_only",
        dispatch=get_operations,
    ),
    "get_traces": ToolSpec(
        definition=ToolDefinition(
            name="get_traces",
            description="",
            parameters={
                "type": "object",
                "properties": {
                    "service": {
                        "type": "string",
                        "description": "Jaeger service name (e.g. 'frontend').",
                    },
                    "last_n_minutes": {
                        "type": "integer",
                        "description": "Lookback window in minutes.",
                    },
                },
                "required": ["service", "last_n_minutes"],
            },
        ),
        safety="read_only",
        dispatch=get_traces,
    ),
    "get_dependency_graph": ToolSpec(
        definition=ToolDefinition(
            name="get_dependency_graph",
            description="",
            parameters={
                "type": "object",
                "properties": {
                    "last_n_minutes": {
                        "type": "integer",
                        "description": "Lookback window in minutes (default 30).",
                    },
                },
                "required": [],
            },
        ),
        safety="read_only",
        dispatch=get_dependency_graph,
    ),
    "get_logs": ToolSpec(
        definition=ToolDefinition(
            name="get_logs",
            description="",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Valid LogQL expression."},
                    "last_n_minutes": {
                        "type": "integer",
                        "description": "Lookback window in minutes (default 15).",
                    },
                },
                "required": ["query"],
            },
        ),
        safety="read_only",
        dispatch=get_logs,
    ),
    "get_labels": ToolSpec(
        definition=ToolDefinition(
            name="get_labels",
            description="",
            parameters={"type": "object", "properties": {}, "required": []},
        ),
        safety="read_only",
        dispatch=get_labels,
    ),
    "get_label_values": ToolSpec(
        definition=ToolDefinition(
            name="get_label_values",
            description="",
            parameters={
                "type": "object",
                "properties": {
                    "label": {
                        "type": "string",
                        "description": "Label name to list values for (e.g. 'namespace', 'app', 'pod').",
                    },
                },
                "required": ["label"],
            },
        ),
        safety="read_only",
        dispatch=get_label_values,
    ),
    "memory_search": ToolSpec(
        definition=ToolDefinition(
            name="memory_search",
            description="",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Concrete symptoms and signatures you have observed, e.g. "
                            "\"frontend pod CrashLoopBackOff, liveness probe HTTP 8080 "
                            "connection refused, container listening on 5000\". Use exact "
                            "error strings, resource kinds and observed values. Do not "
                            "narrate the whole incident: long descriptive queries match "
                            "fewer relevant precedents than short signature-heavy ones."
                        ),
                    },
                    "n_results": {"type": "integer"},
                    "scope": {
                        "type": "string",
                        "enum": ["evidence", "lesson"],
                        "description": "Search observations (evidence) or causal/action learning (lesson). Defaults to evidence.",
                    },
                    "only_verified": {"type": "boolean"},
                    "alert_name": {
                        "type": "string",
                        "description": "Optional. Current incident's alert name for metadata-match boosting.",
                    },
                    "namespace": {
                        "type": "string",
                        "description": "Optional. Current incident's namespace for metadata-match boosting.",
                    },
                    "fault_category": {
                        "type": "string",
                        "description": "Optional. Suspected fault category (e.g. RootCause.fault_type) for boosting.",
                    },
                    "affected_resource": {
                        "type": "string",
                        "description": "Optional. Affected Kubernetes resource for boosting.",
                    },
                },
                "required": ["query"],
            },
        ),
        safety="read_only",
        dispatch=memory_search,
    ),
    "memory_grep": ToolSpec(
        definition=ToolDefinition(
            name="memory_grep",
            description="",
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": (
                            "Case-insensitive regular expression over past incident "
                            "observations, e.g. 'CrashLoopBackOff' or 'targetPort|nodePort'. "
                            "Prefer exact identifiers you have observed."
                        ),
                    },
                    "n_results": {"type": "integer"},
                },
                "required": ["pattern"],
            },
        ),
        safety="read_only",
        dispatch=memory_grep,
    ),
    "open_memory": ToolSpec(
        definition=ToolDefinition(
            name="open_memory",
            description="",
            parameters={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "session_id from a prior memory_search result.",
                    },
                    "scope": {
                        "type": "string",
                        "enum": ["evidence", "lesson"],
                    },
                },
                "required": ["session_id"],
            },
        ),
        safety="read_only",
        dispatch=open_memory,
    ),
    "session_trace_summary": ToolSpec(
        definition=ToolDefinition(
            name="session_trace_summary",
            description="",
            parameters={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string"},
                    "agent": {
                        "type": "string",
                        "enum": ["rca", "planning", "execution"],
                    },
                    "scope": {
                        "type": "string",
                        "enum": ["evidence", "lesson"],
                    },
                },
                "required": ["session_id"],
            },
        ),
        safety="read_only",
        dispatch=session_trace_summary,
    ),
    "playbook_read": ToolSpec(
        definition=ToolDefinition(
            name="playbook_read",
            description="",
            parameters={
                "type": "object",
                "properties": {
                    "playbook": {
                        "type": "string",
                        "description": "Playbook to read: 'global', 'rca', 'planning', or 'execution'.",
                    }
                },
                "required": ["playbook"],
            },
        ),
        safety="read_only",
        dispatch=playbook_read,
    ),
    "playbook_append": ToolSpec(
        definition=ToolDefinition(
            name="playbook_append",
            description="",
            parameters={
                "type": "object",
                "properties": {
                    "playbook": {
                        "type": "string",
                        "description": "Target playbook: 'global', 'rca', 'planning', or 'execution'.",
                    },
                    "section": {
                        "type": "string",
                        "description": "Section heading to append under (created if absent).",
                    },
                    "content": {
                        "type": "string",
                        "description": "Bullet content. Be specific and concrete — no vague generalisations.",
                    },
                },
                "required": ["playbook", "section", "content"],
            },
        ),
        safety="shared_state_write",
        dispatch=playbook_append,
    ),
    "playbook_reconfirm": ToolSpec(
        definition=ToolDefinition(
            name="playbook_reconfirm",
            description="",
            parameters={
                "type": "object",
                "properties": {
                    "playbook": {
                        "type": "string",
                        "description": "Playbook containing the bullet.",
                    },
                    "bullet_id": {
                        "type": "string",
                        "description": "Bullet ID to reconfirm, e.g. '[rca-0003]'.",
                    },
                },
                "required": ["playbook", "bullet_id"],
            },
        ),
        safety="shared_state_write",
        dispatch=playbook_reconfirm,
    ),
    "playbook_remove": ToolSpec(
        definition=ToolDefinition(
            name="playbook_remove",
            description="",
            parameters={
                "type": "object",
                "properties": {
                    "playbook": {
                        "type": "string",
                        "description": "Playbook containing the bullet.",
                    },
                    "bullet_id": {
                        "type": "string",
                        "description": "Bullet ID to remove, e.g. '[rca-0003]'.",
                    },
                },
                "required": ["playbook", "bullet_id"],
            },
        ),
        safety="shared_state_write",
        dispatch=playbook_remove,
    ),
    "submit_learning_result": ToolSpec(
        definition=ToolDefinition(
            name="submit_learning_result",
            description="",
            parameters=_model_schema(LearningResult),
        ),
        safety="agent_loop_finalizer",
    ),
    "submit_root_cause": ToolSpec(
        definition=ToolDefinition(
            name="submit_root_cause",
            description="",
            parameters=_model_schema(RootCause),
        ),
        safety="agent_loop_finalizer",
    ),
    "submit_plan": ToolSpec(
        definition=ToolDefinition(
            name="submit_plan",
            description="",
            parameters=_model_schema(Plan),
        ),
        safety="agent_loop_finalizer",
    ),
    "submit_execution_result": ToolSpec(
        definition=ToolDefinition(
            name="submit_execution_result",
            description="",
            parameters=_model_schema(ExecutionResult),
        ),
        safety="agent_loop_finalizer",
    ),
    "submit_verify_result": ToolSpec(
        definition=ToolDefinition(
            name="submit_verify_result",
            description="",
            parameters=_model_schema(VerifyResult),
        ),
        safety="agent_loop_finalizer",
    ),
    "submit_diagnosis": ToolSpec(
        definition=ToolDefinition(
            name="submit_diagnosis",
            description="",
            parameters=_model_schema(SingleDiagnosis),
        ),
        safety="agent_loop_finalizer",
    ),
    "submit_mitigation": ToolSpec(
        definition=ToolDefinition(
            name="submit_mitigation",
            description="",
            parameters=_model_schema(SingleResult),
        ),
        safety="agent_loop_finalizer",
    ),
}


AGENT_TOOLSETS: dict[str, tuple[str, ...]] = {
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
        "memory_grep",
        "open_memory",
        "session_trace_summary",
        "submit_root_cause",
    ),
    "planning": (
        "memory_search",
        "memory_grep",
        "open_memory",
        "session_trace_summary",
        "exec_kubectl_cmd_safely",
        "bash_exec",
        "code_exec",
        "submit_plan",
    ),
    "execution": (
        "exec_kubectl_cmd_safely",
        "restore_mongodb_access",
        "bash_exec",
        "code_exec",
        "ssh_exec",
        "rollback_command",
        "get_previous_rollbackable_cmd",
        "submit_execution_result",
    ),
    # Verify is read-only end-to-end: re-evaluate the alert/SLI, exercise the
    # request path (traces, logs, and the AIOpsLab-safe Service probe where
    # applicable), and judge system-level recovery. No write/rollback tools.
    "verify": (
        "get_metrics",
        "get_alerts",
        "get_services",
        "get_traces",
        "get_logs",
        "exec_kubectl_cmd_safely",
        "bash_exec",
        "code_exec",
        "probe_service",
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
        "memory_grep",
        "open_memory",
        "session_trace_summary",
        "playbook_read",
        "playbook_append",
        "playbook_reconfirm",
        "playbook_remove",
        "submit_learning_result",
    ),
    # Single-agent baseline (AGENT_MODE=single): the full read+write kit in one
    # toolset — every telemetry read, kubectl (read and write), shell/code, and
    # rollback — plus the two-phase finalizers. Deliberately NO memory or playbook
    # tools: this arm runs without memory augmentation. See agent.agents.single.
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
        "rollback_command",
        "get_previous_rollbackable_cmd",
        "submit_diagnosis",
        "submit_mitigation",
    ),
}


# MCP server backing each cluster/telemetry tool. These dispatch through
# agent.tools.mcp_client.call_mcp_tool(<server>, ...) against SREGym's MCP servers.
# Tools absent from this map run in-process (memory, playbooks, submit_* finalizers).
TOOL_MCP_SERVER: dict[str, str] = {
    "exec_kubectl_cmd_safely": "kubectl",
    "rollback_command": "kubectl",
    "get_previous_rollbackable_cmd": "kubectl",
    "get_metrics": "prometheus",
    "get_alerts": "prometheus",
    "get_services": "jaeger",
    "get_operations": "jaeger",
    "get_traces": "jaeger",
    "get_dependency_graph": "jaeger",
    "get_logs": "loki",
    "get_labels": "loki",
    "get_label_values": "loki",
}


def mcp_server_for(name: str) -> str | None:
    """The MCP server a tool routes through, or None for in-process tools."""
    return TOOL_MCP_SERVER.get(name)


def agents_for_tool(name: str) -> tuple[str, ...]:
    """Agents whose canonical allowlist contains ``name``, in registry order."""
    if name not in TOOL_REGISTRY:
        raise ToolRegistryError(f"unknown tool '{name}'")
    return tuple(agent for agent, names in AGENT_TOOLSETS.items() if name in names)


def _validate_registry() -> None:
    registered_tools = set(TOOL_REGISTRY)
    described_tools = set(_DESCRIPTION_CONFIG)

    unknown_descriptions = described_tools - registered_tools
    if unknown_descriptions:
        names = ", ".join(sorted(unknown_descriptions))
        raise ToolRegistryError(f"descriptions.yaml has unknown tool descriptions: {names}")

    missing_descriptions = registered_tools - described_tools
    if missing_descriptions:
        names = ", ".join(sorted(missing_descriptions))
        raise ToolRegistryError(f"descriptions.yaml is missing tool descriptions: {names}")

    unknown_mcp = set(TOOL_MCP_SERVER) - registered_tools
    if unknown_mcp:
        names = ", ".join(sorted(unknown_mcp))
        raise ToolRegistryError(f"TOOL_MCP_SERVER references unknown tools: {names}")

    for name, raw in _DESCRIPTION_CONFIG.items():
        if not isinstance(raw, dict):
            raise ToolRegistryError(f"tool '{name}' description entry must be a mapping")
        default = raw.get("default")
        if not isinstance(default, str) or not default.strip():
            raise ToolRegistryError(f"tool '{name}' must have a non-empty default description")

        agents = raw.get("agents", {})
        if agents is None:
            agents = {}
        if not isinstance(agents, dict):
            raise ToolRegistryError(f"tool '{name}' agents descriptions must be a mapping")
        allowed_agents = set(agents_for_tool(name))
        unknown_agents = set(agents) - allowed_agents
        if unknown_agents:
            agent_names = ", ".join(sorted(unknown_agents))
            raise ToolRegistryError(
                f"tool '{name}' has descriptions for disallowed agents: {agent_names}"
            )
        for agent, description in agents.items():
            if not isinstance(description, str) or not description.strip():
                raise ToolRegistryError(
                    f"tool '{name}' agent '{agent}' description must be non-empty"
                )

    for agent, names in AGENT_TOOLSETS.items():
        for name in names:
            spec = TOOL_REGISTRY.get(name)
            if spec is None:
                raise ToolRegistryError(f"agent '{agent}' references unknown tool '{name}'")
            spec.definition_for(agent)
            if spec.dispatch is None and not name.startswith("submit_"):
                raise ToolRegistryError(
                    f"agent '{agent}' tool '{name}' has no dispatch function"
                )


_validate_registry()


# Every non-finalizer tool gets a required `rationale` argument injected into its
# schema. The model writes one short line of working state ("leading hypothesis /
# what's confirmed / what this call tests") that rides INSIDE the tool call. Unlike
# a visible-content status line — which reasoning models (Qwen3) keep in their hidden
# <think> channel and the loop then discards — tool-call arguments are always echoed
# back into the conversation (LLMClient.serialize_assistant_turn), so this trace
# genuinely survives across turns. The loop strips it before dispatch (_run_one_tool).
# submit_* finalizers are excluded: they validate against a strict Pydantic model and
# already carry their own summary field.
RATIONALE_ARG = "rationale"

_RATIONALE_PROPERTY = {
    "type": "string",
    "description": (
        "REQUIRED. One short sentence of your current working state BEFORE this call: "
        "your leading hypothesis, what you have established or ruled out, and what this "
        "call is meant to test. Your hidden reasoning is NOT retained between turns — "
        "this argument is the trace future turns see. One sentence; do not restate full "
        "reasoning or paste tool output."
    ),
}


def _with_rationale(definition: ToolDefinition) -> ToolDefinition:
    """Return a copy of `definition` with a required `rationale` property prepended.

    `definition_for` shares the registry's `parameters` dict by reference, so the
    schema is rebuilt rather than mutated in place.
    """
    params = definition.parameters or {}
    properties = params.get("properties") or {}
    if RATIONALE_ARG in properties:
        return definition
    new_params = {
        **params,
        "type": params.get("type", "object"),
        "properties": {RATIONALE_ARG: dict(_RATIONALE_PROPERTY), **properties},
        "required": [RATIONALE_ARG, *(params.get("required") or [])],
    }
    return replace(definition, parameters=new_params)


def build_toolset(
    agent: str,
    names: tuple[str, ...] | list[str] | None = None,
) -> tuple[list[ToolDefinition], dict[str, ToolFn]]:
    """Build model-facing definitions and dispatch map for an agent."""
    requested = tuple(names) if names is not None else AGENT_TOOLSETS.get(agent)
    if requested is None:
        raise ToolRegistryError(f"unknown agent '{agent}'")

    definitions: list[ToolDefinition] = []
    dispatch: dict[str, ToolFn] = {}
    for name in requested:
        spec = TOOL_REGISTRY.get(name)
        if spec is None:
            raise ToolRegistryError(f"unknown tool '{name}'")
        definition = spec.definition_for(agent)
        if spec.safety != "agent_loop_finalizer":
            definition = _with_rationale(definition)
        definitions.append(definition)
        if spec.dispatch is not None:
            dispatch[definition.name] = spec.dispatch
    return definitions, dispatch


TOOL_DEFINITIONS: list[ToolDefinition] = [
    replace(spec.definition, description=_DESCRIPTION_CONFIG[name]["default"].strip())
    for name, spec in TOOL_REGISTRY.items()
    if spec.dispatch is not None
]

TOOL_DISPATCH: dict[str, ToolFn] = {
    name: spec.dispatch
    for name, spec in TOOL_REGISTRY.items()
    if spec.dispatch is not None
}
