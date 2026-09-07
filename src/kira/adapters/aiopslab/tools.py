"""The two tools that exist only against the AIOpsLab bridge.

Both are narrow on purpose. The agent container holds no kubeconfig, and the bridge
refuses generic ``kubectl exec`` because an arbitrary in-container command cannot be
classified as read-only — so a capability that genuinely needs one is expressed as a
tool with a closed argument set rather than as a shell. Probing asks the bridge to
port-forward one Service in the active namespace and issue one HTTP request through it;
the MongoDB repair asks it to run one of four shipped recovery scripts. Neither can run
a command of the model's choosing.

They are supplied by the adapter rather than registered in the core because under any
other environment they are simply absent, which is honest, where a tool that existed and
always failed reads to a model as a fault to work around. Their descriptions travel with
them for the same reason their allowlist entries do not exist.

One thing that was code in the prototype is structure here. ``restore_mongodb_access``
called the approval gate itself (``agent/tools/aiopslab_recovery.py:28-33``); in Kira the
gate wraps every write-capable call, so the tool declares that it writes and which stage
may issue one, and the gate holds it exactly as it holds a ``kubectl delete``. The
tool's own stage check stays, because a gate that was misconfigured or absent must not be
enough to let a diagnostic role mutate a database.
"""

import re
from collections.abc import Mapping
from typing import Any, Protocol, cast

from kira.adapters.aiopslab.config import ADAPTER_NAME, AiopsLabConfig
from kira.adapters.base import AdapterTool
from kira.tools.context import ToolContext
from kira.tools.policy import always_writes, read_only
from kira.tools.registry import ToolSpec

__all__ = ["adapter_tools"]

_SERVICE_NAME = re.compile(r"[a-z0-9](?:[-a-z0-9]{0,61}[a-z0-9])?$")
_MAX_PATH_CHARS = 256

_PROBE_ROLES = frozenset({"verify"})
_RECOVERY_ROLES = frozenset({"execution"})
"""Who each tool is offered to. Verification is read-only end to end and remediation is
the sole writer, so neither set has a second member."""

_BRIDGE = frozenset({"kubernetes"})
"""What the bridge must serve for these to be offered.

Both reach it through the same authenticated action endpoint the cluster tools use, and
``kubernetes`` is the capability that endpoint stands for. Naming a word outside the
declared vocabulary would silently disable them everywhere, which reads in the results
exactly like a model that chose not to call them.
"""

_PROBE_PARAMETERS: dict[str, Any] = {
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
            "description": (
                "Optional declared Service port; omit to use the bridge's HTTP-oriented default."
            ),
        },
    },
    "required": ["service", "path"],
}

_PROBE_DESCRIPTION = {
    "probe_service": {
        "default": (
            "AIOpsLab-only read-only HTTP probe. The bridge port-forwards one named "
            "Service in the active incident namespace and returns its HTTP status; it "
            "cannot execute an in-pod command or modify the cluster."
        ),
        "agents": {
            "verify": (
                "AIOpsLab Verify-only independent HTTP check. The bridge port-forwards "
                "the named Service in this incident's namespace, requests the absolute "
                "path, and returns the actual HTTP status. It cannot execute a command "
                "in a pod or modify the cluster. Choose the user-facing Service "
                "discovered from live Kubernetes evidence. Give its declared port only "
                "when the Service exposes more than one port; otherwise omit port."
            )
        },
    }
}

_RECOVERY_DESCRIPTION = {
    "restore_mongodb_access": {
        "default": (
            "AIOpsLab-only recovery for MongoDB application access. Select mongodb-rate "
            "or mongodb-geo and whether the diagnosed fault removed the admin user's "
            "privileges or removed/unregistered the user. The bridge runs only the "
            "fixed recovery script shipped for that combination and restarts the "
            "corresponding application Deployment. This is available only to Execution "
            "and always goes through runtime enforcement; call it directly rather than "
            "requesting approval. It is not a general shell or kubectl-exec facility."
        )
    }
}


class _Bridge(Protocol):
    """The one transport method these two tools need that the protocol has no word for.

    Structural rather than an import of the transport module: neither tool is offered
    unless this adapter bound its own transport, and naming the class here would put the
    bridge client on the import path of a process that only wanted the tool table.
    """

    async def action(
        self,
        server: str,
        tool: str,
        arguments: Mapping[str, Any] | None = None,
        *,
        structured: bool = False,
        stage: str | None = None,
    ) -> str: ...


async def _probe_service(ctx: ToolContext, service: str, path: str, port: int | None = None) -> str:
    """Issue one HTTP request through a Service in the active AIOpsLab namespace."""
    # Enforced twice, as every write is: the gate keys on the stage, and so does the
    # tool, because a gate that was misconfigured or absent must not be enough to let
    # another role open a request path of its choosing.
    if ctx.role not in _PROBE_ROLES:
        return "error: probe_service is permitted only for the verify agent"
    if not _SERVICE_NAME.fullmatch(service):
        return "error: service must be a DNS-1123 Service name"
    if (
        not path.startswith("/")
        or path.startswith("//")
        or len(path) > _MAX_PATH_CHARS
        or "://" in path
        or "\r" in path
        or "\n" in path
    ):
        return "error: path must be a bounded absolute HTTP path"
    if port is not None and (isinstance(port, bool) or not 1 <= port <= 65535):
        return "error: port must be an integer in 1..65535"
    arguments: dict[str, Any] = {"service": service, "path": path}
    if port is not None:
        arguments["port"] = port
    return await cast("_Bridge", ctx.transport).action(
        ADAPTER_NAME, "probe_service", arguments, stage="verify"
    )


def _restore_mongodb_access(config: AiopsLabConfig) -> Any:
    """Build the recovery dispatch against the pairs this deployment configured.

    Args:
        config: The deployment's adapter configuration.

    Returns:
        The dispatch, closed over the allowed services and repairs.
    """

    async def restore_mongodb_access(ctx: ToolContext, service: str, access: str) -> str:
        """Restore one allowlisted MongoDB access mechanism in AIOpsLab."""
        if ctx.role not in _RECOVERY_ROLES:
            return "error: restore_mongodb_access is permitted only for the execution agent"
        if service not in config.mongodb_services or access not in config.mongodb_access_repairs:
            return "error: unsupported MongoDB recovery selection"
        return await cast("_Bridge", ctx.transport).action(
            ADAPTER_NAME,
            "restore_mongodb_access",
            {"service": service, "access": access},
            stage="execution",
        )

    return restore_mongodb_access


def _recovery_parameters(config: AiopsLabConfig) -> dict[str, Any]:
    """Build the recovery schema from configuration rather than from a literal.

    The enums are sorted because the configured sets are frozensets and a schema whose
    member order changed between processes would change the bytes sent to the model. The
    prototype listed ``mongodb-rate`` first; sorting puts ``mongodb-geo`` there instead,
    which is the one place this tool's model-facing schema is not the prototype's.

    Args:
        config: The deployment's adapter configuration.

    Returns:
        The JSON Schema for the tool's two arguments.
    """
    return {
        "type": "object",
        "properties": {
            "service": {
                "type": "string",
                "enum": sorted(config.mongodb_services),
                "description": "MongoDB service whose application access is broken.",
            },
            "access": {
                "type": "string",
                "enum": sorted(config.mongodb_access_repairs),
                "description": (
                    "Use 'privileges' for a revoked admin role and 'user' "
                    "for a removed/unregistered admin user."
                ),
            },
        },
        "required": ["service", "access"],
    }


# Registered where the tools are declared, so the approval gate reasons about them the
# same way it reasons about any other call. The gate reads a process-wide policy registry
# rather than the toolset, and a tool it cannot find is treated as a write by whichever
# stage calls it — which would put the read-only probe behind an approval it must never
# need, and would lose the refusal wording the remediating role reads.
read_only("probe_service")
always_writes(
    "restore_mongodb_access",
    stages=_RECOVERY_ROLES,
    # The prototype's own wording, and the stage is not interpolated into it: the tool
    # names the one stage that may call it rather than the one that did, because the
    # model reading this needs to know where the capability lives.
    refusal=lambda _stage: (
        "error: restore_mongodb_access is permitted only for the execution agent"
    ),
)


def adapter_tools(config: AiopsLabConfig) -> tuple[AdapterTool, ...]:
    """The tools this environment contributes, with the roles allowed to see them.

    Args:
        config: The deployment's adapter configuration, which carries the closed set of
            MongoDB repairs the bridge will run.

    Returns:
        The probe, offered to the checking role, and the recovery action, offered to the
        remediating role, and neither to anything else.
    """
    return (
        AdapterTool(
            spec=ToolSpec(
                name="probe_service",
                parameters=_PROBE_PARAMETERS,
                safety="read_only",
                dispatch=_probe_service,
                requires=_BRIDGE,
                descriptions=_PROBE_DESCRIPTION,
            ),
            roles=_PROBE_ROLES,
        ),
        AdapterTool(
            spec=ToolSpec(
                name="restore_mongodb_access",
                parameters=_recovery_parameters(config),
                safety="cluster_write",
                dispatch=_restore_mongodb_access(config),
                requires=_BRIDGE,
                descriptions=_RECOVERY_DESCRIPTION,
            ),
            roles=_RECOVERY_ROLES,
        ),
    )
