"""Cluster and telemetry access over the harness's own MCP servers.

Four servers, mounted over server-sent events, and the tool names the model sees are
theirs rather than ours: the agent's tool surface matches what the benchmark's own
reference agents are given, which is what makes the comparison between them a comparison
of agents.

**One connection per call, and that is not an oversight.** A session held open for an
episode dies partway through it, and every later call then fails with a message about
invalid request parameters that says nothing about what happened. Measured across cells
making forty-five or more calls, 59% of calls failed on average and 69 of 82 in the
worst one, contiguously from the moment the session died — so the agent loses its
cluster tools mid-episode and flies blind for the rest of the run, which destroys the
remediation phase silently because that phase runs last. The second reason is
structural: the whole lifecycle stays inside one coroutine, so calls fanned out
concurrently stay independent and a cancellation never crosses a task boundary.

This transport truncates. The servers cut every response at about ten thousand
characters and append a marker, which is harmless for text and fatal for structured
output, since the parse then fails and the caller sees nothing. That is a property of
the measured configuration, not a defect to fix here, and it is one reason a comparison
against the archive has to run on this transport rather than the direct one.

kubectl is the exception, and it does not go over MCP at all. It runs as a local binary
against the harness's filtering proxy, which is what the archived campaign measured: the
proxy holds the credentials, applies the same screens on the way out of the cluster, and
returns a full response, so a namespace read comes back parseable. That is also the
route the model's own kubectl takes inside a shell, so the two agree by construction.
"""

import logging
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from mcp import ClientSession
from mcp.client.sse import sse_client

from maver.adapters.sregym.conductor import SSID_HEADER
from maver.audit.context import current_call
from maver.tools.errors import tool_error
from maver.tools.kubectl_proxy import ProxyRoute, run_via_proxy
from maver.tools.kubernetes import leading_verb
from maver.tools.transport.base import TraceOperation

__all__ = ["TOOL_SERVER", "McpTransport"]

_log = logging.getLogger(__name__)

_CONNECT_TIMEOUT = 10.0
"""Reaching the endpoint should be quick; running a tool is bounded separately."""

TOOL_SERVER: dict[str, str] = {
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
"""Which of the four servers answers each tool."""

_CAPABILITIES = frozenset({"kubernetes", "kubernetes.rollback", "metrics", "logs", "traces"})
"""Rollback is served here and nowhere else: the journal of inverse commands is the
harness's own session state, keyed on the correlation header, and no transport talking
straight to an API server can reconstruct it."""

_TRACE_TOOL: dict[str, str] = {
    "services": "get_services",
    "operations": "get_operations",
    "traces": "get_traces",
    "dependencies": "get_dependency_graph",
}


class McpTransport:
    """One episode's cluster and telemetry access, over four MCP servers."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_s: float = 60.0,
        proxy: ProxyRoute | None = None,
        kubeconfig_dir: Path | None = None,
    ) -> None:
        """Bind the transport to one deployment of the harness's servers.

        Args:
            base_url: Where the servers are mounted.
            timeout_s: How long one tool call may take.
            proxy: How a local kubectl reaches the cluster. The shipped defaults are the
                configuration the archived campaign ran.
            kubeconfig_dir: Where the generated kubeconfig is written. Defaults to the
                system temporary directory, which is where the prototype's own internal
                reads wrote it; the adapter points it at the deployment's working
                directory instead, so a run leaves nothing outside it and the file it
                actually used can be read afterwards.
        """
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_s
        self.proxy = proxy or ProxyRoute()
        """How a local kubectl reaches the cluster.

        Public because the sandboxed execution tools read it: they build the environment
        a shell's own ``kubectl`` runs under, and every field of it is this environment's
        to decide rather than the core's."""
        self._kubeconfig_dir = (
            Path(tempfile.gettempdir()) / ".kubectl-proxy"
            if kubeconfig_dir is None
            else kubeconfig_dir
        )
        self.session_id = ""
        """The episode's correlation id, sent on every request.

        A field on the instance rather than a context variable. The prototype relied on
        a context variable propagating into every task the run spawned, which worked and
        was invisible; an object that carries its own id is the same guarantee written
        down."""

    @property
    def name(self) -> str:
        """Which transport this is, for the audit record.

        Returns:
            ``mcp``.
        """
        return "mcp"

    @property
    def kubectl_proxy_url(self) -> str | None:
        """Where a local kubectl reaches this harness's cluster without credentials.

        Returns:
            The proxy's base URL. Always set here: the harness deploys the proxy as part
            of the environment, and it is the only cluster route this transport has.
        """
        return self.proxy.url

    @property
    def kubectl_proxy_required(self) -> bool:
        """Whether a kubectl inside a shell waits for the proxy before it runs.

        Returns:
            True by default. The proxy is a port-forward the harness restarts between
            problems, and a command that raced the restart reports a connection refused
            the model has no way to act on.
        """
        return self.proxy.required

    @property
    def kubectl_proxy_check_timeout_s(self) -> float:
        """How long one readiness probe of the proxy may take.

        Returns:
            The per-probe timeout.
        """
        return self.proxy.check_timeout_s

    @property
    def kubectl_proxy_ready_timeout_s(self) -> float:
        """How long readiness probing continues before a command is refused.

        Returns:
            The total wait.
        """
        return self.proxy.ready_timeout_s

    @property
    def capabilities(self) -> frozenset[str]:
        """What these servers serve.

        Returns:
            Cluster access including rollback, plus all three telemetry backends.
        """
        return _CAPABILITIES

    @property
    def timeout_seconds(self) -> int:
        """How long one call may take before it is abandoned.

        Floored at a second. The configured value is a float and truncating it toward
        zero turns anything under one second into no timeout at all, which the kubectl
        tool then uses to compute a deadline that has already passed.

        Returns:
            The per-call timeout, never below one second.
        """
        return max(1, int(self._timeout))

    async def kubectl(self, argv: Sequence[str]) -> str:
        """Run one kubectl locally, against the harness's filtering proxy.

        Not over MCP, which is where every other call here goes. The harness's kubectl
        server cuts a response at about ten thousand characters and appends a marker,
        which is invisible in prose and fatal to ``-o json``: the namespace snapshot and
        the trace scoping both parse this text, and a truncated payload leaves them with
        nothing while reporting no failure. The proxy filters the same things the server
        does, on the way out of the cluster rather than on the way back, so nothing is
        hidden by the change — and the model's own kubectl inside a shell reaches the
        cluster through exactly this route, so what a role reads here and what it reads
        there come from one place.

        Nothing is reclassified on the way: authority was settled before this was
        reached, and a second parse is a second chance to disagree about what the command
        says.

        Args:
            argv: The arguments, without a leading ``kubectl``.

        Returns:
            A successful command's own stdout, byte for byte, or a string beginning with
            ``error:``.
        """
        refusal = _blocking_watch_refusal(argv)
        if refusal is not None:
            return refusal
        code, output = await run_via_proxy(
            argv,
            proxy_url=self.proxy.url,
            kubeconfig_dir=self._kubeconfig_dir,
            binary=self.proxy.binary,
            timeout_s=self.proxy.exec_timeout_s,
        )
        return output if code == 0 else tool_error(output)

    async def rollbackable_commands(self) -> str:
        """List the writes the harness could still undo.

        Returns:
            The undoable commands, in the order they would be reverted.
        """
        return await self.call("get_previous_rollbackable_cmd", {})

    async def rollback_last(self) -> str:
        """Undo the most recent write.

        Returns:
            What was undone.
        """
        return await self.call("rollback_command", {})

    async def query_metrics(self, query: str, *, window_minutes: int | None = None) -> str:
        """Run one PromQL query.

        Args:
            query: The PromQL expression.
            window_minutes: Ignored. This backend windows server-side and takes no
                range, and every prompt and published measurement was written against
                the window it chooses.

        Returns:
            The result, or a string beginning with ``error:``.
        """
        return await self.call("get_metrics", {"query": query})

    async def query_alerts(self) -> str:
        """Read the alerts the monitoring stack is currently firing.

        Returns:
            The firing alerts, or a string beginning with ``error:``.
        """
        return await self.call("get_alerts", {})

    async def log_labels(self) -> str:
        """List the label names the log backend indexes.

        Returns:
            The label names, or a string beginning with ``error:``.
        """
        return await self.call("get_labels", {})

    async def log_label_values(self, label: str) -> str:
        """List the values a log label takes.

        Args:
            label: The label name.

        Returns:
            The values, or a string beginning with ``error:``.
        """
        return await self.call("get_label_values", {"label": label})

    async def query_logs(self, query: str, *, last_n_minutes: int = 15) -> str:
        """Run one log query over a recent window.

        Args:
            query: The LogQL expression.
            last_n_minutes: How far back to look.

        Returns:
            Matching log lines, or a string beginning with ``error:``.
        """
        return await self.call("get_logs", {"query": query, "last_n_minutes": last_n_minutes})

    async def query_traces(self, operation: TraceOperation, **kwargs: Any) -> str:
        """Read distributed traces.

        Args:
            operation: Which trace query to run.
            **kwargs: The operation's own arguments, passed through as the server names
                them.

        Returns:
            The result, or a string beginning with ``error:``.
        """
        tool = _TRACE_TOOL.get(operation)
        if tool is None:
            return f"error: unknown trace operation {operation!r}"
        return await self.call(tool, dict(kwargs))

    async def call(self, tool: str, arguments: dict[str, Any]) -> str:
        """Open a session, run one tool, and tear the session down.

        Never raises. A transport, handshake or tool failure comes back as a string
        beginning with ``error:``, because far more of the system reads that prefix
        than reads any exception.

        Args:
            tool: The server-side tool name.
            arguments: Its arguments.

        Returns:
            The tool's text, or the failure as a model-facing string.
        """
        server = TOOL_SERVER.get(tool, "kubectl")
        url = f"{self._base_url}/{server}/sse"
        headers = {SSID_HEADER: self.session_id} if self.session_id else {}
        try:
            async with (
                sse_client(
                    url,
                    headers=headers,
                    timeout=_CONNECT_TIMEOUT,
                    sse_read_timeout=self._timeout,
                ) as (read_stream, write_stream),
                ClientSession(read_stream, write_stream) as session,
            ):
                await session.initialize()
                result = await session.call_tool(
                    tool, arguments, read_timeout_seconds=self._timeout
                )
            return _rendered(result)
        except Exception as exc:
            _log.warning("[mcp] %s.%s failed: %s", server, tool, exc)
            return f"error: MCP {server}.{tool} call failed: {type(exc).__name__}: {exc}"


_INTERACTIVE_FLAGS = frozenset({"-i", "-t", "-it", "-ti", "--stdin", "--tty"})


def _blocking_watch_refusal(argv: Sequence[str]) -> str | None:
    """Refuse the verbs the server cannot bound, with the guidance the archive gave.

    The server runs a watch until kubectl itself gives up and honours no timeout the
    caller sets: one ``rollout status --timeout=120s`` ran for eight and a half minutes,
    and because the server answers one kubectl at a time, every other kubectl in the
    session timed out behind it. The proxy the archived runs went through refused the
    same verbs with the same advice, so the model reads what it read then.
    """
    verb, idx = leading_verb(argv)
    if (
        verb == "rollout"
        and idx is not None
        and idx + 1 < len(argv)
        and argv[idx + 1].lower() == "status"
    ):
        return (
            "error: kubectl rollout status is not permitted on this transport because it "
            "can leave kubectl hanging against the SREGym server; use kubectl get with "
            "-o json/jsonpath to check deployment generation, Ready replicas, pods, "
            "endpoints, events, logs, or the incident SLI"
        )
    if verb == "wait":
        return (
            "error: kubectl wait is not permitted on this transport because it uses a watch "
            "request that can wedge the SREGym server; use kubectl get with -o json/jsonpath "
            "to check readiness or rollout fields"
        )
    if verb in {"exec", "run", "attach"}:
        for arg in argv:
            if arg in _INTERACTIVE_FLAGS:
                return (
                    f"error: kubectl {verb} with interactive flags ({arg}) is not permitted on "
                    "this transport because it hangs the non-interactive server. Run "
                    "non-interactive commands without -i or -t instead."
                )
    return None


def _rendered(result: Any) -> str:
    """Flatten a tool result into the string the model reads.

    A tool-reported failure is marked so the loop's success heuristic sees it as one.
    The field naming that failure was spelled two ways across major versions of the
    client library, and reading only one of them silently drops the prefix, so both are
    tried.
    """
    parts: list[str] = []
    for block in getattr(result, "content", None) or []:
        text = getattr(block, "text", None)
        if text is not None:
            parts.append(text)
            continue
        data = getattr(block, "data", None)
        parts.append(str(data) if data is not None else str(block))
    out = "\n".join(part for part in parts if part).strip() or "(empty MCP response)"
    failed = getattr(result, "is_error", None)
    if failed is None:
        failed = getattr(result, "isError", False)
    if failed or out.startswith(
        (
            "[prom_mcp] Error querying ",
            "[loki_mcp] Error querying ",
            "[ob_mcp] Error querying ",
            "Query failed:",
        )
    ):
        return out if out.startswith("error:") else f"error: {out}"
    if out.endswith(("... [truncated]", "… [truncated]")):
        if call := current_call():
            call.observation.incomplete = True
            call.observation.upstream_truncated = True
            call.observation.errors.append("MCP response was truncated upstream")
        return f"error: MCP response was truncated upstream; partial output follows:\n{out}"
    return out
