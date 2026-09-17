"""Cluster and telemetry access through AIOpsLab's authenticated loopback bridge.

The benchmark process owns the cluster-admin credential, because it is the thing that
deploys and faults the target application. The agent deliberately receives none of it.
Every operation is therefore a POST to a host-side bridge that holds the credential,
knows the active problem namespace, and re-screens the command independently of
anything this process claims about it.

**Three capabilities, not five.** AIOpsLab deploys no Loki, so ``logs`` is absent and
``get_logs``, ``get_labels`` and ``get_label_values`` are dropped from every role's
toolset before the run, with the reason recorded. Logs remain reachable through scoped
``kubectl logs``, which is what the role prompts already say. ``kubernetes.rollback`` is
absent for the reason it is absent from the direct transport: undoing a write means
having recorded its inverse, and only the SREGym harness's own session state keeps such
a journal. In the prototype both facts lived as set membership inside the global
function that decided what every role could call, so adding a second benchmark meant
editing that function; here they are two words missing from a frozenset.

**One client per call.** The same rule the MCP transport carries, for the same reason: a
connection held across an episode dies part-way through it and every later call fails
contiguously, silently, after the diagnosis has been graded and while the remediation is
being attempted. A connection setup is nothing beside a model turn.

**Rendering and complete cluster reads.** :meth:`AiopsLabTransport.action` applies the
shared action bound and re-encodes structured telemetry as the Python ``repr`` its
consumers parse. Alert partitioning uses ``ast.literal_eval``, so leaving JSON ``true``
or ``null`` in the answer would bypass partitioning. Successful kubectl and structured telemetry results are
returned complete: snapshots parse resource JSON and telemetry tools enforce namespace
scope before any model sees it. The tools apply their display bound and save full
scoped output in the session workspace. Cutting these reads at the action bound destroys the JSON and also
prevents ``save_as`` from saving the complete result.
"""

import json
import logging
from collections.abc import Mapping, Sequence
from shlex import join
from typing import Any
from uuid import uuid4

import httpx

from maver.audit.context import current_call
from maver.tools.errors import tool_error
from maver.tools.transport.base import TraceOperation

__all__ = ["AiopsLabTransport"]

_log = logging.getLogger(__name__)

_CAPABILITIES = frozenset({"kubernetes", "metrics", "traces"})
"""What the bridge serves. See the module docstring for the two that are missing."""

_SESSION_HEADER = "X-MAVER-Session"
"""Correlates one agent-side audit event with one host-side bridge access-log line.

The bridge needs nothing from it. Without it the two logs share only a wall clock, and
a timestamp join mis-mapped five sessions of six in one measured case.
"""

_RESPONSE_LIMIT = 10_000
_TRUNCATION_MARKER = "… [truncated]"
"""The existing display bound for bridge actions and errors.

Successful kubectl and structured telemetry bypass it so internal consumers can parse
and scope complete results. Model-facing tools own display truncation and saving.
"""

_TRACE_TOOL: dict[str, str] = {
    "services": "get_services",
    "operations": "get_operations",
    "traces": "get_traces",
    "dependencies": "get_dependency_graph",
}
"""The bridge's own name for each trace query, spelled as the harness spells it."""

_NO_LOGS = (
    "logs are not available on this transport: AIOpsLab deploys no log backend. Read a "
    "container's logs with scoped kubectl instead, for example "
    "`kubectl logs -n <namespace> <pod> --tail=200`."
)

_NO_ROLLBACK = (
    "rollback is not available on this transport: undoing a cluster write means having "
    "recorded its inverse, and the AIOpsLab bridge keeps no such journal. Re-read the "
    "resource and apply the state you want instead."
)


class AiopsLabTransport:
    """One episode's cluster and telemetry access, over the AIOpsLab action bridge."""

    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        timeout_s: float = 120.0,
        session_id: str | None = None,
    ) -> None:
        """Bind the transport to one running bridge.

        Plain values rather than a settings object: the adapter reads the deployment's
        configuration and this stays testable without one.

        Args:
            base_url: Where the bridge listens, normally a loopback address.
            token: The bearer token the bridge was started with.
            timeout_s: How long one action may take.
            session_id: The incident session, when it is already known.
        """
        self._base_url = base_url.rstrip("/")
        self._token = token
        self._timeout = timeout_s
        self.session_id = session_id or ""
        """The incident's correlation id, sent on every request that has one.

        A field rather than a constructor-only value, and assigned by the adapter, for
        the reason the MCP transport's is: the transport is built before the session it
        will run as is known.
        """
        self.namespace = ""
        """The incident's namespace, as the adapter read it from the episode.

        The protocol gives a transport no per-call namespace and neither the direct nor
        the MCP transport learns one, so this follows ``session_id``: a value the
        adapter assigns from the episode context it already holds. The bridge refuses
        any action whose namespace is not the active problem's, so an unset value fails
        closed rather than reaching another application's telemetry.
        """
        self.stage = ""
        """The calling role, as the bridge's access log records it.

        Overridden per call by the two tools whose authority the bridge checks — it
        permits service probing only from verify and MongoDB recovery only from
        execution — because those hold a tool context and therefore know their own role.
        """

    @property
    def name(self) -> str:
        """Which transport this is, for the audit record.

        Returns:
            ``aiopslab``.
        """
        return "aiopslab"

    @property
    def kubectl_proxy_url(self) -> str | None:
        """Report that there is no proxy route here.

        Returns:
            None. Cluster access is the bridge's own HTTP API, which runs kubectl on the
            orchestrator host: there is no endpoint a kubectl inside this process could
            be pointed at.
        """
        return None

    @property
    def capabilities(self) -> frozenset[str]:
        """What the bridge serves.

        Returns:
            Cluster access, metrics and traces. Neither logs nor rollback.
        """
        return _CAPABILITIES

    @property
    def timeout_seconds(self) -> int:
        """How long one call may take before it is abandoned.

        Floored at a second, as both other transports floor it. The configured value is
        a float, and truncating it toward zero turns anything under one second into no
        timeout at all, which the kubectl tool then uses to compute a deadline that has
        already passed.

        Returns:
            The per-call timeout, never below one second.
        """
        return max(1, int(self._timeout))

    async def kubectl(self, argv: Sequence[str]) -> str:
        """Run one kubectl invocation on the host that holds the credential.

        The arguments are re-joined into the command line the bridge's own action takes,
        binary included, because the bridge accepts nothing else and re-tokenises what
        it is given. Nothing is reclassified or stripped on the way: authority was
        settled before this was reached, and a second parse is a second chance to
        disagree about what the command says.

        Args:
            argv: The arguments, without a leading ``kubectl``.

        Returns:
            The complete successful command output, or a bounded failure beginning
            with ``error:``. Internal consumers parse the result; the kubectl tool
            applies display limits after saving the full output.
        """
        return await self.action(
            "kubectl",
            "exec_kubectl_cmd_safely",
            {"cmd": join(["kubectl", *argv])},
            bounded=False,
        )

    async def rollbackable_commands(self) -> str:
        """Report that no journal of undoable writes exists here.

        Returns:
            The unavailability, as a tool error. ``kubernetes.rollback`` is not in
            :attr:`capabilities`, so this is reached only by a caller ignoring them.
        """
        return tool_error(_NO_ROLLBACK)

    async def rollback_last(self) -> str:
        """Report that there is nothing to undo a write with.

        Returns:
            The unavailability, as a tool error.
        """
        return tool_error(_NO_ROLLBACK)

    async def query_metrics(self, query: str, *, window_minutes: int | None = None) -> str:
        """Run one PromQL query.

        Args:
            query: The PromQL expression.
            window_minutes: Ignored. The bridge runs an instant query and takes no
                range, exactly as the harness's own metrics server does, and the prompts
                were written against the window Prometheus chooses for one.

        Returns:
            The result, or a string beginning with ``error:``.
        """
        return await self.action("prometheus", "get_metrics", {"query": query}, structured=True)

    async def query_alerts(self) -> str:
        """Read the alerts the monitoring stack is currently firing.

        The bridge answers with Prometheus's own ``data`` object, in which the alerts
        are one key. Everything above reads the bare list the other two transports
        return, so the envelope comes off here.

        Returns:
            The firing alerts, or a string beginning with ``error:``.
        """
        return await self.action("prometheus", "get_alerts", {}, structured=True, select="alerts")

    async def log_labels(self) -> str:
        """Report that this environment indexes no logs.

        Returns:
            The unavailability, as a tool error. ``logs`` is not in
            :attr:`capabilities`, so the tool is dropped before the run and this is
            reached only by a caller ignoring them.
        """
        return tool_error(_NO_LOGS)

    async def log_label_values(self, label: str) -> str:
        """Report that this environment indexes no logs.

        Args:
            label: The label name, which nothing here can answer for.

        Returns:
            The unavailability, as a tool error.
        """
        return tool_error(_NO_LOGS)

    async def query_logs(self, query: str, *, last_n_minutes: int = 15) -> str:
        """Report that this environment deploys no log backend.

        Args:
            query: The query, which nothing here can run.
            last_n_minutes: How far back it would have looked.

        Returns:
            The unavailability, as a tool error, naming the way logs *are* reachable.
        """
        return tool_error(_NO_LOGS)

    async def query_traces(self, operation: TraceOperation, **kwargs: Any) -> str:
        """Read distributed traces.

        Args:
            operation: Which trace query to run.
            **kwargs: The operation's own arguments — ``service`` for ``operations`` and
                ``traces``, ``last_n_minutes`` for ``traces`` and ``dependencies`` —
                passed through as the bridge names them.

        Returns:
            The result, or a string beginning with ``error:``.
        """
        tool = _TRACE_TOOL.get(operation)
        if tool is None:
            return tool_error(f"unknown trace operation {operation!r}")
        return await self.action("jaeger", tool, dict(kwargs), structured=True)

    async def action(
        self,
        server: str,
        tool: str,
        arguments: Mapping[str, Any] | None = None,
        *,
        structured: bool = False,
        select: str | None = None,
        stage: str | None = None,
        bounded: bool = True,
    ) -> str:
        """POST one bridge action and render its answer.

        Generic on purpose. The two tools this environment adds — probing a service and
        restoring MongoDB access — are the adapter's own and are not in the capability
        vocabulary, so they reach the bridge through this rather than through a method
        the protocol would have to grow a word for.

        Never raises. A transport, protocol or bridge failure comes back as a string
        beginning with ``error:``, because far more of the system reads that prefix than
        reads any exception.

        Args:
            server: Which of the bridge's backends answers, e.g. ``kubectl``.
            tool: The action's name on that backend.
            arguments: The action's arguments.
            structured: Whether the answer is a telemetry payload a consumer above
                parses, and so should be re-rendered as a Python literal.
            select: The one key to lift out of a structured answer's envelope, when the
                backend wraps the payload the consumers above are written against. Part
                of the encoding rather than of the query, so it is applied here, before
                the bound: cutting the envelope first leaves nothing that parses.
            stage: The calling role, when the caller knows it better than this
                transport does.
            bounded: Apply the action display limit to unstructured results. Kubectl
                disables it for internal parsing and complete workspace saves. Structured
                telemetry also stays complete; failures always keep the limit.

        Returns:
            The bridge's text, or the failure as a model-facing string.
        """
        invocation = current_call()
        call_stage = invocation.stage if invocation is not None else self.stage
        payload = {
            "server": server,
            "tool": tool,
            "arguments": dict(arguments or {}),
            "namespace": self.namespace,
            "stage": call_stage if stage is None else stage,
        }
        headers = {
            "Authorization": f"Bearer {self._token}",
            "X-MAVER-Request-ID": uuid4().hex,
        }
        session_id = invocation.session_id if invocation is not None else self.session_id
        if session_id:
            headers[_SESSION_HEADER] = session_id
        if invocation is not None:
            headers["X-MAVER-Invocation-ID"] = invocation.invocation_id
            if invocation.call_id:
                headers["X-MAVER-Call-ID"] = invocation.call_id
            if invocation.run_id:
                headers["X-MAVER-Run-ID"] = invocation.run_id
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    f"{self._base_url}/v1/action", json=payload, headers=headers
                )
            body = None
            try:
                body = _body(response)
            finally:
                if invocation is not None:
                    captured = invocation.journal.capture(
                        "transport_response",
                        {
                            "request_id": headers["X-MAVER-Request-ID"],
                            "server": server,
                            "tool": tool,
                            "status": response.status_code,
                            "body": body if body is not None else {"raw": response.text},
                        },
                    )
                    invocation.observation.output_artifacts.append(
                        {**captured, "source": "aiopslab"}
                    )
            if not isinstance(body, dict):
                return tool_error("AIOpsLab bridge returned a non-object response")
            result = _render_output(
                str(body.get("result", "")),
                structured=structured,
                select=select,
                bounded=(bounded and not structured) or not body.get("ok", False),
            )
            if not body.get("ok", False):
                return tool_error(result or "AIOpsLab bridge action failed")
            return result or "(empty AIOpsLab bridge response)"
        except Exception as exc:
            _log.warning("[aiopslab] %s.%s failed: %s", server, tool, exc)
            return tool_error(
                f"AIOpsLab bridge {server}.{tool} call failed: {type(exc).__name__}: {exc}"
            )


def _body(response: httpx.Response) -> Any:
    """Decode a bridge reply, refusal included.

    The prototype's client called ``raise_for_status`` first, so every 400 and 401 the
    bridge answers with reached the model as an exception's repr and the bridge's own
    words were discarded. Those words are the whole error channel — which flag was
    refused, which namespace the command escaped to — and they are the one thing that
    lets a role repair a command rather than reissue it.

    Args:
        response: The bridge's reply.

    Returns:
        The decoded body.

    Raises:
        httpx.HTTPStatusError: When a failing reply carried no bridge answer to read, so
            that the status still reaches the caller as a failure.
    """
    try:
        body = response.json()
    except ValueError:
        body = None
    # Only an answer in the bridge's own shape is trusted over the status. Anything
    # else — a proxy's error page, a framework's own JSON — is a failure this process
    # cannot read, and reporting it as one names the status the reply carried.
    if isinstance(body, dict) and "ok" in body:
        return body
    response.raise_for_status()
    return body


def _render_output(
    result: str, *, structured: bool, select: str | None = None, bounded: bool = True
) -> str:
    """Encode one answer, then apply its display bound when requested.

    Args:
        result: The bridge's text.
        structured: Whether it is a telemetry payload a consumer above parses.
        select: The key to lift out of the payload's envelope, when it has one.
        bounded: Whether to apply the shared action display limit.

    Returns:
        The text, re-encoded for structured telemetry and bounded when requested.
    """
    text = _as_literal(result, select) if structured else result
    if not bounded or len(text) <= _RESPONSE_LIMIT:
        return text
    invocation = current_call()
    if invocation is not None:
        invocation.observation.upstream_truncated = True
        invocation.observation.incomplete = True
    return text[:_RESPONSE_LIMIT] + _TRUNCATION_MARKER


def _as_literal(result: str, select: str | None = None) -> str:
    """Re-render a JSON payload as the Python literal the consumers above parse.

    Idempotent: a payload the bridge already wrote as a ``repr`` is not JSON, so it is
    returned untouched, and so is anything that is neither.

    Args:
        result: The bridge's text.
        select: The key to lift out of the payload when it is an object wrapping the
            value the consumers read. Absent from the payload, it is not applied, so a
            backend that answers with the bare value is rendered as it stands.

    Returns:
        The payload as a Python literal, or the text unchanged.
    """
    try:
        payload = json.loads(result)
    except (TypeError, ValueError, RecursionError):
        return result
    if select is not None and isinstance(payload, dict) and select in payload:
        payload = payload[select]
    return repr(payload)
