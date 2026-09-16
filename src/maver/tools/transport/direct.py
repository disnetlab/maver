"""The default transport: one real cluster, and real telemetry backends.

An artifact whose only cluster access path is a benchmark harness is a benchmark client
rather than an operations service. This is the implementation an operator points at
their own infrastructure: kubectl against their kubeconfig, and Prometheus, Loki and
Jaeger over their HTTP APIs.

kubectl is executed as a subprocess rather than rebuilt on the Python API client, and
that is a decision rather than an omission. What a tool returns is part of the
interface: the prompts and every published measurement were written against the exact
text the measured path produced, and that path was kubectl's own stdout. Reproducing
kubectl's table rendering, its column choices and its printer flags from the API client
is a large re-implementation whose failure mode is silent divergence in the text a model
reads. Nothing is re-parsed here either — the argument list arrives already tokenised
and classified, so this layer executes and does not decide.

Structured telemetry is rendered as the Python ``repr`` of the parsed response, which is
what the measured backend returned, and the choice is load-bearing rather than
nostalgic. The alert partitioning downstream reads its input with ``ast.literal_eval``,
so a JSON rendering carrying ``true`` and ``null`` would not parse, and every firing
alert would reach the model unpartitioned with nothing to say it had happened.

Two capabilities are deliberately absent. Rollback needs a journal holding the inverse
of each write, which a transport talking straight to an API server does not keep, and
the inverse of ``kubectl delete`` does not exist in general; its tools are dropped
before the run rather than failing during one. An endpoint left unconfigured likewise
withdraws its capability, because a tool that errors reads to a model as a fault to work
around, and it will try again, differently, and again.
"""

import asyncio
import contextlib
import logging
import os
import shutil
import signal
import time
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import quote

import httpx

from maver.settings.groups import TransportSettings
from maver.tools.errors import format_exit_result, timed_out, tool_error
from maver.tools.transport.base import ToolTransport, TraceOperation

__all__ = ["DirectTransport", "direct_transport"]

_log = logging.getLogger(__name__)

_UNWRAP_DEPTH = 10
"""How far a wrapped exception is followed before its own name is used."""

_REASON_LIMIT = 400
"""Characters of a backend's own complaint that reach the model."""

_TRACE_LIMIT = 20
_LOG_LIMIT = 100
"""Result caps, carried from the measured backend so a query returns a familiar size."""

_RANGE_POINTS = 100
_MIN_STEP_SECONDS = 15
"""Bounds on a range query's resolution, so one query cannot fill a context window."""

_REAP_SECONDS = 5.0
"""How long a killed kubectl is waited on before the call returns without it."""

_NO_ROLLBACK = (
    "rollback is not available on this transport: undoing a cluster write means "
    "having recorded its inverse, and a transport talking straight to the API server "
    "keeps no such journal. Re-read the resource and apply the state you want instead."
)


class _BackendError(Exception):
    """A backend could not answer, carrying the string the model should read."""

    def __init__(self, rendered: str) -> None:
        """Hold an already-rendered failure.

        Args:
            rendered: The model-facing string, ``error:`` prefix included.
        """
        super().__init__(rendered)
        self.rendered = rendered


class DirectTransport:
    """Cluster and telemetry access for a deployment that owns its own infrastructure."""

    def __init__(self, settings: TransportSettings) -> None:
        """Bind the transport to one deployment's endpoints.

        Args:
            settings: Where the cluster and the telemetry backends are.
        """
        self._settings = settings
        self._timeout = httpx.Timeout(float(settings.timeout_s))
        # Resolved once, and a capability rather than a per-call failure: a deployment
        # with no kubectl on it cannot serve cluster reads at all, and saying so before
        # the run costs one dropped tool instead of every call erroring.
        self._binary = shutil.which(settings.kubectl_binary)
        self._prometheus = _base_url(settings.prometheus_url)
        self._loki = _base_url(settings.loki_url)
        self._jaeger = _base_url(settings.jaeger_url)

        served = set()
        if self._binary is not None:
            # Reads and writes alike: whether a write is *accepted* is RBAC's answer,
            # given per call, and which roles may attempt one is the approval gate's.
            # This says only that nothing in the transport refuses to issue a command.
            served.add("kubernetes")
        if self._prometheus is not None:
            served.add("metrics")
        if self._loki is not None:
            served.add("logs")
        if self._jaeger is not None:
            served.add("traces")
        self._capabilities = frozenset(served)

        self._env = dict(os.environ)
        if settings.kubeconfig is not None:
            self._env["KUBECONFIG"] = str(settings.kubeconfig)

    @property
    def name(self) -> str:
        """Which transport this is, for the audit record.

        Returns:
            ``direct``.
        """
        return "direct"

    @property
    def kubectl_proxy_url(self) -> str | None:
        """Report that there is no proxy route here.

        Returns:
            None. This transport runs kubectl against the deployment's own kubeconfig,
            so a shell that ran kubectl itself would be running it with the agent's
            cluster credentials and no approval gate in front of it.
        """
        return None

    @property
    def capabilities(self) -> frozenset[str]:
        """What this deployment was actually configured to serve.

        Returns:
            The capabilities backed by a configured endpoint.
        """
        return self._capabilities

    @property
    def timeout_seconds(self) -> int:
        """How long one call may take before it is abandoned.

        Floored at a second, as the transport the harness supplies also floors it. The
        kubectl tool subtracts from this to bound a blocking wait, so a value below one
        gives every call a deadline that has already passed.

        Returns:
            The per-call timeout, never below one second.
        """
        return max(1, self._settings.timeout_s)

    async def kubectl(self, argv: Sequence[str]) -> str:
        """Run one kubectl invocation.

        Args:
            argv: The arguments, without a leading ``kubectl``.

        Returns:
            The command's own output, or a string beginning with ``error:`` when it
            could not be run.
        """
        if self._binary is None:
            return tool_error(
                f"no kubectl executable named {self._settings.kubectl_binary!r} is "
                "available to this deployment"
            )
        try:
            process = await asyncio.create_subprocess_exec(
                self._binary,
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._env,
                # In its own process group, so the timeout below can reach what kubectl
                # started as well as kubectl itself.
                start_new_session=True,
            )
        except OSError as exc:
            return tool_error(f"kubectl could not be executed: {exc}")

        timeout = float(self._settings.timeout_s)
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except TimeoutError:
            # Signalling the group rather than the process is what makes the deadline a
            # bound. kill() reaches kubectl alone, and wait() returns only once every
            # copy of the pipes is closed, so anything kubectl started that inherited
            # them — a wrapper script, an exec credential plugin holding kubectl's own
            # stderr — kept the call alive for its whole life while the model was told
            # it had been abandoned on time. The reap is bounded for the same reason.
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(process.wait(), _REAP_SECONDS)
            return timed_out(
                "kubectl",
                timeout,
                "A command that blocks — a wait, or a rollout status — needs its own "
                "shorter --timeout.",
            )

        out = stdout.decode(errors="replace")
        err = stderr.decode(errors="replace")
        # A successful command's stdout is returned exactly as kubectl wrote it, with no
        # envelope and with warnings left out of it. Two tools parse this text as JSON —
        # the namespace snapshot and the trace scoping's workload inventory — and a
        # prefix in front of the payload turns both into "unavailable" without a word
        # about why. Everything else is rendered, so a failure carries its exit status
        # and its stderr, and an empty result says that it is empty rather than
        # returning nothing for a greedy decoder to react to.
        if process.returncode == 0 and out.strip():
            return out
        return format_exit_result(process.returncode or 0, out + err)

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

        No caller passes a window, and the default is chosen rather than inherited: the
        measured backend ran an instant query, so an unwindowed call stays one. Prometheus
        evaluates it at the current instant, taking each series' most recent sample within
        its own five-minute staleness horizon, and that horizon is the window every
        published prompt was written against. A window, when one is given, becomes a range
        query whose step keeps the reply to about a hundred points per series, because a
        finer one returns a payload larger than the context it has to fit in.

        Args:
            query: The PromQL expression.
            window_minutes: How far back to look, making this a range query.

        Returns:
            The result section of the response, or a string beginning with ``error:``.
        """
        if self._prometheus is None:
            return tool_error("no metrics endpoint is configured for this deployment")
        if window_minutes is None:
            url = f"{self._prometheus}/api/v1/query"
            params: dict[str, Any] = {"query": query}
        else:
            end = time.time()
            span = max(1, window_minutes) * 60
            url = f"{self._prometheus}/api/v1/query_range"
            params = {
                "query": query,
                "start": end - span,
                "end": end,
                "step": f"{max(_MIN_STEP_SECONDS, span // _RANGE_POINTS)}s",
            }
        try:
            body = await self._fetch("prometheus", url, params)
        except _BackendError as failure:
            return failure.rendered
        data = body.get("data") if isinstance(body, dict) else None
        if data is None:
            return tool_error("prometheus answered without a data section")
        return str(data)

    async def query_alerts(self) -> str:
        """Read the alerts the monitoring stack is currently firing.

        Returns:
            The firing alerts as a Python literal the partitioning downstream can read,
            ``No firing alerts`` when there are none, or a string beginning with
            ``error:``.
        """
        if self._prometheus is None:
            return tool_error("no metrics endpoint is configured for this deployment")
        try:
            body = await self._fetch("prometheus", f"{self._prometheus}/api/v1/alerts", {})
        except _BackendError as failure:
            return failure.rendered
        data = body.get("data") if isinstance(body, dict) else None
        alerts = data.get("alerts") if isinstance(data, dict) else None
        if not isinstance(alerts, list):
            return tool_error("prometheus answered without a list of alerts")
        firing = [alert for alert in alerts if isinstance(alert, dict) and _is_firing(alert)]
        return str(firing) if firing else "No firing alerts"

    async def log_labels(self) -> str:
        """List the label names the log backend indexes.

        Returns:
            The label names, or a string beginning with ``error:``.
        """
        if self._loki is None:
            return tool_error("no log endpoint is configured for this deployment")
        try:
            body = await self._fetch("loki", f"{self._loki}/loki/api/v1/labels", {})
        except _BackendError as failure:
            return failure.rendered
        labels = body.get("data") if isinstance(body, dict) else None
        names = labels if isinstance(labels, list) else []
        return "Available labels:\n" + "\n".join(f"  - {name}" for name in names)

    async def log_label_values(self, label: str) -> str:
        """List the values a log label takes.

        Args:
            label: The label name.

        Returns:
            The values, or a string beginning with ``error:``.
        """
        if self._loki is None:
            return tool_error("no log endpoint is configured for this deployment")
        # The label is a path segment and a model writes it, so it is escaped rather
        # than allowed to address a route of its own choosing.
        segment = quote(label, safe="")
        try:
            body = await self._fetch("loki", f"{self._loki}/loki/api/v1/label/{segment}/values", {})
        except _BackendError as failure:
            return failure.rendered
        data = body.get("data") if isinstance(body, dict) else None
        values = data if isinstance(data, list) else []
        if not values:
            return f"No values found for label '{label}'."
        return f"Values for label '{label}':\n" + "\n".join(f"  - {value}" for value in values)

    async def query_logs(self, query: str, *, last_n_minutes: int = 15) -> str:
        """Run one LogQL query over a recent window.

        Args:
            query: The LogQL expression.
            last_n_minutes: How far back to look.

        Returns:
            The matching lines, a notice that there were none, or a string beginning
            with ``error:``.
        """
        if self._loki is None:
            return tool_error("no log endpoint is configured for this deployment")
        end = int(time.time() * 1_000_000_000)
        start = end - max(1, last_n_minutes) * 60 * 1_000_000_000
        try:
            body = await self._fetch(
                "loki",
                f"{self._loki}/loki/api/v1/query_range",
                {"query": query, "start": start, "end": end, "limit": _LOG_LIMIT},
            )
        except _BackendError as failure:
            return failure.rendered
        return _render_log_lines(body)

    async def query_traces(self, operation: TraceOperation, **kwargs: Any) -> str:
        """Read distributed traces.

        Args:
            operation: Which trace query to run.
            **kwargs: ``service`` for ``operations`` and ``traces``, ``last_n_minutes``
                for ``traces`` and ``dependencies``.

        Returns:
            The result, or a string beginning with ``error:``. An empty answer is the
            literal ``None`` the scoping layer above tests for.
        """
        if self._jaeger is None:
            return tool_error("no trace endpoint is configured for this deployment")
        try:
            return await self._trace_query(operation, kwargs)
        except _BackendError as failure:
            return failure.rendered

    async def _trace_query(self, operation: str, kwargs: Mapping[str, Any]) -> str:
        """Run one trace query against the backend's own API.

        Takes the operation as a plain string rather than the protocol's literal, so the
        unrecognised case below is a branch this can take rather than one the type
        checker has already ruled out on a promise the runtime does not keep.

        Args:
            operation: Which query.
            kwargs: The operation's arguments.

        Returns:
            The rendered result.

        Raises:
            _BackendError: When the backend could not answer.
        """
        base = self._jaeger
        if operation == "services":
            body = await self._fetch("jaeger", f"{base}/api/services", {})
            return _render_data_list(body)
        if operation == "operations":
            service = str(kwargs.get("service") or "")
            if not service:
                return tool_error("a service name is needed to list its operations")
            body = await self._fetch("jaeger", f"{base}/api/operations", {"service": service})
            return _render_data_list(body)
        if operation == "traces":
            service = str(kwargs.get("service") or "")
            if not service:
                return tool_error("a service name is needed to fetch its traces")
            end = int(time.time() * 1_000_000)
            span = max(1, int(kwargs.get("last_n_minutes") or 15)) * 60 * 1_000_000
            body = await self._fetch(
                "jaeger",
                f"{base}/api/traces",
                {
                    "service": service,
                    "start": end - span,
                    "end": end,
                    "limit": _TRACE_LIMIT,
                },
            )
            return _render_data_list(body)
        if operation == "dependencies":
            lookback = max(1, int(kwargs.get("last_n_minutes") or 30)) * 60 * 1000
            body = await self._fetch(
                "jaeger",
                f"{base}/api/dependencies",
                {"endTs": int(time.time() * 1000), "lookback": lookback},
            )
            # The whole envelope, not its data: the scoping layer above filters the
            # edges in place and re-renders what it was given.
            return str(body)
        return tool_error(f"unknown trace operation {operation!r}")

    async def _fetch(self, backend: str, url: str, params: Mapping[str, Any]) -> Any:
        """Run one GET and decode the reply.

        A client per call rather than one held for the run. The protocol has no place to
        close one, and a session held open across an incident is the failure mode the
        MCP path was measured hitting: once it dies the tools fail contiguously and the
        stage is lost silently. A connection setup is nothing beside a model turn.

        Args:
            backend: Which backend, named in whatever the model ends up reading.
            url: The absolute URL.
            params: Query parameters.

        Returns:
            The decoded body.

        Raises:
            _BackendError: Whenever the backend could not answer, already rendered as
                the string the model should read.
        """
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                body = response.json()
        except httpx.HTTPStatusError as exc:
            raise _BackendError(
                tool_error(
                    f"{backend} rejected the query ({exc.response.status_code}): "
                    f"{_reason(exc.response)}"
                )
            ) from exc
        except Exception as exc:
            # Never the transport-shaped in-band markers the prototype's backends
            # produced. Six things read the "error:" prefix, and nothing reads those.
            cause = _innermost(exc)
            _log.warning("[direct] %s request failed: %s", backend, cause)
            raise _BackendError(
                tool_error(f"{backend} request failed: {type(cause).__name__}: {cause}")
            ) from exc
        if isinstance(body, dict) and body.get("status") == "error":
            raise _BackendError(tool_error(f"{backend} rejected the query: {_stated_error(body)}"))
        return body


def direct_transport(settings: TransportSettings) -> ToolTransport:
    """Build the default transport for one run.

    Exists so the protocol conformance is checked here, where the implementation is,
    rather than wherever a composition root happens to bind it.

    Args:
        settings: Where the cluster and the telemetry backends are.

    Returns:
        The transport.
    """
    return DirectTransport(settings)


def _base_url(url: object) -> str | None:
    """Render a configured endpoint as a base to build paths on.

    Args:
        url: The setting's value, or None when unconfigured.

    Returns:
        The base URL without its trailing slash, or None.
    """
    return None if url is None else str(url).rstrip("/")


def _is_firing(alert: Mapping[str, Any]) -> bool:
    """Report whether one alert is firing rather than merely pending."""
    return bool(alert.get("state") == "firing")


def _render_data_list(body: Any) -> str:
    """Render a backend's ``data`` list, or the empty answer the caller tests for.

    Args:
        body: The decoded response.

    Returns:
        The list as a Python literal, or ``None`` when it holds nothing. The scoping
        layer above compares against that exact string.
    """
    data = body.get("data") if isinstance(body, dict) else None
    return str(data) if isinstance(data, list) and data else "None"


def _render_log_lines(body: Any) -> str:
    """Render a log query's result in whichever shape the backend answered.

    Args:
        body: The decoded response.

    Returns:
        One line per entry or per sample, prefixed with its timestamp and its series
        labels, or a notice that nothing matched.
    """
    data = body.get("data") if isinstance(body, dict) else None
    result = data.get("result") if isinstance(data, dict) else None
    if not isinstance(result, list) or not result:
        return "No logs found matching the query."

    # The same endpoint answers a LogQL metric query — count_over_time, rate, sum by —
    # in a different shape: the series identity sits under `metric` rather than `stream`
    # and the timestamps are seconds rather than nanoseconds. Read as a stream, such a
    # result rendered as bare numbers with every label dropped and every timestamp in
    # 1970, and carried no error prefix to say the read had gone wrong.
    kind = data.get("resultType") if isinstance(data, dict) else None
    metric = str(kind or "") in {"matrix", "vector"}
    lines = _sample_lines(result) if metric else _entry_lines(result)
    return "\n".join(lines) if lines else "No log entries found."


def _series_labels(series: Mapping[str, Any], key: str) -> str:
    """Render one series' identity as the model reads it."""
    labels = series.get(key)
    if not isinstance(labels, dict):
        return ""
    return ", ".join(f"{name}={value}" for name, value in labels.items())


def _entry_lines(streams: Sequence[Any]) -> list[str]:
    """One line per log entry of a stream result."""
    lines: list[str] = []
    for stream in streams:
        if not isinstance(stream, dict):
            continue
        rendered = _series_labels(stream, "stream")
        entries = stream.get("values")
        for entry in entries if isinstance(entries, list) else []:
            if not isinstance(entry, list) or len(entry) != 2:
                continue
            stamp, line = entry
            lines.append(f"[{_local_time(stamp)}] [{rendered}] {line}")
    return lines


def _sample_lines(series: Sequence[Any]) -> list[str]:
    """One line per sample of a metric result, matrix or vector."""
    lines: list[str] = []
    for one in series:
        if not isinstance(one, dict):
            continue
        rendered = _series_labels(one, "metric")
        # A matrix carries a series of samples; a vector carries the one it reduced to.
        samples = one.get("values")
        if not isinstance(samples, list):
            single = one.get("value")
            samples = [single] if single is not None else []
        for sample in samples:
            if not isinstance(sample, list) or len(sample) != 2:
                continue
            stamp, value = sample
            lines.append(f"[{_sample_time(stamp)}] [{rendered}] {value}")
    return lines


def _local_time(nanoseconds: Any) -> str:
    """Render a log timestamp in the reading operator's own clock.

    Args:
        nanoseconds: The entry's timestamp, as the backend wrote it.

    Returns:
        A local wall-clock time, or the raw value when it is not one.
    """
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(nanoseconds) / 1e9))
    except (TypeError, ValueError, OSError, OverflowError):
        return str(nanoseconds)


def _sample_time(seconds: Any) -> str:
    """Render a metric sample's timestamp, which the backend writes in seconds."""
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(seconds)))
    except (TypeError, ValueError, OSError, OverflowError):
        return str(seconds)


def _stated_error(body: Mapping[str, Any]) -> str:
    """Take a backend's own words for what was wrong with a query."""
    for key in ("error", "message"):
        value = body.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()[:_REASON_LIMIT]
    return "no reason given"


def _reason(response: httpx.Response) -> str:
    """Take a rejected request's reason from wherever the backend put it.

    Prometheus and Loki answer a bad query with the parse error itself, in JSON or as
    plain text depending on which of them and which failure. It is the one thing that
    lets a model repair its own query rather than reissue it.

    Args:
        response: The rejected response.

    Returns:
        What the backend said, bounded.
    """
    try:
        body = response.json()
    except ValueError:
        body = None
    if isinstance(body, dict):
        return _stated_error(body)
    return response.text.strip()[:_REASON_LIMIT] or response.reason_phrase


def _innermost(exc: BaseException) -> BaseException:
    """Reduce a wrapped failure to the one worth naming.

    A task group renders as ``unhandled errors in a TaskGroup (1 sub-exception)``, which
    names neither the failure nor the backend, and the prototype shipped that string to
    the model as a tool result.

    Args:
        exc: The exception as caught.

    Returns:
        The innermost single exception, following at most a bounded number of wrappers.
    """
    current = exc
    for _ in range(_UNWRAP_DEPTH):
        if isinstance(current, BaseExceptionGroup) and len(current.exceptions) == 1:
            current = current.exceptions[0]
            continue
        return current
    return current
