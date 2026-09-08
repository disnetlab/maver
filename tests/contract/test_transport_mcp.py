"""The MCP transport against the shared contract, stubbed at its own I/O seams.

Two seams, since kubectl stopped going over MCP. Every telemetry call opens its own
session, so replacing the client library's connection and session objects replaces the
whole of that half; the stubbed servers answer from the contract's payloads in the
encoding the harness's own servers use, which is a Python literal rather than JSON.
kubectl is a local subprocess run against the harness's proxy, so it is given a script on
PATH that prints one canned read and never looks at a cluster or a proxy.
"""

import stat
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from transport_contract import (
    ALERTS,
    DEPENDENCIES,
    KUBECTL_JSON,
    LOG_COUNTS,
    LOG_ENTRIES,
    METRIC_SERIES,
    OPERATIONS,
    SAMPLE_AT,
    SERVICES,
    TRACES,
    ToolTransportContract,
)

from kira.adapters.sregym.transport import McpTransport
from kira.tools.kubectl_proxy import ProxyRoute

SEAM = "kira.adapters.sregym.transport"

# Never resolves, so the unreachable fixture does not depend on whether the host running
# the suite happens to have kubectl on it.
ABSENT_KUBECTL = "kubectl-that-is-not-installed"

_STAMP = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(SAMPLE_AT))


def _kubectl_stub(tmp_path: Path) -> Path:
    """Install a kubectl that prints the contract's canned read and nothing else.

    It cats a file rather than echoing, because echo and heredocs both append a newline
    the contract's byte-for-byte comparison would then be about.

    Args:
        tmp_path: The test's own directory.

    Returns:
        The path to the executable.
    """
    payload = tmp_path / "read.json"
    payload.write_text(KUBECTL_JSON)
    script = tmp_path / "kubectl"
    script.write_text(f"#!/bin/sh\ncat '{payload}'\n")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return script


def _lines(rows: list[tuple[str, str]]) -> str:
    """Render what the harness's log server returns: one line per entry."""
    return "\n".join(f"[{_STAMP}] [pod={pod}] {body}" for pod, body in rows)


_ANSWER = {
    "get_alerts": repr(ALERTS),
    "get_metrics": repr({"resultType": "vector", "result": METRIC_SERIES}),
    "get_services": repr(SERVICES),
    "get_operations": repr(OPERATIONS),
    "get_traces": repr(TRACES),
    "get_dependency_graph": repr(DEPENDENCIES),
}


class _Block:
    def __init__(self, text: str) -> None:
        self.text = text


class _Result:
    def __init__(self, text: str) -> None:
        self.content = [_Block(text)]
        self.is_error = False


class _Session:
    async def initialize(self) -> None:
        return None

    async def call_tool(self, name: str, arguments: Any, **kwargs: Any) -> _Result:
        if name == "get_logs":
            counted = "count_over_time" in str((arguments or {}).get("query", ""))
            return _Result(_lines(LOG_COUNTS if counted else LOG_ENTRIES))
        return _Result(_ANSWER.get(name, "ok"))

    async def __aenter__(self) -> "_Session":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None


class _Connection:
    async def __aenter__(self) -> tuple[None, None]:
        return (None, None)

    async def __aexit__(self, *exc: Any) -> None:
        return None


class TestMcpTransport(ToolTransportContract):
    @pytest.fixture
    def transport(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[McpTransport]:
        monkeypatch.setattr(f"{SEAM}.sse_client", lambda *a, **k: _Connection())
        monkeypatch.setattr(f"{SEAM}.ClientSession", lambda read, write: _Session())
        yield McpTransport(
            "http://mcp.contract.test:9954",
            timeout_s=30.0,
            proxy=ProxyRoute(
                url="http://proxy.contract.test:16444", binary=str(_kubectl_stub(tmp_path))
            ),
            kubeconfig_dir=tmp_path,
        )

    @pytest.fixture
    def unreachable(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> Iterator[McpTransport]:
        """Nothing listening on the servers' port, and no kubectl to run.

        The connection is where the telemetry half fails rather than the tool call,
        because that is where it failed in the measured runs: the session died and every
        later call in the episode failed contiguously from that moment.
        """

        def refuse(*args: Any, **kwargs: Any) -> Any:
            raise ConnectionRefusedError("nothing listening")

        monkeypatch.setattr(f"{SEAM}.sse_client", refuse)
        yield McpTransport(
            "http://mcp.contract.test:9954",
            timeout_s=30.0,
            proxy=ProxyRoute(url="http://proxy.contract.test:16444", binary=ABSENT_KUBECTL),
            kubeconfig_dir=tmp_path,
        )


class _Counter:
    """Counts how many times a session and its connection were opened and closed."""

    def __init__(self) -> None:
        self.sessions_entered = 0
        self.sessions_exited = 0
        self.connections_entered = 0
        self.connections_exited = 0


class _CountingSession(_Session):
    def __init__(self, counter: _Counter) -> None:
        self.counter = counter

    async def __aenter__(self) -> "_CountingSession":
        self.counter.sessions_entered += 1
        return self

    async def __aexit__(self, *exc: Any) -> None:
        self.counter.sessions_exited += 1


class _CountingConnection(_Connection):
    def __init__(self, counter: _Counter) -> None:
        self.counter = counter

    async def __aenter__(self) -> tuple[None, None]:
        self.counter.connections_entered += 1
        return (None, None)

    async def __aexit__(self, *exc: Any) -> None:
        self.counter.connections_exited += 1


async def test_every_call_opens_and_closes_its_own_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One session per call, and nothing counted how many there were.

    Telemetry calls, because kubectl no longer opens a session at all: it runs a local
    binary against the harness's proxy.

    The server ends a session after a period of inactivity and does not say so. Holding
    one open across an episode is the single worst failure this project has measured: a
    mean 59% tool-call failure rate across cells making forty-odd calls, worst case
    sixty-nine of eighty-two. It is invisible while it happens, because the failures
    begin part-way through a cell — after the diagnosis has been graded and while the
    remediation is being attempted.

    Reconnecting per call is what makes that impossible, and it is exactly the kind of
    thing a later optimisation reintroduces on the reasonable-sounding grounds that
    reconnecting is wasteful.
    """
    counter = _Counter()
    monkeypatch.setattr(f"{SEAM}.sse_client", lambda *a, **k: _CountingConnection(counter))
    monkeypatch.setattr(f"{SEAM}.ClientSession", lambda read, write: _CountingSession(counter))
    transport = McpTransport("http://mcp.contract.test:9954", timeout_s=30.0)

    await transport.query_alerts()
    await transport.query_logs('{namespace="shop"}')
    await transport.query_metrics("up")

    assert counter.sessions_entered == 3
    assert counter.sessions_exited == 3
    assert counter.connections_entered == 3
    assert counter.connections_exited == 3
