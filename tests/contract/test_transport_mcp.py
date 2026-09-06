"""The MCP transport against the shared contract, stubbed at its own I/O seam.

One seam, because every call opens its own session: replacing the client library's
connection and session objects replaces the whole of this transport's I/O. The stubbed
servers answer from the contract's payloads in the encoding the harness's own servers
use, which is a Python literal rather than JSON.
"""

import time
from collections.abc import Iterator
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

SEAM = "kira.adapters.sregym.transport"

_STAMP = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(SAMPLE_AT))


def _lines(rows: list[tuple[str, str]]) -> str:
    """Render what the harness's log server returns: one line per entry."""
    return "\n".join(f"[{_STAMP}] [pod={pod}] {body}" for pod, body in rows)


_ANSWER = {
    "exec_kubectl_cmd_safely": KUBECTL_JSON,
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
    def transport(self, monkeypatch: pytest.MonkeyPatch) -> Iterator[McpTransport]:
        monkeypatch.setattr(f"{SEAM}.sse_client", lambda *a, **k: _Connection())
        monkeypatch.setattr(f"{SEAM}.ClientSession", lambda read, write: _Session())
        yield McpTransport("http://mcp.contract.test:9954", timeout_s=30.0)

    @pytest.fixture
    def unreachable(self, monkeypatch: pytest.MonkeyPatch) -> Iterator[McpTransport]:
        """Nothing listening on the servers' port.

        The connection is where this fails rather than the tool call, because that is
        where it failed in the measured runs: the session died and every later call in
        the episode failed contiguously from that moment.
        """

        def refuse(*args: Any, **kwargs: Any) -> Any:
            raise ConnectionRefusedError("nothing listening")

        monkeypatch.setattr(f"{SEAM}.sse_client", refuse)
        yield McpTransport("http://mcp.contract.test:9954", timeout_s=30.0)


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

    await transport.kubectl(["get", "pods"])
    await transport.kubectl(["get", "svc"])
    await transport.query_metrics("up")

    assert counter.sessions_entered == 3
    assert counter.sessions_exited == 3
    assert counter.connections_entered == 3
    assert counter.connections_exited == 3
