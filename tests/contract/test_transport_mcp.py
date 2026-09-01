"""The MCP transport against the shared contract, stubbed at its own I/O seam.

One seam, because every call opens its own session: replacing the client library's
connection and session objects replaces the whole of this transport's I/O. The stubbed
servers answer from the contract's payloads in the encoding the harness's own servers
use, which is a Python literal rather than JSON.
"""

from collections.abc import Iterator
from typing import Any

import pytest
from transport_contract import ALERTS, KUBECTL_JSON, SERVICES, ToolTransportContract

from kira.adapters.sregym.transport import McpTransport

SEAM = "kira.adapters.sregym.transport"

_ANSWER = {
    "exec_kubectl_cmd_safely": KUBECTL_JSON,
    "get_alerts": repr(ALERTS),
    "get_services": repr(SERVICES),
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
