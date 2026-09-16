"""The AIOpsLab transport against the shared contract, stubbed at its own I/O seam.

One seam, because every action is one POST to the loopback bridge: replacing the HTTP
transport replaces the whole of this implementation's I/O. The stubbed bridge answers
from the contract's payloads in the encoding the real bridge produces — compact JSON for
telemetry, the command's own text for kubectl — so what is exercised here is the
transport's own rendering rather than a fixture that already returned the right shape.

The log properties are the one part of the contract that cannot apply. This environment
deploys no log backend, so ``logs`` is not among the declared capabilities and the three
log tools are dropped before a run ever starts; what has to hold instead is the property
the rollback check states, that a caller ignoring the capability is answered with a
legible failure rather than something that reads as an answer.
"""

import json
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
import respx
from transport_contract import (
    ALERTS,
    DEPENDENCIES,
    KUBECTL_JSON,
    LOG_QUERY,
    METRIC_SERIES,
    OPERATIONS,
    SERVICES,
    TRACES,
    ToolTransportContract,
)

from maver.adapters.aiopslab.transport import AiopsLabTransport
from maver.tools.errors import call_failed

BRIDGE = "http://127.0.0.1:18099"
TOKEN = "contract-token"
SESSION = "session-under-test"
NAMESPACE = "test-hotel-reservation"


def _compact(payload: Any) -> str:
    """Encode as the bridge does: JSON with no spare whitespace."""
    return json.dumps(payload, separators=(",", ":"), default=str)


_ANSWER = {
    ("kubectl", "exec_kubectl_cmd_safely"): KUBECTL_JSON,
    ("prometheus", "get_alerts"): _compact(ALERTS),
    ("prometheus", "get_metrics"): _compact({"resultType": "vector", "result": METRIC_SERIES}),
    ("jaeger", "get_services"): _compact(SERVICES),
    ("jaeger", "get_operations"): _compact(OPERATIONS),
    ("jaeger", "get_traces"): _compact(TRACES),
    ("jaeger", "get_dependency_graph"): _compact(DEPENDENCIES),
}


def _bridge(request: httpx.Request) -> httpx.Response:
    """Answer one action the way a bridge holding the credential would."""
    body = json.loads(request.content)
    if request.headers.get("Authorization") != f"Bearer {TOKEN}":
        return httpx.Response(401, json={"ok": False, "result": "unauthorized"})
    answer = _ANSWER.get((body["server"], body["tool"]))
    if answer is None:
        return httpx.Response(
            400, json={"ok": False, "result": f"unsupported bridge tool: {body['tool']}"}
        )
    return httpx.Response(200, json={"ok": True, "result": answer})


def _transport() -> AiopsLabTransport:
    """Build the transport as the adapter would, namespace and session bound."""
    transport = AiopsLabTransport(BRIDGE, TOKEN, timeout_s=30.0, session_id=SESSION)
    transport.namespace = NAMESPACE
    transport.stage = "rca"
    return transport


class TestAiopsLabTransport(ToolTransportContract):
    @pytest.fixture
    def transport(self) -> Iterator[AiopsLabTransport]:
        with respx.mock(assert_all_called=False) as router:
            router.post(f"{BRIDGE}/v1/action").mock(side_effect=_bridge)
            yield _transport()

    @pytest.fixture
    def unreachable(self) -> Iterator[AiopsLabTransport]:
        """The bridge's port with nothing behind it.

        The benchmark process owns the bridge, so it goes away when the problem it was
        started for ends — which is the shape this failure takes in a real run, part-way
        through an episode rather than at start-up.
        """
        with respx.mock(assert_all_called=False) as router:
            router.route().mock(side_effect=httpx.ConnectError("no route to host"))
            yield _transport()

    async def test_log_reads_arrive_one_line_per_entry_with_the_labels_that_place_them(
        self, transport: AiopsLabTransport
    ) -> None:
        """AIOpsLab deploys no log backend, so this property has nothing to hold over.

        The contract's own rollback check states what replaces it: the three log tools
        are withheld before the run from a transport that does not declare ``logs``, so
        the only way these are reached at all is a caller that ignored the capability,
        and answering such a caller with something that reads as an empty log search
        would tell a role the request path was quiet when nothing looked at it.
        """
        assert "logs" not in transport.capabilities

        for answer in (
            await transport.query_logs(LOG_QUERY),
            await transport.log_labels(),
            await transport.log_label_values("namespace"),
        ):
            assert call_failed(answer)
            # Naming the way logs *are* reachable here, so the role redirects rather
            # than reissuing the call it was just refused.
            assert "kubectl logs" in answer
