"""The direct transport against the shared contract, stubbed at its own I/O seams.

Two seams, because this implementation has two. kubectl is a subprocess, so it is given
a script on PATH that prints one canned read and never looks at a cluster; the telemetry
backends are HTTP, so respx answers them. What the transport does against a real cluster
is not something this file claims anything about.
"""

import stat
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
import respx
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

from maver.settings.groups import TransportSettings
from maver.tools.transport.direct import DirectTransport

PROMETHEUS = "http://prometheus.contract.test"
LOKI = "http://loki.contract.test"
JAEGER = "http://jaeger.contract.test"

# Never resolves, so the unreachable fixture does not depend on whether the host running
# the suite happens to have kubectl on it.
ABSENT_KUBECTL = "kubectl-that-is-not-installed"


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


# Loki writes a log entry's timestamp in nanoseconds and a metric sample's in seconds,
# over the same endpoint, which is the difference the rendering has to notice.
_STREAMS = {
    "status": "success",
    "data": {
        "resultType": "streams",
        "result": [
            {"stream": {"pod": pod}, "values": [[str(SAMPLE_AT * 1_000_000_000), message]]}
            for pod, message in LOG_ENTRIES
        ],
    },
}

_MATRIX = {
    "status": "success",
    "data": {
        "resultType": "matrix",
        "result": [
            {"metric": {"pod": pod}, "values": [[SAMPLE_AT, count]]} for pod, count in LOG_COUNTS
        ],
    },
}


def _logs(request: httpx.Request) -> httpx.Response:
    """Answer a LogQL query in the shape the query asked for."""
    query = request.url.params.get("query", "")
    return httpx.Response(200, json=_MATRIX if "count_over_time" in query else _STREAMS)


class TestDirectTransport(ToolTransportContract):
    @pytest.fixture
    def transport(self, tmp_path: Path) -> Iterator[DirectTransport]:
        with respx.mock(assert_all_called=False) as router:
            router.get(f"{PROMETHEUS}/api/v1/alerts").mock(
                return_value=httpx.Response(
                    200, json={"status": "success", "data": {"alerts": ALERTS}}
                )
            )
            router.get(f"{PROMETHEUS}/api/v1/query").mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "status": "success",
                        "data": {"resultType": "vector", "result": METRIC_SERIES},
                    },
                )
            )
            router.get(f"{LOKI}/loki/api/v1/query_range").mock(side_effect=_logs)
            router.get(f"{JAEGER}/api/services").mock(
                return_value=httpx.Response(200, json={"data": SERVICES})
            )
            router.get(f"{JAEGER}/api/operations").mock(
                return_value=httpx.Response(200, json={"data": OPERATIONS})
            )
            router.get(f"{JAEGER}/api/traces").mock(
                return_value=httpx.Response(200, json={"data": TRACES})
            )
            router.get(f"{JAEGER}/api/dependencies").mock(
                return_value=httpx.Response(200, json=DEPENDENCIES)
            )
            yield DirectTransport(
                TransportSettings(
                    kubectl_binary=str(_kubectl_stub(tmp_path)),
                    prometheus_url=PROMETHEUS,
                    loki_url=LOKI,
                    jaeger_url=JAEGER,
                    timeout_s=30,
                )
            )

    @pytest.fixture
    def unreachable(self) -> Iterator[DirectTransport]:
        """Every endpoint configured and none of them there.

        The three telemetry backends refuse the connection and the kubectl binary is
        absent, which is the shape a deployment takes when its cluster access dies
        mid-incident rather than at start-up.
        """
        with respx.mock(assert_all_called=False) as router:
            router.route().mock(side_effect=httpx.ConnectError("no route to host"))
            yield DirectTransport(
                TransportSettings(
                    kubectl_binary=ABSENT_KUBECTL,
                    prometheus_url=PROMETHEUS,
                    loki_url=LOKI,
                    jaeger_url=JAEGER,
                    timeout_s=30,
                )
            )
