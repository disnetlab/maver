"""The adapter object: what it builds once, and what it re-points per incident.

This adapter is bound for the whole process rather than per episode, so the two things
that would otherwise be constructor arguments — which incident's namespace the bridge is
serving, and which role is calling — are set at every role entry instead. That is the
only mutable state in the adapter, and it is what these tests are about.
"""

import pathlib
from typing import Any

from pydantic import SecretStr

from maver.adapters.aiopslab import ADAPTER, ADAPTER_NAME
from maver.adapters.aiopslab.adapter import AiopsLabAdapter
from maver.adapters.aiopslab.config import AiopsLabConfig
from maver.settings.authority import AuthoritySettings
from maver.settings.groups import ServerSettings
from maver.settings.root import Settings

TOKEN = "a-bridge-token-long-enough"
NAMESPACE = "astronomy-shop"


def built(**overrides: Any) -> AiopsLabAdapter:
    config = AiopsLabConfig(enabled=True, bridge_token=SecretStr(TOKEN), **overrides)
    return AiopsLabAdapter(config, authority_mode="auto")


def state(task: str = "mitigation", *, benchmark: str = ADAPTER_NAME) -> Any:
    return {
        "session_id": "session-7",
        "alert": {"benchmark": benchmark, "benchmark_task": task, "namespace": NAMESPACE},
    }


def test_the_transport_is_built_from_the_configuration_and_then_kept() -> None:
    """One transport for the whole process, which is what makes the per-incident
    re-pointing below necessary in the first place."""
    adapter = built(bridge_url="http://127.0.0.1:2000/", bridge_timeout_s=45.0)

    transport = adapter.tool_transport()

    assert transport is not None
    assert transport.name == ADAPTER_NAME
    assert transport.timeout_seconds == 45
    assert adapter.tool_transport() is not transport, "a second call rebuilds"
    # The one the adapter kept is the one the last call returned, so an overlay points
    # at what the deployment is actually using.
    assert adapter.role_overlay("execution", state()) is not None


def test_each_role_entry_points_the_bridge_at_the_incident_it_is_serving() -> None:
    """The bridge refuses any action whose namespace is not the active problem's, so an
    unset value fails closed rather than reaching another application's telemetry."""
    adapter = built()
    transport = adapter.tool_transport()
    assert transport is not None

    adapter.role_overlay("verify", state())

    assert transport.namespace == NAMESPACE  # type: ignore[attr-defined]
    assert transport.stage == "verify"  # type: ignore[attr-defined]
    # Correlates one agent-side audit event with one host-side access-log line; without
    # it the two logs share only a wall clock.
    assert transport.session_id == "session-7"  # type: ignore[attr-defined]

    adapter.role_overlay("execution", state())
    assert transport.stage == "execution"  # type: ignore[attr-defined]


def test_an_incident_from_elsewhere_is_neither_overlaid_nor_allowed_to_move_the_bridge() -> None:
    """A process-bound adapter sees every incident, including ones that did not come
    from its driver. Re-pointing the bridge at one of those would take the transport
    away from the problem AIOpsLab actually has deployed."""
    adapter = built()
    transport = adapter.tool_transport()
    assert transport is not None
    transport.namespace = "test-social-network"  # type: ignore[attr-defined]
    transport.stage = "rca"  # type: ignore[attr-defined]

    for role in ("rca", "planning", "execution", "verify"):
        assert adapter.role_overlay(role, state(benchmark="sregym")) is None

    assert transport.namespace == "test-social-network"  # type: ignore[attr-defined]
    assert transport.stage == "rca"  # type: ignore[attr-defined]


def test_the_environments_own_tools_travel_with_it() -> None:
    """The core's allowlists must not carry the name of an environment's tool, so the
    role each is offered to arrives with the tool."""
    offered = {tool.spec.name: tool.roles for tool in built().extra_tools()}

    assert offered == {
        "probe_service": frozenset({"verify"}),
        "restore_mongodb_access": frozenset({"execution"}),
    }


def test_the_deployments_authority_decides_what_the_remediating_role_is_told(
    tmp_path: pathlib.Path,
) -> None:
    """Read at composition rather than at incident time: a role told to wait for a human
    on an unattended run spends its budget waiting for nobody, and the mode that decides
    that is the deployment's, not the adapter's."""
    slice_: dict[str, Any] = {"enabled": True, "bridge_token": TOKEN}

    def suffix(mode: str) -> str:
        settings = Settings(
            server=ServerSettings(workdir=tmp_path),
            adapters={ADAPTER_NAME: slice_},
            authority=AuthoritySettings(mode=mode),  # type: ignore[arg-type]
        )
        overlay = ADAPTER(settings).role_overlay("execution", state())
        assert overlay is not None
        return overlay.system_suffix

    assert "This run is non-interactive" in suffix("auto")
    assert "Authorization is handled by the runtime" in suffix("prompt")
