"""The two engines, and the seeding rule they start from."""

from typing import Any

import pytest

from kira.contracts.artifacts import ExecutionResult, Plan, PlanStep, RootCause, VerifyResult
from kira.contracts.state import IncidentState
from kira.orchestration.engine import (
    Engine,
    LangGraphEngine,
    SequentialEngine,
    StepLimitExceeded,
)
from kira.orchestration.graph import build_graph
from kira.orchestration.routing import RoutingPolicy
from kira.orchestration.state import initial_state

POLICY = RoutingPolicy(verify_enabled=True, remediation_retries=2, inconclusive_retries=1)


def seeded(**overrides: Any) -> IncidentState:
    state = initial_state(
        session_id="s1",
        alert={"name": "HighLatency"},
        target_stage="full",
        incident_dir="/tmp/incident",  # type: ignore[arg-type]
        shared_dir="/tmp/shared",  # type: ignore[arg-type]
    )
    state.update(overrides)  # type: ignore[typeddict-item]
    return state


def test_initial_state_seeds_every_declared_key() -> None:
    """A missing key and a key holding a default read the same to a caller.

    The prototype seeded eleven of nineteen, which is how two keys came to be read
    before anything had written them.
    """
    assert set(seeded()) == set(IncidentState.__annotations__)


# --- the two engines agree ---------------------------------------------------------


def pipeline_nodes() -> dict[str, Any]:
    """Four fake roles walking one interesting path: a verification that first cannot
    tell, looks again, and then confirms."""

    async def rca(state: IncidentState) -> dict[str, Any]:
        return {
            "root_cause": RootCause(
                fault_type="misconfig",
                affected_resource="deploy/web",
                confidence=0.8,
                evidence=["readiness probe fails"],
                summary="the probe points at the wrong port",
            )
        }

    async def planning(state: IncidentState) -> dict[str, Any]:
        return {
            "plan": Plan(
                steps=[
                    PlanStep(action="patch", target="deploy/web", justification="repoint the probe")
                ],
                expected_outcome="the probe passes",
                confidence=0.8,
            )
        }

    async def execution(state: IncidentState) -> dict[str, Any]:
        return {
            "execution_result": ExecutionResult(
                succeeded=True,
                steps=[],
                final_state_summary="patched",
                confidence=0.8,
            )
        }

    async def verify(state: IncidentState) -> dict[str, Any]:
        first = state.get("verify_result") is None
        # The wrapper charges the re-check in a real run; here the node does it, so the
        # test stays about the engine rather than about what surrounds a role.
        return {
            "verify_result": VerifyResult(
                recovered=not first,
                verdict="inconclusive" if first else "recovered",
                checks=["curl /health"],
                summary="telemetry gap" if first else "healthy",
                entry_point_status="" if first else "GET / through the gateway returned 200",
                confidence=0.7,
            ),
            "verify_retries": 0 if first else 1,
        }

    return {"rca": rca, "planning": planning, "execution": execution, "verify": verify}


async def drive(engine: Engine) -> tuple[IncidentState, list[str]]:
    seen: list[str] = []

    async def on_stage(name: str, state: IncidentState) -> None:
        seen.append(name)

    return await engine.run(seeded(), on_stage=on_stage), seen


@pytest.mark.asyncio
async def test_both_engines_reach_the_same_end_by_the_same_route() -> None:
    """The cheapest evidence for the containment claim.

    If the framework disappeared, deleting graph.py and defaulting to the sequential
    engine would be the whole migration. This is the test that says so.
    """
    nodes = pipeline_nodes()
    graphed, graph_stages = await drive(LangGraphEngine(build_graph(nodes, POLICY)))
    walked, walk_stages = await drive(SequentialEngine(nodes, POLICY))

    assert graph_stages == walk_stages == ["rca", "planning", "execution", "verify", "verify"]
    assert graphed["verify_result"] == walked["verify_result"]
    assert graphed["root_cause"] == walked["root_cause"]
    assert graphed["plan"] == walked["plan"]
    assert graphed["execution_result"] == walked["execution_result"]


@pytest.mark.asyncio
async def test_the_unstreamed_path_returns_the_same_state() -> None:
    """With no hook the graph is invoked rather than streamed, and the two paths must
    not disagree about what the incident ended as."""
    nodes = pipeline_nodes()
    invoked = await LangGraphEngine(build_graph(nodes, POLICY)).run(seeded())
    streamed, _ = await drive(LangGraphEngine(build_graph(nodes, POLICY)))
    assert invoked["verify_result"] == streamed["verify_result"]


@pytest.mark.asyncio
async def test_a_cycling_configuration_is_stopped() -> None:
    """A verification that can always look again would never end. The bound is matched
    to the executor's own default so a routing bug is caught in either engine."""
    nodes = pipeline_nodes()

    async def never_sure(state: IncidentState) -> dict[str, Any]:
        return {
            "verify_result": VerifyResult(
                recovered=False,
                verdict="inconclusive",
                checks=["curl /health"],
                summary="telemetry gap",
                confidence=0.3,
            )
        }

    nodes["verify"] = never_sure
    generous = RoutingPolicy(verify_enabled=True, remediation_retries=2, inconclusive_retries=99)
    with pytest.raises(StepLimitExceeded):
        await SequentialEngine(nodes, generous).run(seeded())
