"""The two engines, and the seeding rule they start from."""

from typing import Any

import pytest

from maver.contracts.artifacts import ExecutionResult, Plan, PlanStep, RootCause, VerifyResult
from maver.contracts.state import IncidentState
from maver.orchestration.engine import (
    STEP_LIMIT,
    Engine,
    LangGraphEngine,
    SequentialEngine,
    StepLimitExceeded,
)
from maver.orchestration.graph import build_graph
from maver.orchestration.modes import resolve_mode
from maver.orchestration.routing import RoutingPolicy, apply_handback, pending_handback
from maver.orchestration.state import initial_state
from maver.settings.groups import PipelineSettings
from maver.settings.root import Settings

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


def pipeline_nodes(seen: list[str]) -> dict[str, Any]:
    """Four fake roles walking one interesting path: a verification that first cannot
    tell, looks again, and then confirms. Each records that it ran, so the route is
    read off the roles rather than off the executor being compared."""

    async def rca(state: IncidentState) -> dict[str, Any]:
        seen.append("rca")
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
        seen.append("planning")
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
        seen.append("execution")
        return {
            "execution_result": ExecutionResult(
                succeeded=True,
                steps=[],
                final_state_summary="patched",
                confidence=0.8,
            )
        }

    async def verify(state: IncidentState) -> dict[str, Any]:
        seen.append("verify")
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


@pytest.mark.asyncio
async def test_both_engines_reach_the_same_end_by_the_same_route() -> None:
    """The cheapest evidence for the containment claim.

    If the framework disappeared, deleting graph.py and defaulting to the sequential
    engine would be the whole migration. This is the test that says so.
    """
    graph_stages: list[str] = []
    walk_stages: list[str] = []
    graphed = await LangGraphEngine(build_graph(pipeline_nodes(graph_stages), POLICY)).run(seeded())
    walked = await SequentialEngine(pipeline_nodes(walk_stages), POLICY).run(seeded())

    assert graph_stages == walk_stages == ["rca", "planning", "execution", "verify", "verify"]
    assert graphed["verify_result"] == walked["verify_result"]
    assert graphed["root_cause"] == walked["root_cause"]
    assert graphed["plan"] == walked["plan"]
    assert graphed["execution_result"] == walked["execution_result"]


@pytest.mark.asyncio
@pytest.mark.parametrize("engine", ["sequential", "langgraph"])
async def test_a_cycling_configuration_is_stopped(engine: str) -> None:
    """A verification that can always look again would never end.

    Both engines, because the bound is ours rather than either executor's. The graph
    engine passed none, so it inherited the framework's 10007 and a routing bug ran ten
    thousand role runs — each a model call, and in the remediating role a cluster
    mutation — where the sequential engine stopped at twenty-five.
    """
    nodes = pipeline_nodes([])
    runs: list[str] = []

    async def never_sure(state: IncidentState) -> dict[str, Any]:
        runs.append("verify")
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
    running: Engine = (
        SequentialEngine(nodes, generous)
        if engine == "sequential"
        else LangGraphEngine(build_graph(nodes, generous))
    )

    with pytest.raises(StepLimitExceeded):
        await running.run(seeded())
    assert len(runs) < STEP_LIMIT


# --- mode dispatch -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("target_stage", "configured", "expected"),
    [
        ("full", "multi", "multi"),
        ("full", "single", "single"),
        # A diagnosis-only task is a request about what may be done to the environment,
        # and no configured mode overrides it. In the prototype this held only because
        # its branch happened to be written first.
        ("diagnose", "multi", "diagnose"),
        ("diagnose", "single", "diagnose"),
    ],
)
def test_a_diagnosis_only_task_outranks_the_configured_mode(
    target_stage: Any, configured: str, expected: str
) -> None:
    settings = Settings(pipeline=PipelineSettings(mode=configured))  # type: ignore[arg-type]
    assert resolve_mode(target_stage=target_stage, settings=settings) == expected


def _reassessing_pipeline(charged: list[str]) -> dict[str, Any]:
    """Roles that go once around the loop the hard way.

    Verification asks for a re-diagnosis; the new plan then fails to apply and asks for a
    replan. Every node runs the real hand-back accounting, so what is being exercised is
    the budget, not the fakes.
    """

    def cause(summary: str) -> RootCause:
        return RootCause(
            fault_type="misconfig",
            affected_resource="deploy/web",
            confidence=0.8,
            evidence=["readiness probe fails"],
            summary=summary,
        )

    async def rca(state: IncidentState) -> dict[str, Any]:
        delta = apply_pending(state, "rca")
        charged.append("rca")
        return {**delta, "root_cause": cause(f"attempt {len(charged)}")}

    async def planning(state: IncidentState) -> dict[str, Any]:
        delta = apply_pending(state, "planning")
        charged.append("planning")
        return {
            **delta,
            "plan": Plan(
                steps=[
                    PlanStep(
                        action="patch",
                        target="deploy/web",
                        params={},
                        justification="repoint the probe",
                    )
                ],
                expected_outcome="the probe passes",
                confidence=0.8,
            ),
        }

    async def execution(state: IncidentState) -> dict[str, Any]:
        delta = apply_pending(state, "execution")
        charged.append("execution")
        # The first apply is clean, which is the only way the pipeline reaches
        # verification at all. Every later one fails and asks for a replan, so whether
        # the incident terminates is entirely a question of whether the budget falls.
        first = charged.count("execution") == 1
        return {
            **delta,
            "execution_result": ExecutionResult(
                succeeded=first,
                steps=[],
                final_state_summary="applied" if first else "the patch was rejected",
                confidence=0.5,
                needs_replan=not first,
                replan_reason=None if first else "the target does not exist",
            ),
        }

    async def verify(state: IncidentState) -> dict[str, Any]:
        charged.append("verify")
        return {
            "verify_result": VerifyResult(
                recovered=False,
                verdict="not_recovered",
                checks=["the request path still 503s"],
                summary="the symptom persists",
                confidence=0.8,
                needs_reassessment=True,
                reassessment_reason="the cause was somewhere else",
            )
        }

    return {"rca": rca, "planning": planning, "execution": execution, "verify": verify}


def apply_pending(state: IncidentState, role: str) -> dict[str, Any]:
    """The hand-back accounting the real node wrapper does before a role runs.

    Including the part that matters here: a hand-back addressed to another role is not
    this role's to act on, and charges nothing.
    """
    handback = pending_handback(state)
    if handback is None or handback.target.value != role:
        return {}
    return dict(apply_handback(state, handback, POLICY))


async def test_a_consumed_verdict_stops_being_a_pending_instruction() -> None:
    """A hand-back is derived from the artifacts, so a verdict left in place is re-derived
    at every later node entry.

    A reassessment kept naming `rca`. Planning therefore read a hand-back addressed to
    someone else, treated it as no hand-back at all, and charged nothing — so the budget
    never fell and planning and execution cycled until the step limit killed the incident,
    each cycle another mutation of the cluster. The incident ended in an exception rather
    than an outcome, which is not a result anything downstream can read.
    """
    charged: list[str] = []
    state = seeded(
        root_cause=None,
        remediation_retries=0,
    )

    final = await SequentialEngine(_reassessing_pipeline(charged), POLICY).run(state)

    # Two cycles is the configured budget. Without the clear this raised StepLimitExceeded
    # after twenty-five role runs.
    assert final["remediation_retries"] == POLICY.remediation_retries
    assert charged.count("execution") <= POLICY.remediation_retries + 1
    assert final["verify_result"] is None


async def test_a_node_that_recursed_away_its_stack_is_not_a_routing_loop() -> None:
    """The executor's own overflow subclasses the builtin, which is how it is
    recognised without importing the framework this module is kept free of — but so
    does a node that genuinely recursed, and relabelling that as "exceeded 25 role
    runs" sends whoever reads the incident to the edge table for a bug that is not
    there."""

    async def deep(state: IncidentState) -> dict[str, Any]:
        def recurse(depth: int) -> int:
            return recurse(depth + 1)

        return {"error": str(recurse(0))}

    nodes = pipeline_nodes([]) | {"rca": deep}

    with pytest.raises(RecursionError) as raised:
        await LangGraphEngine(build_graph(nodes, POLICY)).run(seeded())
    assert not isinstance(raised.value, StepLimitExceeded)
