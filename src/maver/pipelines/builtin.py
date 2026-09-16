"""The measured workflows, exposed through the same definitions as local extensions.

Each multi-agent transition delegates to the existing pure predicate. Registration
changes composition, never the retry or verification rules those predicates encode.
"""

from dataclasses import replace

from pydantic import Field

from maver.contracts.artifacts import ExecutionResult, Plan, RootCause, SingleResult, VerifyResult
from maver.contracts.routing import Route
from maver.contracts.state import IncidentState
from maver.orchestration.outcomes import OutcomeRecord, resolve_outcome
from maver.orchestration.routing import (
    RoutingPolicy,
    after_execution,
    after_planning,
    after_rca,
    after_verify,
)
from maver.pipelines.definitions import (
    PipelineRegistry,
    StageDefinition,
    Transition,
    WorkflowConfig,
    WorkflowDefinition,
)

__all__ = ["DIAGNOSE", "MULTI", "SINGLE", "MultiWorkflowConfig", "builtin_registry"]


class MultiWorkflowConfig(WorkflowConfig):
    """Per-pipeline limits that may narrow the deployment's recovery loop."""

    verify_handback_retries: int | None = Field(default=None, ge=0)


def _destination(route: Route) -> str | None:
    """Represent the existing terminal route without a reserved stage name."""
    return None if route is Route.END else route.value


def _rca(state: IncidentState, config: WorkflowConfig, policy: RoutingPolicy) -> str | None:
    """Delegate the diagnostic transition without changing its conditions."""
    return _destination(after_rca(state, policy))


def _planning(state: IncidentState, config: WorkflowConfig, policy: RoutingPolicy) -> str | None:
    """Delegate the planning transition without changing its conditions."""
    return _destination(after_planning(state, policy))


def _execution(state: IncidentState, config: WorkflowConfig, policy: RoutingPolicy) -> str | None:
    """Delegate the execution transition without changing its conditions."""
    return _destination(after_execution(state, policy))


def _verify(state: IncidentState, config: WorkflowConfig, policy: RoutingPolicy) -> str | None:
    """Apply the workflow's optional tighter cap to the shared recovery policy."""
    if isinstance(config, MultiWorkflowConfig) and config.verify_handback_retries is not None:
        policy = replace(
            policy,
            verify_handback_retries=min(
                policy.verify_handback_limit, config.verify_handback_retries
            ),
        )
    return _destination(after_verify(state, policy))


def _end(state: IncidentState, config: WorkflowConfig, policy: RoutingPolicy) -> None:
    """End a one-role workflow after its role has completed."""
    return None


def _diagnosis_outcome(state: IncidentState) -> OutcomeRecord:
    """Treat explicitly selecting this workflow as a request for diagnosis only."""
    return resolve_outcome({**state, "target_stage": "diagnose"})


_RCA = StageDefinition("rca", "rca", RootCause, "root_cause", "rca")

MULTI = WorkflowDefinition(
    name="multi",
    version="2",
    config_type=MultiWorkflowConfig,
    entry="rca",
    stages=(
        _RCA,
        StageDefinition("planning", "planning", Plan, "plan", "planning"),
        StageDefinition("execution", "execution", ExecutionResult, "execution_result", "execution"),
        StageDefinition("verify", "verify", VerifyResult, "verify_result", "verify"),
    ),
    transitions={
        "rca": Transition(("planning", None), _rca),
        "planning": Transition(("execution", None), _planning),
        "execution": Transition(("verify", "planning", None), _execution),
        "verify": Transition(("rca", "planning", "verify", None), _verify),
    },
)

SINGLE = WorkflowDefinition(
    name="single",
    version="1",
    entry="single",
    stages=(StageDefinition("single", "single", SingleResult, None, "single"),),
    transitions={"single": Transition((None,), _end)},
)

DIAGNOSE = WorkflowDefinition(
    name="diagnose",
    version="1",
    entry="rca",
    stages=(_RCA,),
    transitions={"rca": Transition((None,), _end)},
    outcome=_diagnosis_outcome,
    post_hooks=False,
)


def builtin_registry() -> PipelineRegistry:
    """Build an independent registry users can extend in deployment assembly code."""
    registry = PipelineRegistry()
    for workflow in (MULTI, SINGLE, DIAGNOSE):
        registry.register(workflow)
    return registry
