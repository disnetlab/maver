"""The measured role implementations, exposed as inheritable Python agents."""

from maver.agents.base import Agent
from maver.agents.registry import AgentRegistry
from maver.roles.execution import EXECUTION
from maver.roles.learning import LEARNING
from maver.roles.planning import PLANNING
from maver.roles.rca import RCA
from maver.roles.single import SINGLE
from maver.roles.verify import VERIFY


class RCAAgent(Agent):
    """The existing diagnostic agent, including its prompt and context builders."""

    version = "2"
    role_spec = RCA


class PlanningAgent(Agent):
    """The existing planner and its submission guards."""

    role_spec = PLANNING


class ExecutionAgent(Agent):
    """The existing execution agent and its serialized write behaviour."""

    version = "2"
    role_spec = EXECUTION


class VerifyAgent(Agent):
    """The existing blind verifier and deployment health checks."""

    version = "3"
    role_spec = VERIFY


class LearningAgent(Agent):
    """The existing asynchronous experience distiller."""

    role_spec = LEARNING


class SingleAgent(Agent):
    """The existing two-phase single-agent implementation."""

    role_spec = SINGLE


def builtin_registry() -> AgentRegistry:
    """Build a local registry an adopter may extend without global mutation."""
    registry = AgentRegistry()
    for implementation in (
        RCAAgent,
        PlanningAgent,
        ExecutionAgent,
        VerifyAgent,
        LearningAgent,
        SingleAgent,
    ):
        registry.register(implementation.role_spec.name, implementation)
    return registry
