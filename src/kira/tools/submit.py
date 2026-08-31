"""The terminal tools, one per artifact a role produces.

A submit tool is how a role ends. It has no implementation: the loop intercepts the call,
validates the arguments against the artifact model, and returns them as the role's
output. That is why these are registered separately — a submit tool with a dispatch
function would be a tool that could run and then not end the run.

The schemas come from the artifact models, so the thing a role is asked to produce and
the thing the next role receives cannot drift apart. They are held constant across every
experiment configuration: a comparison between configurations is only single-variable if
the submission schema does not move with it.
"""

from pydantic import BaseModel

from kira.contracts.artifacts import (
    ExecutionResult,
    LearningResult,
    Plan,
    RootCause,
    SingleDiagnosis,
    SingleResult,
    VerifyResult,
)
from kira.tools.registry import register_submit_tool
from kira.tools.schema import normalise_schema

__all__ = ["SUBMIT_TOOLS", "register_all"]

SUBMIT_TOOLS: dict[str, type[BaseModel]] = {
    "submit_root_cause": RootCause,
    "submit_plan": Plan,
    "submit_execution_result": ExecutionResult,
    "submit_verify_result": VerifyResult,
    "submit_learning_result": LearningResult,
    "submit_diagnosis": SingleDiagnosis,
    "submit_mitigation": SingleResult,
}
"""Which artifact each role's terminal tool produces."""


def register_all() -> None:
    """Register every submit tool.

    Called once when the tools package is imported, alongside the capability modules.
    """
    for name, model in SUBMIT_TOOLS.items():
        register_submit_tool(name=name, parameters=normalise_schema(model.model_json_schema()))
