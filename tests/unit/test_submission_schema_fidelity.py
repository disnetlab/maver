"""The model-facing half of the submission schemas, pinned to the original.

A class docstring on a submission model would otherwise become the schema's top-level
``description`` and therefore part of the instructions every role receives, on a
contract that is held byte-compatible with the research prototype. Rewriting these
classes for readability once added a description to six schemas that had none and
reworded three that had one, and nothing failed.

The expected values below are the descriptions the prototype's ``agent/state.py``
actually shipped, extracted from it directly. Six are None because those classes
carried no docstring at all.
"""

import pytest
from pydantic import BaseModel

from kira.contracts import (
    ExecutionResult,
    LearningResult,
    Plan,
    PlanStep,
    RootCause,
    SingleDiagnosis,
    SingleResult,
    StepResult,
    SubmissionModel,
    VerifyResult,
)

EXPECTED: dict[type[BaseModel], str | None] = {
    RootCause: None,
    PlanStep: None,
    Plan: None,
    StepResult: None,
    ExecutionResult: None,
    LearningResult: None,
    VerifyResult: "End-to-end recovery verdict from the Verify step.\n\nVerify runs AFTER a clean apply and judges the incident at the SYSTEM level —\nnot whether a step applied (Execution already confirmed that), but whether the\naffected request path actually works now. Pods being Running/Ready is not\nenough: a request must be able to flow through. Verify owns the decision to end\nthe pipeline (recovered) or hand back (replan / reassess).",
    SingleDiagnosis: "Free-text diagnosis from the single-agent baseline (AGENT_MODE=single).\n\nDeliberately unstructured: the minimal baseline submits a natural-language\nroot-cause description — exactly as SREGym's reference agents do — rather than\nthe structured RootCause the multi-agent pipeline produces. Stored in the\n`root_cause` state slot; persistence and outcome resolution duck-type it (it\nhas no `halt`, so the pipeline treats it as a normal diagnosis).",
    SingleResult: "Terminal result from the single-agent baseline's mitigation phase.\n\nStored in the `execution_result` state slot. `succeeded` is the field\nresolve_pipeline_outcome reads; SREGym grades cluster health independently.",
}


@pytest.mark.parametrize("model", list(EXPECTED), ids=lambda m: m.__name__)
def test_the_schema_description_matches_what_the_prototype_shipped(
    model: type[BaseModel],
) -> None:
    assert model.model_json_schema().get("description") == EXPECTED[model]


@pytest.mark.parametrize("model", list(EXPECTED), ids=lambda m: m.__name__)
def test_a_docstring_never_reaches_the_schema(model: type[BaseModel]) -> None:
    """Documentation and instruction are separate channels, structurally.

    Every one of these classes has a docstring for human readers. None of them may
    appear in the schema, so editing one cannot change what a role is told.
    """
    doc = (model.__doc__ or "").strip()
    description = model.model_json_schema().get("description")
    assert doc, f"{model.__name__} should still document itself for a reader"
    assert description != doc


def test_a_new_submission_model_ships_no_description_by_accident() -> None:
    """The guarantee is structural, not a convention someone has to remember."""

    class Invented(SubmissionModel):
        """A docstring a contributor added in good faith."""

        field: str = "x"

    assert Invented.model_json_schema().get("description") is None


def test_an_explicit_description_is_still_emitted() -> None:
    class Declared(SubmissionModel):
        """Documentation for a reader."""

        schema_description = "Instruction for the model."
        field: str = "x"

    assert Declared.model_json_schema()["description"] == "Instruction for the model."


def test_nested_step_models_add_nothing_at_the_items_level() -> None:
    """The registry inlines definitions, so a nested description lands on items."""
    for model, field in ((Plan, "steps"), (ExecutionResult, "steps")):
        schema = model.model_json_schema()
        ref = schema["properties"][field]["items"]["$ref"].rsplit("/", 1)[-1]
        assert "description" not in schema["$defs"][ref]
