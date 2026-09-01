"""Every word the model reads from a submission schema, pinned to the prototype.

The submission schemas are the frozen interface between the roles, and their text is
not documentation: it ships to the model as the JSON Schema of its submission tool, so
an edit is a change to the instructions every role receives on a contract held
byte-compatible with the research prototype.

Two channels carry that text and both drifted while these models were rewritten for
readability. Field descriptions lost clauses: the sentence separating case constraints
from cross-cutting playbook rules, the requirement clause on a verification verdict,
and a summary field's word limit, its layer label and the tool that fetches it. Class
docstrings became descriptions on six schemas that had none and were reworded on three
that had one. Nothing failed, because nothing was watching.

The fixture beside this test is extracted directly from the prototype's ``state.py``.
A change here is a research-validity decision, not a refactor: see STYLE_GUIDE.md
section 3 for what it requires.
"""

import json
import pathlib

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

FIXTURE = json.loads(
    (
        pathlib.Path(__file__).resolve().parents[1] / "fixtures" / "prototype_submission_text.json"
    ).read_text(encoding="utf-8")
)

MODELS: dict[str, type[BaseModel]] = {
    "RootCause": RootCause,
    "PlanStep": PlanStep,
    "Plan": Plan,
    "StepResult": StepResult,
    "ExecutionResult": ExecutionResult,
    "VerifyResult": VerifyResult,
    "SingleDiagnosis": SingleDiagnosis,
    "SingleResult": SingleResult,
    "LearningResult": LearningResult,
}

FIELD_CASES = [
    (name, field, text)
    for name, entry in FIXTURE.items()
    for field, text in entry["fields"].items()
]


@pytest.mark.parity
def test_the_fixture_covers_every_submission_model() -> None:
    assert set(FIXTURE) == set(MODELS)


@pytest.mark.parity
@pytest.mark.parametrize("name", sorted(MODELS), ids=sorted(MODELS))
def test_the_class_level_description_matches_the_prototype(name: str) -> None:
    """Six of these shipped no description at all. That is the value to preserve."""
    expected = FIXTURE[name]["class_description"]
    assert MODELS[name].model_json_schema().get("description") == expected


@pytest.mark.parity
@pytest.mark.parametrize(
    ("name", "field", "expected"), FIELD_CASES, ids=[f"{n}.{f}" for n, f, _ in FIELD_CASES]
)
def test_every_field_description_matches_the_prototype(
    name: str, field: str, expected: str | None
) -> None:
    info = MODELS[name].model_fields[field]
    assert info.description == expected


@pytest.mark.parity
@pytest.mark.parametrize("name", sorted(MODELS), ids=sorted(MODELS))
def test_no_field_was_added_or_removed(name: str) -> None:
    """A field added here changes what validates, on a schema held constant."""
    assert set(MODELS[name].model_fields) == set(FIXTURE[name]["fields"])


@pytest.mark.parametrize("name", sorted(MODELS), ids=sorted(MODELS))
def test_a_docstring_never_reaches_the_schema(name: str) -> None:
    """Documentation and instruction are separate channels, structurally."""
    model = MODELS[name]
    doc = (model.__doc__ or "").strip()
    assert doc, f"{name} should still document itself for a reader"
    assert model.model_json_schema().get("description") != doc


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


@pytest.mark.parametrize(("model", "field"), [(Plan, "steps"), (ExecutionResult, "steps")])
def test_nested_step_models_add_nothing_at_the_items_level(
    model: type[BaseModel], field: str
) -> None:
    """A tool registry inlines definitions, so a nested description lands on items."""
    schema = model.model_json_schema()
    ref = schema["properties"][field]["items"]["$ref"].rsplit("/", 1)[-1]
    assert "description" not in schema["$defs"][ref]


@pytest.mark.parity
def test_only_the_measured_container_repairs_are_attached() -> None:
    """The tolerant decoders exist where a measured failure justified them, nowhere else.

    Adding one elsewhere enlarges the set of submissions that validate: a payload that
    would have been rejected and re-prompted instead succeeds, and a record gets written
    where the prototype wrote none.
    """
    expected = {
        ("RootCause", "evidence"),
        ("PlanStep", "params"),
        ("Plan", "steps"),
        ("ExecutionResult", "steps"),
        ("VerifyResult", "checks"),
    }
    found: set[tuple[str, str]] = set()
    for name, model in MODELS.items():
        for decorator in model.__pydantic_decorators__.field_validators.values():
            if decorator.func.__name__.startswith("_decode"):
                found.update((name, field) for field in decorator.info.fields)
    assert found == expected
