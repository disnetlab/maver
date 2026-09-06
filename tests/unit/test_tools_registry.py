import json
import pathlib

import pytest

import kira.contracts.artifacts as artifacts
from kira.tools.registry import (
    ToolRegistryError,
    ToolSpec,
    definition_for,
    description_for,
    validate_registry,
)
from kira.tools.schema import RATIONALE_ARG, normalise_schema, with_rationale

SCHEMAS = json.loads(
    (
        pathlib.Path(__file__).resolve().parents[1] / "fixtures" / "prototype_submit_schemas.json"
    ).read_text(encoding="utf-8")
)


def read_only(name: str, **kwargs: object) -> ToolSpec:
    async def fn(ctx: object, **_: object) -> str:
        return "ok"

    return ToolSpec(
        name=name,
        parameters={"type": "object", "properties": {}},
        safety="read_only",
        dispatch=fn,
        **kwargs,  # type: ignore[arg-type]
    )


# --- what the model is sent -------------------------------------------------------


@pytest.mark.parity
@pytest.mark.parametrize("name", sorted(SCHEMAS))
def test_submit_schemas_normalise_exactly_as_the_prototype(name: str) -> None:
    """The schema a submission validates against is a held-constant input.

    This pins the artifact model and the normaliser together: the fixture is the
    prototype's own normaliser run over the prototype's own models, so a drift in either
    fails here. The rewrites exist because Gemini's validator returns INTERNAL 500 on
    ``$ref``/``$defs`` and on the ``anyOf`` wrapper Pydantic emits for optional fields.
    """
    model = getattr(artifacts, name)
    assert normalise_schema(model.model_json_schema()) == SCHEMAS[name]


def test_a_genuine_union_is_not_collapsed() -> None:
    """Only a nullable wrapper collapses. A real two-type union must survive."""
    schema = {"anyOf": [{"type": "string"}, {"type": "integer"}]}
    assert normalise_schema(schema) == schema


def test_the_rationale_is_offered_first_and_required() -> None:
    """The model reads the schema in order, and writes the trace before the call.

    The rationale carries working state inside the tool call, which is the only channel
    that survives to the next turn: a reasoning model keeps its visible planning in a
    hidden channel the loop discards, but tool-call arguments are always echoed back.
    """
    built = with_rationale(
        {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
    )
    assert next(iter(built["properties"])) == RATIONALE_ARG
    assert built["required"] == [RATIONALE_ARG, "query"]


def test_a_submission_carries_no_rationale() -> None:
    """Submit tools validate against a strict model and carry their own summary."""
    spec = ToolSpec(name="submit_root_cause", parameters={"type": "object"}, safety="submit")
    assert RATIONALE_ARG not in (definition_for(spec).get("properties") or {})


# --- rules that would otherwise fail quietly --------------------------------------


def test_a_misplaced_role_description_is_rejected() -> None:
    """A role name written beside `default` instead of under `agents` is a typo.

    The prototype had exactly this: a carefully written description for one role sat as
    a sibling key, nothing rejected it, and that role was served the generic text for
    the life of the system. Nothing failed and nothing logged.
    """
    with pytest.raises(ToolRegistryError, match="unknown keys"):
        validate_registry(
            descriptions={"get_logs": {"default": "read logs", "verify": "read logs, for verify"}},
            allowlists={"verify": ("get_logs",)},
            registry={"get_logs": read_only("get_logs")},
        )


def test_a_description_for_a_role_that_cannot_call_the_tool_is_rejected() -> None:
    """Writing one is a sign the allowlist and the catalogue disagree about the design."""
    with pytest.raises(ToolRegistryError, match="not allowed to call it"):
        validate_registry(
            descriptions={"get_logs": {"default": "read logs", "agents": {"planning": "…"}}},
            allowlists={"rca": ("get_logs",)},
            registry={"get_logs": read_only("get_logs")},
        )


def test_a_mistyped_capability_is_rejected() -> None:
    """A typo would silently drop the tool from every role, in every run.

    In the results that is indistinguishable from a model that chose not to call it.
    """
    with pytest.raises(ToolRegistryError, match="unknown transport capabilities"):
        validate_registry(
            descriptions={"get_logs": {"default": "read logs"}},
            allowlists={"rca": ("get_logs",)},
            registry={"get_logs": read_only("get_logs", requires=frozenset({"logz"}))},
        )


def test_a_write_capable_tool_must_register_the_policy_the_gate_reads() -> None:
    """The check has to test the registration the gate actually consults.

    It tested a field on the spec that nothing read, so a write tool declared exactly as
    the docstrings prescribed passed at import and was refused at incident time by every
    stage — including the one its author had allowed — because the gate looks the tool
    up in the policy registry and treats a miss as a write nobody may issue.
    """
    spec = ToolSpec(
        name="apply", parameters={"type": "object"}, safety="cluster_write", dispatch=lambda: None
    )
    with pytest.raises(ToolRegistryError, match="must register a policy"):
        validate_registry(
            descriptions={"apply": {"default": "apply a manifest"}},
            allowlists={"execution": ("apply",)},
            registry={"apply": spec},
        )


def test_registry_and_descriptions_must_match_exactly() -> None:
    """An undescribed tool reaches the model with no instructions at all."""
    with pytest.raises(ToolRegistryError, match="undescribed tools"):
        validate_registry(
            descriptions={},
            allowlists={"rca": ("get_logs",)},
            registry={"get_logs": read_only("get_logs")},
        )


def test_a_per_role_description_replaces_the_default_outright() -> None:
    """It is a replacement, not a suffix, so a role-specific text must be complete."""
    descriptions = {"get_logs": {"default": "generic", "agents": {"rca": "for rca"}}}
    assert description_for("get_logs", "rca", descriptions=descriptions) == "for rca"
    assert description_for("get_logs", "verify", descriptions=descriptions) == "generic"
