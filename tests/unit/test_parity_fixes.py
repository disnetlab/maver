"""Behaviours restored after a parity audit found them missing or diverged.

Each of these was carried by the prototype, lost in the rebuild, and would have changed
a graded run without raising anything.
"""

import pytest

from kira.contracts.artifacts import RootCause
from kira.llm.client import serialise_assistant
from kira.llm.types import ModelResponse, ToolCall, Usage
from kira.roles.policy import repair_submission
from kira.tools.context import NullEvidencePolicy, redact


def response(**kw) -> ModelResponse:
    kw.setdefault("text", None)
    kw.setdefault("tool_calls", ())
    kw.setdefault("finish_reason", "length")
    kw.setdefault("reasoning", None)
    kw.setdefault("usage", Usage())
    kw.setdefault("duration_ms", 1)
    kw.setdefault("attempts", 1)
    return ModelResponse(**kw)


def test_a_confidence_out_of_range_is_clipped_not_handed_back() -> None:
    """Handing it back costs one of three shared retries, and three end the stage.

    A model writing 1.4 instead of 0.9 would lose the whole cell, and the loss would be
    indistinguishable in the results from a model that had nothing to say.
    """
    repaired, repairs = repair_submission({"confidence": 1.4, "summary": "s"})
    assert repaired["confidence"] == 1.0
    assert repairs == [
        {"field": "confidence", "original": 1.4, "repaired": 1.0, "reason": "clipped_to_0_1"}
    ]


@pytest.mark.parametrize("value", [0.0, 0.5, 1.0, "not a number", None])
def test_only_an_out_of_range_number_is_repaired(value: object) -> None:
    """The repair is deliberately narrow. Anything else is a real validation failure."""
    assert repair_submission({"confidence": value})[1] == []


def test_reasoning_survives_a_truncated_tool_call() -> None:
    """The nudge for this turn tells the model to continue from the state it was cut off
    in. Without the fold that state is just the discard note, which is the stall the
    fold exists to prevent — on the one branch its own wording does not cover."""
    message = serialise_assistant(
        response(
            tool_calls=(ToolCall(id="c1", name="get_logs", arguments={}, raw_arguments='{"q":'),),
            reasoning="I was checking whether the probe port matches the container port",
        )
    )
    assert "probe port matches the container port" in message["content"]
    assert not message.get("tool_calls"), "the partial call must still be dropped"


def test_an_artifact_rejects_a_field_it_does_not_declare() -> None:
    """Pydantic's default discards it silently, so a schema drifting from what a role is
    prompted to produce looks exactly like a clean submission."""
    with pytest.raises(Exception, match="invented_field"):
        RootCause(
            fault_type="misconfig",
            affected_resource="deploy/web",
            confidence=0.5,
            evidence=["e"],
            summary="s",
            invented_field=1,
        )


def test_forbidding_extras_does_not_change_the_schema_the_model_is_sent() -> None:
    """Rejection is a validation decision. The schema is a measured input, and it already
    lists exactly the properties that exist."""
    assert "additionalProperties" not in RootCause.model_json_schema()


def test_a_failing_evidence_policy_does_not_destroy_the_observation() -> None:
    """Fail open here, closed on authority. A filter that raises has not found anything
    to hide, and turning a good result into an error costs the role a real observation
    over a bug in an adapter's redaction."""

    class Broken(NullEvidencePolicy):
        def filter_result(self, content: str) -> str:
            raise RuntimeError("bad pattern")

    assert redact(Broken(), "pod/web is CrashLoopBackOff") == "pod/web is CrashLoopBackOff"
