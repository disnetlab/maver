"""The two projections that carry a finished session into the retrieval corpus.

`evidence_text` and `lesson_text` are at once what is indexed and what a later incident's
model reads, so their exact wording is a held-constant input to every published retrieval
measurement. The expectations are a golden file produced by importing the prototype's own
`build_memory_projections` and calling it, never by recording Kira's output: a fixture
generated from the code under test asserts nothing.
`test_the_golden_file_came_from_the_prototype` re-derives it whenever the prototype tree
is on disk, so the file cannot quietly drift into being our own output later.
"""

import importlib.util
import json
import os
import pathlib
import sys
import types
from collections.abc import Callable
from typing import Any

import pytest

from kira.contracts import LearningResult
from kira.memory.projections import build_projections

FIXTURE = pathlib.Path(__file__).resolve().parents[1] / "fixtures" / "prototype_projections.json"
GOLDEN: dict[str, dict[str, Any]] = json.loads(FIXTURE.read_text(encoding="utf-8"))["expected"]

_DEFAULT_PROTOTYPE = pathlib.Path(__file__).resolve().parents[2].parent / "k8s-agentic-operator"
PROTOTYPE = (
    pathlib.Path(os.environ.get("KIRA_PROTOTYPE_ROOT", _DEFAULT_PROTOTYPE))
    / "agent"
    / "memory"
    / "projections.py"
)


def load_prototype_renderer() -> tuple[Callable[..., Any], Callable[[bool], None]]:
    """Load the prototype's renderer, stubbing only the two names it imports.

    Returns its `build_memory_projections` and a setter for the masking flag it reads
    from `agent.config`. Nothing else of the prototype is involved, so the rendering
    code that runs is the prototype's own.
    """
    mask = {"on": False}
    package = types.ModuleType("agent")
    package.__path__ = []  # type: ignore[attr-defined]
    config = types.ModuleType("agent.config")
    config.memory_mask_identifiers = lambda: mask["on"]  # type: ignore[attr-defined]
    state = types.ModuleType("agent.state")
    state.LearningResult = LearningResult  # type: ignore[attr-defined]
    sys.modules.update({"agent": package, "agent.config": config, "agent.state": state})
    spec = importlib.util.spec_from_file_location("_prototype_projections", PROTOTYPE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_memory_projections, lambda on: mask.__setitem__("on", on)


def result(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "observations": ["model-written observation, visible only when none are supplied"],
        "attempted_hypothesis": "The checkout readiness probe pointed at port 8081.",
        "attempted_action": "Patched the readiness probe to port 8080.",
        "observed_outcome": "Pods became ready and the 5xx rate fell back to baseline.",
        "case_negative_constraints": [],
        "corrected_cause": "",
        "discriminator": "",
        "summary": "Checkout was unreachable for eleven minutes.",
        "fault_category": "probe_misconfig",
    }
    base.update(over)
    return base


ALERT = {"alertname": "TargetDown", "namespace": "social-network", "service": "user-service"}
OBSERVATIONS = ["  0/3 pods ready in social-network  ", "", "probe :8081 connection refused"]
MODEL_CAUSE = "the model's own account of what the cause turned out to be"
GROUND_TRUTH = "[fault_spec] component=user-service; the service port was renamed"
CONSTRAINTS = ["Do not restart the deployment before reading the probe spec."]

CASES: list[dict[str, Any]] = [
    {
        "name": "no_feedback_with_discriminator",
        "result": result(
            discriminator="Ready=False with the container process healthy points at the probe.",
            corrected_cause=MODEL_CAUSE,
            case_negative_constraints=CONSTRAINTS,
        ),
        "alert": ALERT,
        "external_feedback": None,
        "evidence_observations": OBSERVATIONS,
    },
    {
        # No discriminator, so the prototype emits neither marker and the lesson reads as
        # a bare conclusion. Pinned as-is; 07-MEMORY.md §7.3 (P-9) proposes changing it.
        "name": "no_feedback_without_discriminator",
        "result": result(corrected_cause=MODEL_CAUSE),
        "alert": ALERT,
        "external_feedback": None,
        "evidence_observations": OBSERVATIONS,
    },
    {
        "name": "empty_feedback_dict_is_no_feedback",
        "result": result(corrected_cause=MODEL_CAUSE),
        "alert": ALERT,
        "external_feedback": {},
        "evidence_observations": OBSERVATIONS,
    },
    {
        "name": "scores_rejected_diagnosis_with_mitigation_success",
        "result": result(corrected_cause=MODEL_CAUSE, discriminator="Probe port, not image tag."),
        "alert": ALERT,
        "external_feedback": {"diagnosis": {"success": False}, "mitigation": {"success": True}},
        "evidence_observations": OBSERVATIONS,
    },
    {
        "name": "details_rejected_diagnosis",
        "result": result(corrected_cause=MODEL_CAUSE, case_negative_constraints=CONSTRAINTS),
        "alert": ALERT,
        "external_feedback": {
            "diagnosis": {"success": False},
            "mitigation": {"success": False},
            "ground_truth_root_cause": GROUND_TRUTH,
        },
        "evidence_observations": OBSERVATIONS,
    },
    {
        "name": "confirmed_diagnosis_and_mitigation",
        "result": result(corrected_cause=MODEL_CAUSE),
        "alert": ALERT,
        "external_feedback": {"diagnosis": {"success": True}, "mitigation": {"success": True}},
        "evidence_observations": OBSERVATIONS,
    },
    {
        "name": "details_but_diagnosis_confirmed",
        "result": result(corrected_cause=MODEL_CAUSE),
        "alert": ALERT,
        "external_feedback": {
            "diagnosis": {"success": True},
            "ground_truth_root_cause": GROUND_TRUTH,
        },
        "evidence_observations": OBSERVATIONS,
    },
    {
        "name": "explicit_empty_observations",
        "result": result(),
        "alert": ALERT,
        "external_feedback": None,
        "evidence_observations": [],
    },
    {
        "name": "no_identifiers_and_model_observations",
        "result": result(),
        "alert": {},
        "external_feedback": None,
        "evidence_observations": None,
    },
    {
        "name": "masked_identifiers",
        "result": result(discriminator="user-service in social-network stayed Ready=False."),
        "alert": ALERT,
        "external_feedback": None,
        "evidence_observations": OBSERVATIONS,
        "mask_identifiers": True,
    },
]

BY_NAME = {case["name"]: case for case in CASES}


def render(case: dict[str, Any]) -> dict[str, Any]:
    got = build_projections(
        LearningResult(**case["result"]),
        alert=case["alert"],
        external_feedback=case["external_feedback"],
        evidence_observations=case["evidence_observations"],
        mask_identifiers=bool(case.get("mask_identifiers")),
    )
    return {
        "evidence_text": got.evidence_text,
        "lesson_text": got.lesson_text,
        "corrected_cause": got.corrected_cause,
        "feedback_depth": got.feedback_depth,
        "authoritative_correction": got.authoritative_correction,
    }


MARKER = "INCIDENT-BOUND (this incident only — not an answer for yours):"


@pytest.mark.parametrize("name", sorted(BY_NAME))
def test_projections_are_byte_identical_except_the_marker(name: str) -> None:
    """Every label, separator and template is corpus text and model-facing text at once.

    The one deliberate difference is the INCIDENT-BOUND marker, which Kira emits on
    every lesson. Removing it from Kira's output must restore the prototype's exactly:
    the change is one added line, not a reworded lesson.
    """
    got = render(BY_NAME[name])
    want = dict(GOLDEN[name])

    lines = got["lesson_text"].split("\n")
    assert MARKER in lines, "the marker must be present on every lesson"
    if MARKER not in want["lesson_text"]:
        got = dict(got)
        got["lesson_text"] = "\n".join(line for line in lines if line != MARKER)
    assert got == want


@pytest.mark.parametrize("name", sorted(BY_NAME))
def test_the_incident_bound_marker_is_always_present(name: str) -> None:
    """A lesson without it reads as a bare conclusion about incidents in general.

    The prototype emitted it only alongside a discriminator, which the schema does not
    require and which defaults to empty — so the marker was missing from seven of these
    ten cases, and missing precisely from the records most likely to be over-generalised.
    """
    assert MARKER in render(BY_NAME[name])["lesson_text"]


@pytest.mark.skipif(not PROTOTYPE.exists(), reason="the prototype tree is not on disk")
def test_the_golden_file_came_from_the_prototype() -> None:
    """Re-derive the golden file from the prototype so it cannot become our own output."""
    renderer, set_mask = load_prototype_renderer()
    for case in CASES:
        set_mask(bool(case.get("mask_identifiers")))
        got = renderer(
            LearningResult(**case["result"]),
            alert=case["alert"],
            external_feedback=case["external_feedback"],
            evidence_observations=case["evidence_observations"],
        )
        assert got.evidence_text == GOLDEN[case["name"]]["evidence_text"], case["name"]
        assert got.lesson_text == GOLDEN[case["name"]]["lesson_text"], case["name"]
        assert got.corrected_cause == GOLDEN[case["name"]]["corrected_cause"], case["name"]


@pytest.mark.parametrize(
    ("feedback", "depth", "cause", "authoritative"),
    [
        (None, "none", MODEL_CAUSE, False),
        ({}, "none", MODEL_CAUSE, False),
        ({"diagnosis": {"success": False}}, "scores", "", False),
        ({"diagnosis": {"success": True}}, "scores", "", False),
        ({"mitigation": {"success": True}}, "scores", "", False),
        (
            {"diagnosis": {"success": False}, "ground_truth_root_cause": GROUND_TRUTH},
            "details",
            GROUND_TRUTH,
            True,
        ),
        (
            {"diagnosis": {"success": True}, "ground_truth_root_cause": GROUND_TRUTH},
            "details",
            "",
            False,
        ),
        ({"ground_truth_root_cause": GROUND_TRUTH}, "details", MODEL_CAUSE, False),
    ],
)
def test_external_feedback_decides_which_correction_is_authoritative(
    feedback: dict[str, Any] | None, depth: str, cause: str, authoritative: bool
) -> None:
    """Scores can reject a hypothesis but can never manufacture its correction.

    A correction is the oracle's only when the oracle rejected the diagnosis; a confirmed
    diagnosis has nothing to correct; otherwise the model's own account stands, unlabelled.
    """
    got = build_projections(
        LearningResult(**result(corrected_cause=MODEL_CAUSE)),
        alert=ALERT,
        external_feedback=feedback,
    )
    assert (got.feedback_depth, got.corrected_cause, got.authoritative_correction) == (
        depth,
        cause,
        authoritative,
    )


@pytest.mark.parametrize(
    "name",
    [
        "no_feedback_with_discriminator",
        "details_rejected_diagnosis",
        "scores_rejected_diagnosis_with_mitigation_success",
    ],
)
def test_the_corrected_cause_never_reaches_the_lesson_text(name: str) -> None:
    """It is the most copyable sentence in a search result and it is about another incident.

    A memory branch once submitted one verbatim as its own diagnosis for an unrelated
    fault, so the affordance is removed rather than warned against. Keeping it out of
    `lesson_text` also keeps it out of the lesson embedding.
    """
    case = BY_NAME[name]
    got = render(case)
    assert MODEL_CAUSE not in got["lesson_text"]
    assert GROUND_TRUTH not in got["lesson_text"]
    assert MODEL_CAUSE not in got["evidence_text"]
    assert GROUND_TRUTH not in got["evidence_text"]


def test_evidence_text_uses_the_observations_the_caller_supplies() -> None:
    """Post-feedback learning can copy the oracle's correction into its own observations.

    The caller passes the pre-feedback RCA evidence instead. An explicitly empty list
    renders identifiers only rather than falling back to that contaminated model text;
    only `None` falls back, which is a convenience for rendering a record standalone.
    """
    args: dict[str, Any] = {"alert": ALERT, "external_feedback": None}
    supplied = build_projections(
        LearningResult(**result()), evidence_observations=["only this line"], **args
    )
    assert supplied.evidence_text.endswith("\n- only this line")
    assert "model-written observation" not in supplied.evidence_text

    empty = build_projections(LearningResult(**result()), evidence_observations=[], **args)
    assert empty.evidence_text == "Incident observations:\n" + (
        "Identifiers: alert=TargetDown; namespace=social-network; service=user-service"
    )

    fallback = build_projections(LearningResult(**result()), evidence_observations=None, **args)
    assert "model-written observation" in fallback.evidence_text
