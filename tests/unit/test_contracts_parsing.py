"""Tolerant decoding of model-submitted fields.

Each case here corresponds to a serializer failure that was observed to discard
substantively correct work. A regression in any of them is silent: the submission
simply fails and the stage produces nothing.
"""

import json

import pytest

from maver.contracts.parsing import (
    cites_request_path,
    decode_container,
    is_blank_reason,
    loads_tolerant,
    normalize_nullish,
)


@pytest.mark.parametrize("value", ["null", "NULL", " none ", "n/a", "nil", "", "no"])
def test_nullish_placeholders_become_none(value: str) -> None:
    assert normalize_nullish(value) is None


@pytest.mark.parametrize("value", ["the probe could not run", "0", "unknown"])
def test_real_text_survives(value: str) -> None:
    assert normalize_nullish(value) == value


def test_non_string_values_pass_through() -> None:
    assert normalize_nullish(False) is False
    assert normalize_nullish(None) is None


@pytest.mark.parametrize("value", ["null", "", "  ", None, 3, "N/A"])
def test_blank_reasons_are_rejected(value: object) -> None:
    assert is_blank_reason(value)


def test_a_real_reason_is_accepted() -> None:
    assert not is_blank_reason("the target Deployment had no such container")


@pytest.mark.parametrize(
    "check",
    [
        "curl http://frontend:8080/health returned 200",
        "probe_success{job='blackbox'} = 1",
        "p99 latency fell to 120ms",
        "error rate 0.2% over 5m",
        "trace shows the span completing",
    ],
)
def test_request_path_evidence_is_recognised(check: str) -> None:
    assert cites_request_path(check)


@pytest.mark.parametrize(
    "check",
    [
        "kubectl get pods shows 3/3 Running",
        "deployment rollout complete, 0 restarts",
        "all replicas Ready",
    ],
)
def test_control_plane_status_is_not_request_path_evidence(check: str) -> None:
    assert not cites_request_path(check)


def test_container_arriving_as_a_string_is_decoded() -> None:
    """Models routinely emit a container serialised into a string field."""
    payload = '[{"action": "patch", "target": "svc/web"}]'
    assert decode_container(payload) == [{"action": "patch", "target": "svc/web"}]


def test_list_of_stringified_objects_is_decoded_elementwise() -> None:
    assert decode_container(['{"a": 1}', '{"b": 2}']) == [{"a": 1}, {"b": 2}]


def test_undecodable_input_is_returned_untouched() -> None:
    """Pydantic must still report the real error rather than a repair artefact."""
    assert decode_container("not json at all") == "not json at all"
    assert decode_container(17) == 17


def test_object_closed_one_level_early_is_repaired() -> None:
    """The dominant malformation: the step object shuts before its trailing keys.

    The parser stops at the stray colon reporting a missing comma, and every retry
    reproduces it. Deleting the closer that shut the object too early recovers the
    step list verbatim.
    """
    malformed = '[{"action": "a", "target": "t", "params": {"x": 1}}, "justification": "j"}]'
    with pytest.raises(json.JSONDecodeError):
        json.loads(malformed)
    assert loads_tolerant(malformed) == [
        {"action": "a", "target": "t", "params": {"x": 1}, "justification": "j"}
    ]


def test_trailing_keys_past_a_complete_array_are_ignored() -> None:
    assert loads_tolerant('[{"a": 1}] , "stray": 2') == [{"a": 1}]


def test_well_formed_json_is_unchanged() -> None:
    doc = '[{"action": "restart", "target": "deploy/api", "params": {}, "justification": "j"}]'
    assert loads_tolerant(doc) == json.loads(doc)


def test_irreparable_json_still_raises() -> None:
    with pytest.raises(json.JSONDecodeError):
        loads_tolerant('{"unterminated": ')
