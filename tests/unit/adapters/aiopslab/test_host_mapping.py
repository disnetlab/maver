"""The encoders decide what AIOpsLab grades, so every one of them is pinned here.

Migrated from the prototype's `tests/test_aiopslab_adapter.py`, whose expectations came
out of watching a small local model answer real problems. Two of them are the reason the
adapter exists at all: a halted-but-healthy diagnosis is a `"No"` rather than a crash,
and a missing diagnosis is a crash rather than a guessed `"Yes"`.
"""

import json

import pytest

from maver.adapters.aiopslab.host.mapping import (
    _analysis_from_root_cause,
    _is_terminal_record,
    build_alert,
    submission_action,
)

pytestmark = pytest.mark.parity


def test_alert_carries_the_task_contract_and_no_problem_id() -> None:
    alert = build_alert(
        "localization",
        "test-hotel-reservation",
        "Hotel app",
        observation_window_start="2026-08-26T16:23:20Z",
    )

    assert alert["benchmark"] == "aiopslab"
    assert alert["benchmark_task"] == "localization"
    assert alert["namespace"] == "test-hotel-reservation"
    assert alert["description"] == "Hotel app"
    assert alert["observation_window_start"] == "2026-08-26T16:23:20Z"
    assert "problem" not in json.dumps(alert).lower()


def test_alert_leaves_the_connectivity_contract_to_the_agent_side() -> None:
    """The credential-holding half no longer carries Verify's entry-point table.

    The prototype attached it here, keyed by namespace. In MAVER it is adapter
    configuration on the agent side, so nothing under `host/` needs to know which
    Service a benchmark application is fronted by.
    """
    alert = build_alert("mitigation", "test-social-network", "Social app")

    assert not [key for key in alert if key.startswith("_")]


def test_an_unknown_task_type_is_refused() -> None:
    with pytest.raises(ValueError, match="unsupported AIOpsLab task type"):
        build_alert("forecasting", "test-social-network", "Social app")


def test_detection_maps_a_halted_healthy_diagnosis_to_no() -> None:
    record = {
        "root_cause": {
            "halt": True,
            "fault_type": "none",
            "affected_resource": "none",
            "summary": "No anomaly detected; the service is healthy.",
        }
    }

    assert submission_action("detection", record) == '```\nsubmit("No")\n```'


def test_detection_accepts_explicit_none_even_when_the_model_omits_halt() -> None:
    record = {
        "root_cause": {
            "halt": False,
            "fault_type": "none",
            "affected_resource": "none",
            "summary": "AIOpsLab detection: has_anomaly=No. Current state is healthy.",
        }
    }

    assert submission_action("detection", record) == '```\nsubmit("No")\n```'


def test_detection_maps_a_live_diagnosis_to_yes() -> None:
    record = {
        "root_cause": {
            "halt": False,
            "fault_type": "Misconfiguration",
            "affected_resource": "service/frontend",
        }
    }

    assert submission_action("detection", record) == '```\nsubmit("Yes")\n```'


def test_detection_refuses_to_fabricate_an_answer_when_the_run_produced_none() -> None:
    """Guessing here would score about half the detection tasks correct for free."""
    with pytest.raises(RuntimeError, match="without a RootCause"):
        submission_action("detection", {"error": "RCA failed: max_turns_exceeded"})


def test_localization_extracts_the_stable_component() -> None:
    record = {"root_cause": {"affected_resource": "test-social-network/deployment/user-service"}}

    assert submission_action("localization", record) == '```\nsubmit(["user-service"])\n```'


def test_localization_strips_a_trailing_namespace_phrase() -> None:
    record = {
        "root_cause": {"affected_resource": "deployment/user-service in namespace test-social"}
    }

    assert submission_action("localization", record) == '```\nsubmit(["user-service"])\n```'


def test_localization_strips_a_generated_pod_suffix() -> None:
    record = {"root_cause": {"affected_resource": "pod/user-service-7f6dbf8f7b-abcd1"}}

    assert submission_action("localization", record) == '```\nsubmit(["user-service"])\n```'


def test_localization_submits_an_empty_list_for_a_no_anomaly_artifact() -> None:
    record = {
        "root_cause": {
            "halt": True,
            "fault_type": "none",
            "affected_resource": "none",
            "summary": "No fault; the namespace is healthy.",
        }
    }

    assert submission_action("localization", record) == "```\nsubmit([])\n```"


def test_analysis_honours_an_explicit_classification() -> None:
    answer = _analysis_from_root_cause(
        {
            "summary": (
                "AIOpsLab classification: system_level=Virtualization; "
                "fault_type=Operation Error. Replicas were scaled to zero."
            )
        }
    )

    assert answer["system_level"] == "Virtualization"
    assert answer["fault_type"] == "Operation Error"
    assert answer["source"] == "model"


def test_analysis_normalizes_scale_to_zero_as_a_virtualization_operation() -> None:
    answer = _analysis_from_root_cause(
        {
            "fault_type": "misconfiguration",
            "affected_resource": "deployment/user-service",
            "summary": (
                "AIOpsLab classification: system_level=Application; "
                "fault_type=Misconfiguration. The deployment replicas were set to 0."
            ),
        }
    )

    assert answer == {
        "system_level": "Virtualization",
        "fault_type": "Operation Error",
        "source": "normalized",
        "rule": "operator_mutation",
    }


def test_analysis_normalizes_mongodb_auth_as_an_application_authentication_fault() -> None:
    answer = _analysis_from_root_cause(
        {
            "fault_type": "Authentication Issue",
            "affected_resource": "service/geo",
            "summary": (
                "AIOpsLab classification: system_level=Virtualization; "
                "fault_type=Authentication Issue. The MongoDB admin user's "
                "privileges were revoked."
            ),
        }
    )

    assert answer == {
        "system_level": "Application",
        "fault_type": "Authentication Issue",
        "source": "normalized",
        "rule": "database_auth",
    }


def test_analysis_falls_back_to_the_keyword_ladder_and_says_so() -> None:
    answer = _analysis_from_root_cause({"summary": "the container image tag is wrong"})

    assert answer["source"] == "fallback"
    assert answer["rule"] == "keyword_ladder"


def test_analysis_submits_only_the_two_graded_keys() -> None:
    """The provenance is for the record we keep, not for the answer AIOpsLab parses."""
    record = {"root_cause": {"summary": "the container image tag is wrong"}}

    submitted = submission_action("analysis", record)

    assert json.loads(submitted.split("submit(", 1)[1].rsplit(")", 1)[0]) == {
        "system_level": "Virtualization",
        "fault_type": "Misconfiguration",
    }


def test_mitigation_is_a_parameterless_submit() -> None:
    assert submission_action("mitigation", {}) == "```\nsubmit()\n```"


def test_an_unknown_task_type_has_no_encoder() -> None:
    with pytest.raises(ValueError, match="unsupported AIOpsLab task type"):
        submission_action("forecasting", {})


def test_a_run_status_terminal_record_ends_the_poll() -> None:
    for status in ("completed", "failed", "cancelled"):
        assert _is_terminal_record({"run_status": status, "status": "pending"})


def test_a_terminal_lifecycle_ends_the_poll_even_when_the_run_status_is_stale() -> None:
    """The crash path writes the lifecycle first, so a process that dies between the
    two writes leaves a session that is `running` forever."""
    assert _is_terminal_record({"run_status": "running", "status": "invalidated"})
    assert _is_terminal_record({"run_status": "running", "status": "halted"})


def test_a_run_parked_on_a_person_keeps_polling() -> None:
    assert not _is_terminal_record(
        {"run_status": "awaiting_human", "status": "pending", "awaiting_request_id": "r-1"}
    )
    assert not _is_terminal_record({"run_status": "running", "status": "tentative_success"})
