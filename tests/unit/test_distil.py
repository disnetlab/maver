"""Writing one incident's lesson into the corpus."""

import pathlib
from typing import Any

import pytest

from kira.contracts.artifacts import LearningPacket, LearningResult, RoleTraceSummary
from kira.memory.distil import distil
from kira.memory.factory import build_storage
from kira.settings.groups import MemorySettings, ServerSettings
from kira.settings.root import Settings

RESULT = LearningResult(
    observations=["svc/user had no endpoints"],
    attempted_hypothesis="the service targeted a port nothing listens on",
    attempted_action="patched targetPort to 8080",
    observed_outcome="endpoints appeared and the 5xx rate fell",
    case_negative_constraints=["restarting the deployment does not help"],
    corrected_cause="",
    discriminator="check svc endpoints before restarting anything",
    summary="a Service targeting a port no container listens on",
    fault_category="misconfiguration",
    verification_verdict="recovered",
)


def packet(**overrides: Any) -> LearningPacket:
    fields: dict[str, Any] = {
        "session_id": "s1",
        "lifecycle_state": "tentative_success",
        "alert": {"alertname": "HighErrorRate", "namespace": "shop"},
        "root_cause": {"affected_resource": "shop/Service/user"},
        "trace_summaries": [
            RoleTraceSummary(
                role="rca",
                goal="diagnose",
                key_observations=["endpoints empty for svc/user"],
            ),
            RoleTraceSummary(role="planning", goal="plan", key_observations=["patch it"]),
        ],
    }
    fields.update(overrides)
    return LearningPacket(**fields)


def storage(tmp_path: pathlib.Path, corpus: str = "default") -> Any:
    return build_storage(
        Settings(
            server=ServerSettings(workdir=tmp_path),
            memory=MemorySettings(corpus_id=corpus),
        )
    )


def test_a_distilled_session_becomes_retrievable_and_is_marked(
    tmp_path: pathlib.Path,
) -> None:
    stack = storage(tmp_path)
    session = stack.sessions.create_session({"alertname": "HighErrorRate"})

    memory = distil(
        packet(session_id=session.session_id),
        RESULT,
        index=stack.index,
        sessions=stack.sessions,
    )

    assert "endpoints" in memory.evidence_text
    assert "restarting the deployment does not help" in memory.lesson_text
    assert stack.index.open_record(session.session_id) is not None
    stored = stack.sessions.get_session(session.session_id)
    assert stored is not None and stored.distilled is True
    stack.close()


def test_re_distilling_replaces_rather_than_accumulates(tmp_path: pathlib.Path) -> None:
    """A corpus with one record per attempt would let a session distilled twice outvote
    one distilled once, purely by having been retried."""
    stack = storage(tmp_path)
    stack.sessions.create_session({})

    distil(packet(), RESULT, index=stack.index, sessions=stack.sessions)
    distil(packet(), RESULT, index=stack.index, sessions=stack.sessions)

    assert stack.index.count() == 1
    stack.close()


def test_the_evidence_index_uses_what_was_seen_before_any_grade_arrived(
    tmp_path: pathlib.Path,
) -> None:
    """An external grade is authoritative about the answer, not about what the run
    observed, and indexing post-grade text would let a record be found by a conclusion
    it did not reach on its own."""
    stack = storage(tmp_path)
    graded = packet(external_feedback={"diagnosis": {"success": False}, "source": "oracle"})

    memory = distil(graded, RESULT, index=stack.index, sessions=stack.sessions)

    assert "endpoints empty for svc/user" in memory.evidence_text
    assert memory.evaluation_label == "unknown"
    assert memory.evaluation_source == "oracle"
    stack.close()


@pytest.mark.parametrize(
    ("feedback", "label"),
    [
        ({"diagnosis": {"success": True}, "mitigation": {"success": True}}, "success"),
        ({"diagnosis": {"success": True}, "mitigation": {"success": False}}, "partial"),
        ({"diagnosis": {"success": False}, "mitigation": {"success": False}}, "failure"),
        (
            {"diagnosis": {"success": False, "score": 0.4}, "mitigation": {"success": False}},
            "partial",
        ),
        ({"diagnosis": {"success": True}}, "unknown"),
    ],
)
def test_partial_credit_is_not_flattened_into_a_verdict(
    tmp_path: pathlib.Path, feedback: dict[str, Any], label: str
) -> None:
    stack = storage(tmp_path)

    memory = distil(
        packet(external_feedback=feedback), RESULT, index=stack.index, sessions=stack.sessions
    )

    assert memory.evaluation_label == label
    stack.close()


def test_two_campaigns_cannot_read_each_other_s_records(tmp_path: pathlib.Path) -> None:
    """The partition is the database, so a run configured for one corpus cannot reach
    another's even by mistake."""
    first = storage(tmp_path, "sweep-a")
    distil(packet(), RESULT, index=first.index, sessions=first.sessions)
    first.close()

    second = storage(tmp_path, "sweep-b")
    assert second.index.count() == 0
    second.close()


def test_a_corpus_name_that_could_escape_the_workspace_is_refused(
    tmp_path: pathlib.Path,
) -> None:
    """A corpus name reaches the filesystem. Rewriting it silently would put a
    campaign's records somewhere nobody looked for them."""
    with pytest.raises(ValueError, match="not a usable name"):
        storage(tmp_path, "../elsewhere")
