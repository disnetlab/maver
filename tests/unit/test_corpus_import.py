"""Loading an archived corpus, and what a bad file may not be allowed to leave behind.

The importer exists so that a campaign whose records were written by an earlier system
can be retrieved by this one. That only holds if an imported record is indistinguishable
from a distilled one at the point retrieval reads it, which is what the round-trip below
pins: the same retriever, the same query path, the same normalised index.
"""

import json
import pathlib
from typing import Any

import pytest
from typer.testing import CliRunner

from kira.cli import app
from kira.memory.corpus import ImportReport, MalformedRecord, import_records, parse_records
from kira.memory.factory import build_storage
from kira.settings.groups import MemorySettings, ServerSettings
from kira.settings.root import Settings

runner = CliRunner()

RECORD: dict[str, Any] = {
    "session_id": "archived-1",
    "evidence_text": "the Service targets a port no container listens on; endpoints empty",
    "lesson_text": "repoint the Service at the port the container serves",
    "incident_summary": "a target port misconfiguration on the user Service",
    "lifecycle_state": "tentative_success",
    "fault_category": "misconfiguration",
    "alert_name": "ServiceHealthDegraded",
    "namespace": "shop",
    "affected_resource": "shop/Service/user",
    "verification_verdict": "recovered",
    "evaluation_label": "graded",
    "evaluation_source": "harness",
    "evaluation_diagnosis_success": True,
    "evaluation_diagnosis_score": 0.75,
    "evaluation_mitigation_success": False,
    "corrected_cause": "the Service selector was right and the target port was not",
    "labels": {"external_key": "k8s_target_port-misconfig"},
}

OTHER: dict[str, Any] = {
    "session_id": "archived-2",
    "evidence_text": "the deployment references an image tag that does not exist",
    "lesson_text": "roll the deployment back to the tag that was serving",
    "incident_summary": "a bad image tag",
    "lifecycle_state": "invalidated",
    "fault_category": "deployment",
    "alert_name": "PodCrashLooping",
    "namespace": "shop",
    "affected_resource": "shop/Deployment/cart",
}


def _file(tmp_path: pathlib.Path, *records: dict[str, Any]) -> pathlib.Path:
    path = tmp_path / "corpus.jsonl"
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    return path


def _storage(tmp_path: pathlib.Path, corpus: str = "default") -> Any:
    return build_storage(
        Settings(server=ServerSettings(workdir=tmp_path), memory=MemorySettings(corpus_id=corpus))
    )


def test_an_imported_record_is_retrieved_by_the_normal_path(tmp_path: pathlib.Path) -> None:
    """The point of going through the port: the backend's own index is filled.

    A copy that wrote the rows and not the normalised terms would store a record that
    ``open_record`` returns and no query ever reaches, which is the failure this asserts
    against — the search below is the retriever every role uses, not a scan.
    """
    storage = _storage(tmp_path)
    report = import_records(
        storage.index, parse_records(_file(tmp_path, RECORD, OTHER).read_text())
    )

    assert report == ImportReport(imported=2, replaced=0, skipped=0)
    hits = storage.retriever.search("Service targets a port no container listens on")
    assert [hit.session_id for hit in hits] == ["archived-1"]
    assert hits[0].metadata["state"] == "tentative_success"

    stored = storage.index.open_record("archived-1")
    assert stored is not None
    # Every field the file carried, including the ones retrieval withholds from a hit.
    assert stored.corrected_cause == RECORD["corrected_cause"]
    assert stored.evaluation_diagnosis_score == 0.75
    assert stored.evaluation_mitigation_success is False
    assert stored.labels == {"external_key": "k8s_target_port-misconfig"}
    storage.close()


def test_an_import_is_keyed_by_session_and_replaces_only_when_told_to(
    tmp_path: pathlib.Path,
) -> None:
    """Re-running an interrupted import must not revise what a run has already read."""
    storage = _storage(tmp_path)
    import_records(storage.index, parse_records(_file(tmp_path, RECORD).read_text()))

    revised = {**RECORD, "lesson_text": "a lesson somebody rewrote afterwards"}
    again = parse_records(_file(tmp_path, revised).read_text())

    assert import_records(storage.index, again) == ImportReport(imported=0, replaced=0, skipped=1)
    kept = storage.index.open_record("archived-1")
    assert kept is not None
    assert kept.lesson_text == RECORD["lesson_text"]

    assert import_records(storage.index, again, replace=True) == ImportReport(
        imported=0, replaced=1, skipped=0
    )
    written = storage.index.open_record("archived-1")
    assert written is not None
    assert written.lesson_text == "a lesson somebody rewrote afterwards"
    # Replaced, not added beside: FTS5 has no upsert and a session indexed twice would
    # be retrieved twice.
    assert storage.index.count() == 1
    storage.close()


def test_a_state_this_build_cannot_name_is_stored_and_reported(tmp_path: pathlib.Path) -> None:
    """Ranking demotes by state, so an export's typo silently re-weights the corpus."""
    storage = _storage(tmp_path)
    odd = {**RECORD, "lifecycle_state": "probably_fine"}

    report = import_records(storage.index, parse_records(_file(tmp_path, odd).read_text()))

    assert report.unknown_states == ("probably_fine",)
    stored = storage.index.open_record("archived-1")
    assert stored is not None
    assert stored.lifecycle_state == "probably_fine"
    storage.close()


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ('{"session_id": "x", "evidence_text": "a", "fault": "typo"}', "unknown field(s) fault"),
        ('{"evidence_text": "a"}', "session_id is required"),
        ('{"session_id": "x"}', "neither evidence_text nor lesson_text"),
        ('{"session_id": "x", "evidence_text": 7}', "evidence_text must be a string"),
        (
            '{"session_id": "x", "evidence_text": "a", "evaluation_diagnosis_score": true}',
            "evaluation_diagnosis_score must be a number or null",
        ),
        (
            '{"session_id": "x", "evidence_text": "a", "evaluation_diagnosis_success": "yes"}',
            "evaluation_diagnosis_success must be true, false or null",
        ),
        ('{"session_id": "x", "evidence_text": "a", "labels": ["k"]}', "labels must be an object"),
        (
            '{"session_id": "x", "evidence_text": "a", "labels": {"k": 2}}',
            "labels.k must be a string",
        ),
        ('["not an object"]', "expected one JSON object per line"),
        ("{oops", "not JSON"),
    ],
)
def test_a_line_that_cannot_be_read_is_refused_by_number(line: str, expected: str) -> None:
    """Every refusal names its line: "some line is wrong" is not actionable over 123."""
    with pytest.raises(MalformedRecord) as refused:
        parse_records(f"{json.dumps(RECORD)}\n\n{line}\n")
    assert str(refused.value).startswith("line 3: ")
    assert expected in str(refused.value)


def test_one_session_may_not_arrive_twice_in_one_file() -> None:
    """Two rows for one session make the result depend on which one was written last."""
    with pytest.raises(MalformedRecord) as refused:
        parse_records(f"{json.dumps(RECORD)}\n{json.dumps(RECORD)}\n")
    assert "line 2: session 'archived-1' was already given on line 1" in str(refused.value)


def test_the_command_loads_a_named_corpus_and_reports_what_it_did(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The corpus is the database, so ``--corpus`` decides which campaign these become."""
    monkeypatch.setenv("KIRA_SERVER__WORKDIR", str(tmp_path))
    path = _file(tmp_path, RECORD, OTHER)

    first = runner.invoke(app, ["memory", "import", str(path), "--corpus", "archived_sweep"])
    assert first.exit_code == 0, first.output
    assert "archived_sweep: imported 2, replaced 0, skipped 0 already present" in first.output

    again = runner.invoke(app, ["memory", "import", str(path), "--corpus", "archived_sweep"])
    assert "imported 0, replaced 0, skipped 2 already present" in again.output

    replaced = runner.invoke(
        app, ["memory", "import", str(path), "--corpus", "archived_sweep", "--replace"]
    )
    assert "imported 0, replaced 2, skipped 0 already present" in replaced.output

    # The named corpus, and not the configured default, is what was written.
    named = _storage(tmp_path, "archived_sweep")
    assert named.index.count() == 2
    named.close()
    assert not (tmp_path / "memory" / "default.sqlite").exists()


def test_a_bad_file_leaves_no_corpus_behind(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Validated whole before the store is opened, so a refusal is not a half-import."""
    monkeypatch.setenv("KIRA_SERVER__WORKDIR", str(tmp_path))
    path = tmp_path / "corpus.jsonl"
    path.write_text(f"{json.dumps(RECORD)}\n{json.dumps({'session_id': 'x'})}\n", encoding="utf-8")

    refused = runner.invoke(app, ["memory", "import", str(path), "--corpus", "archived_sweep"])

    assert refused.exit_code == 2
    assert "line 2: session 'x' carries neither evidence_text nor lesson_text" in refused.output
    assert not (tmp_path / "memory").exists()


def test_a_missing_file_is_a_message_rather_than_a_traceback(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KIRA_SERVER__WORKDIR", str(tmp_path))
    missing = runner.invoke(app, ["memory", "import", str(tmp_path / "nothing.jsonl")])
    assert missing.exit_code == 2
    assert "nothing.jsonl" in missing.output
