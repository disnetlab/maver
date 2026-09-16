"""MAVER's renamed settings preserve existing deployment configuration."""

import pytest
from apscheduler.util import ref_to_obj

from maver.settings.root import legacy_names_in_use, load_settings


def test_kira_environment_is_read_and_maver_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KIRA_PIPELINE__MAX_TURNS", "23")
    assert load_settings().pipeline.max_turns == 23
    monkeypatch.setenv("MAVER_PIPELINE__MAX_TURNS", "31")
    assert load_settings().pipeline.max_turns == 31
    assert legacy_names_in_use()["KIRA_PIPELINE__MAX_TURNS"] == "MAVER_PIPELINE__MAX_TURNS"


def test_adapter_settings_survive_the_prefix_change(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KIRA_ADAPTERS__SREGYM__MCP_BASE_URL", "http://old:9954")
    assert load_settings().adapters["sregym"]["mcp_base_url"] == "http://old:9954"
    monkeypatch.setenv("MAVER_ADAPTERS__SREGYM__MCP_BASE_URL", "http://new:9954")
    assert load_settings().adapters["sregym"]["mcp_base_url"] == "http://new:9954"


def test_a_misspelled_legacy_setting_is_not_silently_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KIRA_PIPELINE__MAX_TURNSS", "23")
    with pytest.raises(ValueError, match="KIRA_PIPELINE__MAX_TURNSS"):
        load_settings()


@pytest.mark.parametrize(
    "module,name",
    [
        ("jobs", "expire_approval"),
        ("jobs", "resume_approval"),
        ("jobs", "sweep_decided_approvals"),
        ("learning", "distil_incident"),
    ],
)
def test_persisted_scheduler_references_resolve_to_the_maver_runtime(
    module: str, name: str
) -> None:
    old = ref_to_obj(f"kira.orchestration.{module}:{name}")
    current = ref_to_obj(f"maver.orchestration.{module}:{name}")
    assert old is current
