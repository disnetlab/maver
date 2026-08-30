"""Prompt asset loading and strict composition."""

from pathlib import Path

import pytest

from kira.prompts.loader import (
    PromptError,
    compose_role_prompt,
    markers,
    read_asset,
    read_override,
    render,
)

SHIPPED_ROLES = ["rca", "planning", "execution"]


@pytest.mark.parametrize("role", SHIPPED_ROLES)
@pytest.mark.parametrize("memory_enabled", [True, False])
def test_every_shipped_role_composes_in_both_memory_states(role: str, memory_enabled: bool) -> None:
    """A role whose blocks are missing would only fail at incident time."""
    prompt = compose_role_prompt(role, memory_enabled=memory_enabled)
    assert prompt.strip()
    assert "{{" not in prompt


def test_memory_guidance_appears_only_when_memory_is_on() -> None:
    on = compose_role_prompt("rca", memory_enabled=True)
    off = compose_role_prompt("rca", memory_enabled=False)
    assert "memory_search" in on
    assert "memory_search" not in off


def test_an_empty_block_is_valid_and_leaves_no_gap() -> None:
    """The memory-off RCA block is empty upstream and is carried over as such.

    Inserting guidance here would change the configuration the published baseline
    was measured on.
    """
    assert read_asset("blocks/memory/rca.disabled.txt") == ""
    prompt = compose_role_prompt("rca", memory_enabled=False)
    assert "\n\n\n" not in prompt


def test_execution_declares_no_memory_marker() -> None:
    assert markers(read_asset("roles/execution.txt")) == {"KUBECTL_GUIDANCE"}


def test_roles_receive_the_kubectl_guidance_matching_their_authority() -> None:
    assert "Write verbs are rejected" in compose_role_prompt("planning", memory_enabled=False)
    assert "for every cluster operation" in compose_role_prompt("execution", memory_enabled=False)


class TestRender:
    def test_substitutes_declared_markers(self) -> None:
        assert render("a {{X}} b", {"X": "value"}) == "a value b\n"

    def test_a_marker_with_no_value_raises(self) -> None:
        with pytest.raises(PromptError, match="no value supplied"):
            render("a {{X}} b", {})

    def test_a_value_with_no_marker_raises(self) -> None:
        """Renaming a marker must not silently leave a prompt half-wired."""
        with pytest.raises(PromptError, match="absent prompt marker"):
            render("a b", {"X": "value"})

    def test_blank_runs_left_by_an_empty_block_are_collapsed(self) -> None:
        assert render("head\n\n{{X}}\n\ntail", {"X": ""}) == "head\n\ntail\n"


class TestAssets:
    def test_a_missing_asset_raises(self) -> None:
        with pytest.raises(PromptError, match="not found"):
            read_asset("roles/does_not_exist.txt")

    def test_an_override_is_read_from_outside_the_package(self, tmp_path: Path) -> None:
        target = tmp_path / "rca.txt"
        target.write_text("custom {{KUBECTL_GUIDANCE}}", encoding="utf-8")
        assert read_override(target).startswith("custom")

    def test_an_override_cannot_escape_its_root(self, tmp_path: Path) -> None:
        outside = tmp_path.parent / "elsewhere.txt"
        outside.write_text("x", encoding="utf-8")
        with pytest.raises(PromptError, match="escapes"):
            read_override(outside, root=tmp_path)

    def test_a_missing_override_raises(self, tmp_path: Path) -> None:
        with pytest.raises(PromptError, match="cannot read"):
            read_override(tmp_path / "absent.txt")


def test_an_override_replaces_the_template_but_still_composes(tmp_path: Path) -> None:
    """Configuration is the supported way to change a prompt, not editing the package."""
    override = "## Goal\n\nDo it my way.\n\n{{KUBECTL_GUIDANCE}}\n"
    prompt = compose_role_prompt("rca", memory_enabled=False, override=override)
    assert "Do it my way." in prompt
    assert "exec_kubectl_cmd_safely" in prompt


def test_a_role_without_cluster_tools_may_not_declare_the_kubectl_marker() -> None:
    with pytest.raises(PromptError, match="holds no cluster tools"):
        compose_role_prompt("learning", memory_enabled=False, override="{{KUBECTL_GUIDANCE}}")
