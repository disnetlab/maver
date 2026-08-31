"""Prompt asset loading and strict composition."""

import pathlib
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
@pytest.mark.parametrize("retrieval_enabled", [True, False])
def test_every_shipped_role_composes_in_both_memory_states(
    role: str, retrieval_enabled: bool
) -> None:
    """A role whose blocks are missing would only fail at incident time."""
    prompt = compose_role_prompt(role, retrieval_enabled=retrieval_enabled)
    assert prompt.strip()
    assert "{{" not in prompt


def test_memory_guidance_appears_only_when_memory_is_on() -> None:
    on = compose_role_prompt("rca", retrieval_enabled=True)
    off = compose_role_prompt("rca", retrieval_enabled=False)
    assert "memory_search" in on
    assert "memory_search" not in off


def test_an_empty_block_is_carried_over_as_empty() -> None:
    """The memory-off RCA block is empty upstream and is carried over as such.

    Inserting guidance here would change the configuration the published baseline was
    measured on. The blank run it leaves behind is also carried over: the prototype did
    not collapse it, and tidying a held-constant input changes it.
    """
    assert read_asset("blocks/memory/rca.disabled.txt") == ""


def test_execution_declares_no_memory_marker() -> None:
    assert markers(read_asset("roles/execution.txt")) == {"KUBECTL_GUIDANCE"}


def test_roles_receive_the_kubectl_guidance_matching_their_authority() -> None:
    assert "Write verbs are rejected" in compose_role_prompt("planning", retrieval_enabled=False)
    assert "for every cluster operation" in compose_role_prompt(
        "execution", retrieval_enabled=False
    )


class TestRender:
    def test_substitutes_declared_markers(self) -> None:
        assert render("a {{X}} b", {"X": "value"}) == "a value b"

    def test_a_marker_with_no_value_raises(self) -> None:
        with pytest.raises(PromptError, match="no value supplied"):
            render("a {{X}} b", {})

    def test_a_value_with_no_marker_raises(self) -> None:
        """Renaming a marker must not silently leave a prompt half-wired."""
        with pytest.raises(PromptError, match="absent prompt marker"):
            render("a b", {"X": "value"})

    def test_substitution_is_literal_and_changes_no_whitespace(self) -> None:
        """The composed prompt is a held-constant input; tidying it changes it.

        An empty block leaves a blank run behind. The prototype left it there, so a
        prompt composed here has to leave it there too.
        """
        assert render("head\n\n{{X}}\n\ntail", {"X": ""}) == "head\n\n\n\ntail"
        assert render("x {{X}}", {"X": "  padded  "}) == "x   padded  "


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
    prompt = compose_role_prompt("rca", retrieval_enabled=False, override=override)
    assert "Do it my way." in prompt
    assert "exec_kubectl_cmd_safely" in prompt


def test_a_role_without_cluster_tools_may_not_declare_the_kubectl_marker() -> None:
    with pytest.raises(PromptError, match="holds no cluster tools"):
        compose_role_prompt("invented", retrieval_enabled=False, override="{{KUBECTL_GUIDANCE}}")


@pytest.mark.parametrize(
    ("role", "expected"),
    [
        ("rca", "read_only"),
        ("planning", "planning"),
        ("execution", "execution"),
        ("verify", "read_only"),
        ("learning", "read_only"),
        ("single", "single"),
    ],
)
def test_the_kubectl_block_map_mirrors_the_prototype(role: str, expected: str) -> None:
    """The prototype gave three roles their own guidance and everything else read-only.

    The single-agent configuration has its own text, not the execution role's. It
    mentions the rollback tool and permits one bounded rollout wait, and substituting
    the execution block would move the baseline the pipeline is measured against.
    """
    from kira.prompts.loader import _KUBECTL_BLOCK  # noqa: PLC0415

    assert _KUBECTL_BLOCK[role] == expected


COMPOSED = pathlib.Path(__file__).resolve().parents[1] / "fixtures" / "composed_prompts"


@pytest.mark.parametrize("role", SHIPPED_ROLES)
@pytest.mark.parametrize("retrieval_enabled", [True, False])
def test_composition_is_byte_identical_to_the_prototype(role: str, retrieval_enabled: bool) -> None:
    """The composed system prompt is a held-constant input, down to its whitespace.

    The fixtures are produced by replaying the prototype's own composition over its own
    files. Any difference, including a trailing newline or a collapsed blank run, is a
    change to the string the published baseline was measured on.
    """
    expected = (
        COMPOSED / f"{role}.{'memory_on' if retrieval_enabled else 'memory_off'}.txt"
    ).read_text(encoding="utf-8")
    assert compose_role_prompt(role, retrieval_enabled=retrieval_enabled) == expected
