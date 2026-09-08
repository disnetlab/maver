"""Prompt asset loading and strict composition."""

import importlib
import pathlib
from hashlib import sha256
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
from kira.roles.registry import RoleRegistryError, validate_roles
from kira.settings.root import load_settings, reset_settings

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

    def test_an_asset_path_cannot_leave_the_package(self) -> None:
        """The prototype refused this and the three readers layered above still do; the
        one they funnel through lost it. It is reachable through the roles entry point,
        where a third party's RoleSpec.prompt would be read into a system prompt."""
        with pytest.raises(PromptError, match=r"escapes kira\.prompts"):
            read_asset("../../../pyproject.toml")

    def test_an_asset_ships_in_another_package_and_is_read_the_same_way(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An environment adapter ships its prompts inside its own package.

        It reads them through this function rather than a second one, so what it gets
        is what the core gets: the same strip on load, the same containment check, and
        the same two-way strict render — a renamed marker fails loudly instead of
        shipping a literal `{{TASK_TYPE}}` to the model.
        """
        package = tmp_path / "kira_asset_anchor_probe"
        package.mkdir()
        (package / "__init__.py").write_text("", encoding="utf-8")
        (package / "contract.txt").write_text("Answer the {{TASK_TYPE}} task.\n", encoding="utf-8")
        monkeypatch.syspath_prepend(str(tmp_path))
        importlib.invalidate_caches()

        anchor = "kira_asset_anchor_probe"
        template = read_asset("contract.txt", package=anchor)

        assert template == "Answer the {{TASK_TYPE}} task."
        assert render(template, {"TASK_TYPE": "detection"}) == "Answer the detection task."
        with pytest.raises(PromptError, match="no value supplied"):
            render(template, {})
        with pytest.raises(PromptError, match="escapes kira_asset_anchor_probe"):
            read_asset("../pyproject.toml", package=anchor)

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
    """The loader called this the single supported way to change a prompt, and nothing
    called it and no setting reached it: every role read the packaged asset, so the one
    modularity claim the prompt layer makes was unsupplied."""
    (tmp_path / "roles").mkdir()
    (tmp_path / "roles" / "rca.txt").write_text(
        "## Goal\n\nDo it my way.\n\n{{KUBECTL_GUIDANCE}}\n", encoding="utf-8"
    )
    prompt = compose_role_prompt("rca", retrieval_enabled=False, override_dir=tmp_path)
    assert "Do it my way." in prompt
    assert "exec_kubectl_cmd_safely" in prompt

    # A role the directory says nothing about keeps the packaged prompt, so replacing
    # one role does not mean forking the set.
    assert "Do it my way." not in compose_role_prompt(
        "planning", retrieval_enabled=False, override_dir=tmp_path
    )


def test_a_role_without_cluster_tools_may_not_declare_the_kubectl_marker(
    tmp_path: Path,
) -> None:
    (tmp_path / "roles").mkdir()
    (tmp_path / "roles" / "invented.txt").write_text("{{KUBECTL_GUIDANCE}}", encoding="utf-8")
    with pytest.raises(PromptError, match="holds no cluster tools"):
        compose_role_prompt("invented", retrieval_enabled=False, override_dir=tmp_path)


ROLES_WITH_A_KUBECTL_MARKER = ["rca", "planning", "execution", "verify", "learning", "single"]


@pytest.mark.parametrize("role", ROLES_WITH_A_KUBECTL_MARKER)
def test_every_role_composes_on_the_proxy_route_too(role: str) -> None:
    """A deployment whose transport offers the proxy has to compose every role.

    A missing block is only reachable through that route, so without this it would
    surface as a role that cannot build its system prompt during an incident.
    """
    prompt = compose_role_prompt(role, retrieval_enabled=False, kubectl_route="proxy_exec")
    assert prompt.strip()
    assert "{{" not in prompt


def test_the_proxy_route_tells_a_role_to_run_kubectl_and_the_tool_route_not_to() -> None:
    """The two routes prescribe opposite things, which is why the route has to reach here.

    On the tool route the sandbox refuses kubectl and the block says to use the cluster
    tool. On the proxy route kubectl resolves to a stage-aware shim and the block says
    to run it directly. Composing the wrong one leaves the model with a system prompt
    contradicting the tool it was handed, which costs it turns on refusals.
    """
    tool_route = compose_role_prompt("execution", retrieval_enabled=False)
    proxied = compose_role_prompt("execution", retrieval_enabled=False, kubectl_route="proxy_exec")

    assert "exec_kubectl_cmd_safely" in tool_route
    assert "run normal `kubectl` commands directly" in proxied
    assert "`exec_kubectl_cmd_safely` and rollback tools are not available" in proxied


def test_an_unknown_route_is_refused_rather_than_silently_composing_the_other_one() -> None:
    """A typo would otherwise fall through to the cluster-tool blocks, and a whole
    campaign would run on guidance describing a route it was not using."""
    with pytest.raises(PromptError, match="unknown kubectl route"):
        compose_role_prompt("rca", retrieval_enabled=False, kubectl_route="proxy-exec")


@pytest.mark.parity
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


@pytest.mark.parity
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


def test_a_replaced_prompt_reaches_the_run_that_used_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The directory path alone does not say what was in it. A result produced under a
    prompt nobody recorded is not attributable, which is the same reason the arm file is
    digested into the same snapshot.

    Written with a `~`, because that is what a ConfigMap or a compose `environment:`
    block carries literally and pydantic keeps: expanded by the digester and not by the
    reader, the snapshot recorded a replacement while the packaged prompt ran.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "prompts" / "roles").mkdir(parents=True)
    override = tmp_path / "prompts" / "roles" / "rca.txt"
    override.write_text("## Goal\n\nDo it my way.\n\n{{KUBECTL_GUIDANCE}}\n", encoding="utf-8")

    monkeypatch.setenv("KIRA_PROMPTS__OVERRIDE_DIR", "~/prompts")
    reset_settings()
    settings = load_settings()

    assert settings.snapshot()["prompt_overrides"] == {
        "rca": sha256(override.read_bytes()).hexdigest()
    }
    composed = compose_role_prompt(
        "rca", retrieval_enabled=False, override_dir=settings.prompts.override_dir
    )
    assert "Do it my way." in composed


def test_an_empty_override_fails_at_startup_rather_than_silently(tmp_path: Path) -> None:
    """A Helm template that renders to nothing, or `--from-file` on an empty file, was
    falsy: startup validated the packaged prompt and passed, and the first incident sent
    the diagnosing role an empty system prompt with no error anywhere."""
    (tmp_path / "roles").mkdir()
    (tmp_path / "roles" / "rca.txt").write_text("", encoding="utf-8")
    with pytest.raises(RoleRegistryError, match="is empty"):
        validate_roles(override_dir=tmp_path)


def test_an_unfillable_override_fails_at_startup_not_at_the_incident(tmp_path: Path) -> None:
    """A packaged prompt declaring a marker nothing fills is refused when the roles are
    validated. An override doing the same thing has to fail in the same place, or the
    first incident to use it burns its budget on a prompt that cannot render."""
    (tmp_path / "roles").mkdir()
    (tmp_path / "roles" / "rca.txt").write_text("{{NOT_A_REAL_MARKER}}", encoding="utf-8")
    with pytest.raises(RoleRegistryError, match="declares markers nothing fills"):
        validate_roles(override_dir=tmp_path)


@pytest.mark.parity
@pytest.mark.parametrize(
    ("role", "expected"),
    [
        ("rca", "read_only"),
        ("planning", "read_only"),
        ("execution", "execution"),
        ("verify", "read_only"),
        ("learning", "read_only"),
        ("single", "single"),
    ],
)
def test_the_proxy_route_block_map_mirrors_the_archive(role: str, expected: str) -> None:
    """Planning is the one that differs from the cluster-tool map.

    It has its own block on the tool route and reads the shared read-only text on the
    proxy route, which is what the archived host's own composed system prompts carry.
    Giving it a Planning-specific block here would change a prompt the published
    measurements were taken under.
    """
    from kira.prompts.loader import _KUBECTL_PROXY_BLOCK  # noqa: PLC0415

    assert _KUBECTL_PROXY_BLOCK[role] == expected
