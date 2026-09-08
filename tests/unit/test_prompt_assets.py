"""Every shipped prompt asset is byte-identical to the prototype it came from.

Prompt text is a held-constant input to every published measurement, so a migrated
asset that has been reflowed, retyped or silently "fixed" invalidates the comparison
it is supposed to preserve. The check walks the shipped tree rather than listing
files, so an asset added without a prototype counterpart fails instead of passing
unnoticed.

The prototype is not part of this repository, so the assets compared against are
frozen under ``tests/fixtures/prototype``; see :mod:`prototype_tree` for the copy and
for the environment name that compares against a live checkout instead.
"""

from pathlib import Path

import pytest
from prototype_tree import prototype_root

pytestmark = pytest.mark.parity
"""Prototype-pinned throughout: see the module docstring for where the data came from."""

REPO_ROOT = Path(__file__).resolve().parents[2]
SHIPPED = REPO_ROOT / "src" / "kira" / "prompts"

# An environment adapter ships its own prompts inside its own package, read through the
# same loader. They are migrated text like every other asset, so they are walked here
# rather than compared somewhere the frozen-copy check cannot see: an adapter asset with
# no recorded prototype source would otherwise be the one prompt nothing holds to the
# migration rule.
ADAPTER_TREES = {
    Path("adapters", "aiopslab"): REPO_ROOT / "src" / "kira" / "adapters" / "aiopslab" / "prompts"
}

PROTOTYPE = prototype_root() / "agent" / "prompts"

# Composition markers were renamed on the way in, and only in the role system prompts.
# Runtime blocks carry their own markers, including a {{MEMORY_GUIDANCE}} that names a
# different block entirely, so the rename must not be applied to them.
ROLE_MARKER_RENAMES = {
    "{{KUBECTL_MODE_PROMPT}}": "{{KUBECTL_GUIDANCE}}",
    "{{MODE_SPECIFIC_PROMPT}}": "{{MEMORY_GUIDANCE}}",
}

# The three roles whose shipped prompt is the prototype's goal-only variant. The
# prescriptive prompt is the ablation arm, not the default.
MINIMAL_VARIANT_ROLES = {"rca", "planning", "execution"}

MEMORY_STATE_TO_MODE = {"enabled": "learning", "disabled": "non_learning"}

# Assets whose prototype source is Python, not a prompt file. The health gate's
# model-facing blocks were string literals inside `agent/agents/health_gate.py`, and the
# house rule that prompt text lives in .txt files applies to them on the way in. There is
# nothing here to diff against, so they are listed by name: an asset added without a
# prototype counterpart still fails unless someone writes down where it came from.
CARRIED_FROM_PROTOTYPE_CODE = {
    Path("runtime/verify/health_gate_header.txt"),
    Path("runtime/verify/health_gate_healthy.txt"),
    Path("runtime/verify/health_gate_unhealthy.txt"),
    Path("runtime/verify/submit_while_unhealthy.txt"),
    # The AIOpsLab adapter's four: the connectivity-contract block was a string literal
    # in `agent/agents/verify.py`, the incident-scope block and its two approval
    # sentences in `agent/agents/execution.py`, and the detection guard's rejection in
    # `agent/agents/rca.py`. Same rule as the health gate's blocks: model-facing text
    # belongs in a .txt file on the way in, and there is nothing to diff against.
    Path("adapters/aiopslab/verify_contract.txt"),
    Path("adapters/aiopslab/execution_scope.txt"),
    Path("adapters/aiopslab/approval_auto.txt"),
    Path("adapters/aiopslab/approval_gated.txt"),
    Path("adapters/aiopslab/detection_guard_rejection.txt"),
}

# Assets that deliberately no longer match, and why. Byte fidelity is the migration
# rule because prompt text is a held-constant input, so a divergence is a decision
# someone has to make and write down rather than a diff someone has to explain later.
DELIBERATELY_DIVERGED = {
    Path("roles/learning.txt"): (
        "prescribed a playbook layer this system does not have: the four playbook_* "
        "tools are not registered and the domain/human triggers are not shipped"
    ),
    Path("runtime/learning/post_incident.txt"): (
        "prescribed the same playbook layer as the role prompt it is appended to: "
        "playbook_read and playbook_remove are not registered tools"
    ),
    Path("runtime/verify/context_blind.txt"): (
        "dropped {{AIOPSLAB_VERIFY_CONNECTIVITY_CONTRACT}}, a benchmark-named slot no "
        "adapter can fill; it rendered empty, so the composed message is unchanged"
    ),
}


def _prototype_source(relative: Path) -> Path:
    """Map a shipped asset back to the prototype file it was copied from."""
    parts = relative.parts
    match parts:
        case ("roles", "learning.txt"):
            # The learning role's system prompt is the prototype's shared learning base.
            # Its trigger-specific half is appended at runtime, exactly as there.
            return PROTOTYPE / "learning" / "base.txt"
        case ("roles", filename):
            role = filename.removesuffix(".txt")
            if role in MINIMAL_VARIANT_ROLES:
                return PROTOTYPE / "minimal" / filename
            return PROTOTYPE / filename
        case ("blocks", "kubectl", filename):
            # The prototype kept one block family per kubectl route: `safe.*` for the
            # cluster-tool route and `proxy_exec.*` for the one where a role runs its
            # own kubectl against the proxy. Only the first was prefixed on the way in,
            # so the second keeps the name it already has.
            if filename.startswith("proxy_exec."):
                return PROTOTYPE / "shared" / "kubectl" / filename
            return PROTOTYPE / "shared" / "kubectl" / f"safe.{filename}"
        case ("blocks", "memory", filename):
            role, state = filename.removesuffix(".txt").split(".")
            return PROTOTYPE / f"{role}.{MEMORY_STATE_TO_MODE[state]}.txt"
        case ("control", filename):
            return PROTOTYPE / "runtime" / "control" / filename
        case ("runtime", "learning", filename):
            return PROTOTYPE / "learning" / filename
        case ("runtime", "shared", filename):
            return PROTOTYPE / "runtime" / "roles" / filename
        case ("runtime", role, filename):
            return PROTOTYPE / "runtime" / "roles" / role / filename
        case ("adapters", "aiopslab", filename):
            return PROTOTYPE / "runtime" / "aiopslab" / filename
        case ("secondpass", filename):
            # The blinded selector was borrowed from the counterfactual fork arm, which
            # Kira drops. Its two prompts are live second-pass code, so they move here
            # while the rest of that module dies with the arm.
            if filename.startswith("selector_"):
                return PROTOTYPE / "runtime" / "counterfactual" / filename
            return PROTOTYPE / "runtime" / "second_opinion" / filename
        case _:
            pytest.fail(f"shipped prompt asset has no known prototype source: {relative}")


def _shipped_assets() -> list[Path]:
    trees = {Path("."): SHIPPED, **ADAPTER_TREES}
    return sorted(
        prefix / path.relative_to(root)
        for prefix, root in trees.items()
        for path in root.rglob("*")
        if path.is_file() and path.suffix in {".txt", ".yaml"}
    )


def _shipped_file(relative: Path) -> Path:
    """Resolve a walked asset back to the tree it was found in."""
    for prefix, root in ADAPTER_TREES.items():
        if relative.is_relative_to(prefix):
            return root / relative.relative_to(prefix)
    return SHIPPED / relative


@pytest.mark.parametrize("relative", _shipped_assets(), ids=str)
def test_every_shipped_asset_is_byte_identical_to_the_prototype(relative: Path) -> None:
    if relative in CARRIED_FROM_PROTOTYPE_CODE:
        pytest.skip("carried from prototype Python; no prompt file to diff against")
    if relative in DELIBERATELY_DIVERGED:
        pytest.skip(DELIBERATELY_DIVERGED[relative])
    expected = _prototype_source(relative).read_bytes()
    if relative.parts[0] == "roles":
        text = expected.decode("utf-8")
        for old, new in ROLE_MARKER_RENAMES.items():
            text = text.replace(old, new)
        expected = text.encode("utf-8")
    assert _shipped_file(relative).read_bytes() == expected
