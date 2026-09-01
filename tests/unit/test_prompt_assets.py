"""Every shipped prompt asset is byte-identical to the prototype it came from.

Prompt text is a held-constant input to every published measurement, so a migrated
asset that has been reflowed, retyped or silently "fixed" invalidates the comparison
it is supposed to preserve. The check walks the shipped tree rather than listing
files, so an asset added without a prototype counterpart fails instead of passing
unnoticed.

The prototype is not part of this repository. Point ``PROTOTYPE_PROMPTS`` at its
``agent/prompts`` directory, or keep it checked out beside this one; the test skips
when it is absent.
"""

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.parity
"""Prototype-pinned throughout: see the module docstring for where the data came from."""

REPO_ROOT = Path(__file__).resolve().parents[2]
SHIPPED = REPO_ROOT / "src" / "kira" / "prompts"

PROTOTYPE = Path(
    os.environ.get(
        "PROTOTYPE_PROMPTS",
        REPO_ROOT.parent / "k8s-agentic-operator" / "agent" / "prompts",
    )
)

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
    return sorted(
        path.relative_to(SHIPPED)
        for path in SHIPPED.rglob("*")
        if path.is_file() and path.suffix in {".txt", ".yaml"}
    )


@pytest.mark.skipif(not PROTOTYPE.is_dir(), reason=f"prototype prompts not at {PROTOTYPE}")
@pytest.mark.parametrize("relative", _shipped_assets(), ids=str)
def test_every_shipped_asset_is_byte_identical_to_the_prototype(relative: Path) -> None:
    if relative in CARRIED_FROM_PROTOTYPE_CODE:
        pytest.skip("carried from prototype Python; no prompt file to diff against")
    expected = _prototype_source(relative).read_bytes()
    if relative.parts[0] == "roles":
        text = expected.decode("utf-8")
        for old, new in ROLE_MARKER_RENAMES.items():
            text = text.replace(old, new)
        expected = text.encode("utf-8")
    assert (SHIPPED / relative).read_bytes() == expected
