"""The frozen copy of the prototype is complete, and no larger than complete.

The three parity checks resolve their prototype through `prototype_tree.prototype_root`,
which is the frozen copy unless a live checkout is named. Each of them reads its own
files, so a refreeze that copies some of the set and forgets the rest leaves them
comparing against a mixture of two prototype revisions — and every one of them still
passes, because a file that is there is all any of them looks at.

So the set is checked here rather than by each reader: every file the three open is
present, the copy holds nothing besides, and the manifest says which commit it was
taken from. Without the commit the copy is unauditable text of unknown provenance,
which is the thing the comparisons exist to rule out.
"""

import re
from pathlib import Path

import pytest
import test_kubectl_proxy
import test_projections
import test_prompt_assets
import test_toolsets
from prototype_tree import FROZEN, MANIFEST, prototype_root

ROOT = prototype_root()
"""Resolved at import, as in the three modules below: `clean_env` scrubs the name."""

COMMIT = re.compile(r"^commit: ([0-9a-f]{40})$", re.MULTILINE)
DATE = re.compile(r"^date: (\d{4}-\d{2}-\d{2})$", re.MULTILINE)


def _prompt_assets() -> set[Path]:
    """The prototype prompt files test_prompt_assets actually opens."""
    unread = test_prompt_assets.CARRIED_FROM_PROTOTYPE_CODE | set(
        test_prompt_assets.DELIBERATELY_DIVERGED
    )
    sources = set()
    for relative in test_prompt_assets._shipped_assets():
        if relative in unread:
            continue
        try:
            sources.add(test_prompt_assets._prototype_source(relative))
        except pytest.fail.Exception:
            # A shipped asset with no known source is that module's failure to report.
            continue
    return sources


COMPARED = tuple(
    sorted(
        path.relative_to(ROOT)
        for path in {
            test_toolsets.PROTOTYPE_REGISTRY,
            test_projections.PROTOTYPE,
            test_kubectl_proxy.PROTOTYPE_SHIM,
        }
        | _prompt_assets()
    )
)
"""Every prototype file the parity checks read, relative to whichever tree they read."""


@pytest.mark.parity
def test_the_manifest_names_the_commit_it_was_frozen_from() -> None:
    manifest = MANIFEST.read_text(encoding="utf-8")
    assert COMMIT.search(manifest), f"{MANIFEST} names no 40-character prototype commit"
    assert DATE.search(manifest), f"{MANIFEST} gives no date for the freeze"


@pytest.mark.parity
def test_the_frozen_copy_holds_exactly_what_the_checks_read() -> None:
    """Vendoring more than the checks read makes the copy a second prototype to keep
    current, and the extra text ages without anything noticing."""
    # Running the frozen renderer leaves bytecode beside it, which git ignores.
    present = {
        path.relative_to(FROZEN)
        for path in FROZEN.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }
    assert present - {MANIFEST.relative_to(FROZEN)} == set(COMPARED)
