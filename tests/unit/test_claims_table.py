"""The claims table, and the limitations beside it, have to stay true.

A table that cites a test file which has since been renamed is worse than no table: it
is a claim of evidence, and a reader who runs the command and gets "file not found"
learns that the artifact was not checked rather than that one path moved.

The limitations section is the same problem read the other way. It is the only place a
reader is told what the system does not do, so a deferral that is quietly dropped from it
while the code still defers reads as a capability. The list below is what the deferral
document declares, and the check is that the README still says all of it.
"""

import pathlib
import re

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
README = (REPO / "README.md").read_text(encoding="utf-8")
PROSE = " ".join(README.split())
"""The same text with its line wrapping removed, so a check on a sentence does not
depend on where the paragraph happened to break."""

CITED_TESTS = sorted(
    {pathlib.Path(match) for match in re.findall(r"tests/unit/test_[a-z_]+\.py", README)}
)
CITED_FILES = sorted(
    {
        pathlib.Path(match)
        for match in re.findall(r"(?:deploy|configs|examples|src)/[\w./*-]+", README)
    }
)

LIMITATIONS = "## Limitations and known issues"

DECLARED_DEFERRALS = (
    "no authentication or authorisation on the api",
    "single tenant",
    "no rate limiting",
    "no scale-out and no leader election",
    "no backup or restore automation",
    "no prompt-injection hardening",
)
"""The six deferrals the production-readiness document declares, lowercased.

Nothing in the code enforces this list; that is exactly why it is checked here. Each
is a posture an adopter has to plan around, and the README is where they read it.
"""


def test_the_table_cites_something() -> None:
    """A regex that silently matched nothing would make every check below vacuous."""
    assert len(CITED_TESTS) >= 10


@pytest.mark.parametrize("path", CITED_TESTS, ids=str)
def test_every_cited_test_exists(path: pathlib.Path) -> None:
    assert (REPO / path).is_file()


@pytest.mark.parametrize("pattern", CITED_FILES, ids=str)
def test_every_cited_path_exists(pattern: pathlib.Path) -> None:
    if "*" in str(pattern):
        assert list(REPO.glob(str(pattern))), f"{pattern} matches nothing"
    else:
        assert (REPO / pattern).exists()


def test_the_approval_caveat_survives_an_edit() -> None:
    """The single most overstatable row. Kira ships the mechanism and tests it; no
    measured campaign ever put a request in front of a person, and a table that let
    that sentence be edited away would be claiming otherwise by omission."""
    assert "no request ever reached a person" in PROSE


def _limitations_start() -> int:
    """Where the section begins, said once so a missing heading names itself."""
    start = README.find(LIMITATIONS)
    assert start != -1, f"README has no {LIMITATIONS!r} section"
    return start


def _limitations_section() -> str:
    """The README from the limitations heading to the next one, wrapping removed."""
    start = _limitations_start()
    rest = README.index("\n## ", start + len(LIMITATIONS))
    return " ".join(README[start:rest].split()).lower()


def test_the_readme_carries_a_limitations_section() -> None:
    """Without the heading every check below would pass on a README that says nothing."""
    assert LIMITATIONS in README


@pytest.mark.parametrize("deferral", DECLARED_DEFERRALS)
def test_every_declared_deferral_is_disclosed(deferral: str) -> None:
    """A reader-facing list that drifts from the code's posture is what this guards."""
    assert deferral in _limitations_section()


def test_the_provenance_statement_sits_with_the_claims() -> None:
    """The claims table is where the numbers are read, so it is where the sentence that
    says none of them were measured here has to be; a reader who stops after the table
    must not leave believing this code produced them."""
    where = README.lower().find("no measured campaign has been run on this code")
    assert where != -1, "the provenance sentence is gone"
    assert where < _limitations_start(), "it belongs with the claims, not only below them"
