"""The claims table has to stay true.

A table that cites a test file which has since been renamed is worse than no table: it
is a claim of evidence, and a reader who runs the command and gets "file not found"
learns that the artifact was not checked rather than that one path moved.
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
    {pathlib.Path(match) for match in re.findall(r"(?:deploy|configs|examples)/[\w./*-]+", README)}
)


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
