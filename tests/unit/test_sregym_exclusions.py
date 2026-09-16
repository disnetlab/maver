"""Leave-one-out over an imported corpus, where no record can say which problem it is.

The label path and the map path answer the same question from opposite ends, and a
deployment can hold records of both kinds at once, so what is asserted here is the union
and the two ways it can be wrong: a map that silently withholds nothing, and a map that
is read at the cell it was needed for rather than at startup.
"""

import logging
import pathlib
import re
from collections.abc import Sequence

import pytest

from maver.adapters.base import EpisodeContext
from maver.adapters.sregym import SREGymAdapter, SregymConfig
from maver.adapters.sregym.exclusions import load_exclusion_map
from maver.settings.root import load_settings

PROBLEM = "k8s_target_port-misconfig"

MAP = (
    f"{PROBLEM}\timported-1, imported-2,imported-1\n"
    "\n"
    "misc_config_baseline_hotel_reservation\timported-9\n"
    "produced_nothing\t\n"
)


class Corpus:
    """A stand-in for the corpus, answering the one read leave-one-out makes.

    A fake rather than a store, because what is under test is which sets are combined and
    not how either is found; the label read itself is pinned against a real store in
    ``test_sregym_adapter.py``.
    """

    def __init__(self, labelled: Sequence[str] = ()) -> None:
        self.labelled = list(labelled)
        self.asked: list[tuple[str, str]] = []

    def session_ids_with_label(self, name: str, value: str) -> list[str]:
        self.asked.append((name, value))
        return list(self.labelled)


def _map(tmp_path: pathlib.Path, text: str = MAP) -> pathlib.Path:
    path = tmp_path / "exclusions.tsv"
    path.write_text(text, encoding="utf-8")
    return path


def _adapter(path: pathlib.Path | None) -> SREGymAdapter:
    return SREGymAdapter(SregymConfig(exclusion_map=path))


def test_the_map_reads_one_row_per_problem(tmp_path: pathlib.Path) -> None:
    """Blank lines are skipped, a repeated id is one id, and an empty row is a statement."""
    excluded = load_exclusion_map(_map(tmp_path))

    assert excluded[PROBLEM] == ("imported-1", "imported-2")
    assert excluded["misc_config_baseline_hotel_reservation"] == ("imported-9",)
    assert excluded["produced_nothing"] == ()


def test_the_withheld_set_is_the_union_of_the_labels_and_the_map(
    tmp_path: pathlib.Path,
) -> None:
    """A corpus can be part imported and part written here, and both parts are withheld."""
    corpus = Corpus(["distilled-1", "imported-2"])

    withheld = _adapter(_map(tmp_path)).withheld_records(
        EpisodeContext(namespace="shop", external_key=PROBLEM), corpus
    )

    assert list(withheld) == ["distilled-1", "imported-1", "imported-2"]
    assert corpus.asked == [("external_key", PROBLEM)]


def test_a_problem_the_map_does_not_name_warns_and_withholds_what_it_can(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The archived runner aborted the cell here. This one runs it and says so.

    A long-lived process serves whatever episode the harness starts next, so refusing
    would take a sweep down over one missing row; the warning is the record that this
    cell ran with less withheld than the protocol says.
    """
    corpus = Corpus(["distilled-1"])
    ctx = EpisodeContext(namespace="shop", external_key="a_problem_nobody_listed")

    with caplog.at_level(logging.WARNING, logger="maver.adapters.sregym"):
        withheld = _adapter(_map(tmp_path)).withheld_records(ctx, corpus)

    assert list(withheld) == ["distilled-1"]
    assert "no row for problem a_problem_nobody_listed" in caplog.text


def test_no_map_is_the_derived_set_and_no_warning(caplog: pytest.LogCaptureFixture) -> None:
    """The map is opt-in: a deployment whose records carry their own labels needs none."""
    corpus = Corpus(["distilled-1"])

    with caplog.at_level(logging.WARNING, logger="maver.adapters.sregym"):
        withheld = _adapter(None).withheld_records(
            EpisodeContext(namespace="shop", external_key=PROBLEM), corpus
        )

    assert list(withheld) == ["distilled-1"]
    assert caplog.text == ""


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (f"{PROBLEM} imported-1\n", "found 1 tab-separated field(s)"),
        (f"{PROBLEM}\timported-1\textra\n", "found 3 tab-separated field(s)"),
        ("\timported-1\n", "the problem id is empty"),
        (f"{PROBLEM}\ta\n{PROBLEM}\tb\n", "already has a row on line 1"),
    ],
)
def test_a_line_that_cannot_be_read_stops_the_deployment(
    tmp_path: pathlib.Path, text: str, expected: str
) -> None:
    """At startup, where the adapter is built, and not at the cell that needed the row.

    An exclusion that fails open voids the cell it was measuring, and does so in the
    direction that flatters the result, so this is a refusal rather than a warning.
    """
    path = _map(tmp_path, text)

    with pytest.raises(ValueError, match=re.escape(expected)):
        _adapter(path)


def test_a_map_that_is_not_there_is_refused_by_name(tmp_path: pathlib.Path) -> None:
    with pytest.raises(ValueError, match="could not be read"):
        _adapter(tmp_path / "absent.tsv")


def test_the_map_is_named_by_the_deployment_environment(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The adapter slice is untyped in the core, so the name is checked nowhere else."""
    path = _map(tmp_path)
    monkeypatch.setenv("MAVER_ADAPTERS__SREGYM__EXCLUSION_MAP", str(path))

    assert SregymConfig.resolve(load_settings(), env={}).exclusion_map == path
