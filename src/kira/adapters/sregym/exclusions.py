"""Leave-one-out for a corpus whose records cannot say which problem they came from.

The adapter derives the withheld set from the records themselves: the distilling pass
copies the episode's problem id onto every record it writes, so an episode asks the
corpus which records are its own and nothing has to be kept in step. An *imported* corpus
breaks that, and the archive is one — the campaign's sessions carried the problem id
nowhere, so the records exported from them carry no label to key on and the derived set
comes back empty. Empty is the direction that flatters a memory arm: the cell retrieves
its own answer and scores well.

The archive solved this with a file, and so does this. One line per problem naming the
sessions that problem produced, built once from the run log, read at startup, and
consulted per episode alongside the labels rather than instead of them — a corpus that is
partly imported and partly written by this build has both, and both hold.
"""

import logging
from collections.abc import Mapping
from pathlib import Path

__all__ = ["load_exclusion_map"]

_log = logging.getLogger(__name__)


def load_exclusion_map(path: Path) -> Mapping[str, tuple[str, ...]]:
    """Read the problem-to-sessions map, or refuse to start.

    The format is the archive's: one line per problem, the problem id and a
    comma-separated list of session ids separated by a tab. Blank lines are skipped. A
    row whose list is empty is a problem that produced no records, which is a statement
    and not an omission: it withholds nothing and warns about nothing.

    Args:
        path: The TSV file.

    Returns:
        Problem id to the session ids it produced, in the file's order and without
        duplicates.

    Raises:
        ValueError: When the file is missing or a line cannot be read, naming the line.
            Raised at startup, where the adapter is built, rather than at the episode
            that would have needed the row: a leave-one-out set that fails open voids
            the cell it was measuring and does so silently.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"the exclusion map {path} could not be read: {exc}") from exc

    excluded: dict[str, tuple[str, ...]] = {}
    lines: dict[str, int] = {}
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        problem, sessions = _row(line, number, path)
        if (first := lines.get(problem)) is not None:
            # Two rows for one problem would make the withheld set depend on which one
            # the reader kept, and the whole point of the file is that it is auditable.
            raise ValueError(
                f"{path} line {number}: problem {problem!r} already has a row on line "
                f"{first}; one problem is one row"
            )
        lines[problem] = number
        excluded[problem] = sessions
    _log.info("read leave-one-out rows for %d problem(s) from %s", len(excluded), path)
    return excluded


def _row(line: str, number: int, path: Path) -> tuple[str, tuple[str, ...]]:
    """Split one line into its problem and its sessions."""
    fields = line.split("\t")
    if len(fields) != 2:
        raise ValueError(
            f"{path} line {number}: expected a problem id and a comma-separated list of "
            f"session ids separated by one tab, found {len(fields)} tab-separated "
            f"field(s): {line!r}"
        )
    problem = fields[0].strip()
    if not problem:
        raise ValueError(f"{path} line {number}: the problem id is empty: {line!r}")
    sessions = tuple(dict.fromkeys(part.strip() for part in fields[1].split(",") if part.strip()))
    return problem, sessions
