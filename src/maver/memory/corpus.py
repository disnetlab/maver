"""Loading a corpus that was written somewhere else.

The distilling pass is the only thing that writes a record during a run, and it writes
one incident at a time. An archived corpus arrives whole, out of a database this build
does not own, so it needs a way in — and that way must not be a database restore. Both
backends keep a normalised index beside the raw text (FTS5 terms in the embedded store, a
``tsvector`` in the server one), built by the same normaliser retrieval queries with, so a
corpus loaded by copying rows is a corpus whose index was filled by whoever wrote the copy
script. Everything here therefore goes through
:meth:`~maver.memory.ports.TextSearch.upsert_summary`, exactly as the distilling pass does,
and nothing here opens a database.

The file is JSON Lines whose keys are the record's own field names. That is a deliberate
non-format: it names no source schema, so an export from any prototype table is a
``SELECT`` an operator can read, and a record that arrives with a field this build does
not know about is refused by name rather than dropped in silence.
"""

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields
from json import JSONDecodeError, loads
from typing import Any

from maver.contracts.lifecycle import LifecycleState
from maver.memory.ports import SessionMemory, TextSearch

__all__ = ["ImportReport", "MalformedRecord", "import_records", "parse_records"]

_log = logging.getLogger(__name__)

_STATES = frozenset(member.value for member in LifecycleState)

_FIELDS = frozenset(field.name for field in fields(SessionMemory))

_FLAGS = frozenset({"evaluation_diagnosis_success", "evaluation_mitigation_success"})
_BOOLS = frozenset({"authoritative_correction"})
_NUMBERS = frozenset({"evaluation_diagnosis_score"})
_STRINGS = _FIELDS - _FLAGS - _BOOLS - _NUMBERS - {"labels"}
"""Derived rather than listed, so that a new text field on the record is importable in
the commit that adds it. A field that is *not* text has to be named above, which is the
one thing this arrangement makes noisy on purpose."""


class MalformedRecord(ValueError):
    """A line that cannot be read as a record.

    Every message names the line, because an import is refused whole: a file that half
    loaded would leave a corpus nobody can describe, and "some line is wrong" is not
    something an operator can act on with a 123-line export in front of them.
    """


@dataclass(frozen=True, slots=True)
class ImportReport:
    """What one import did."""

    imported: int
    """Records the corpus did not already hold."""
    replaced: int
    """Records that were already there and were overwritten."""
    skipped: int
    """Records that were already there and were left alone."""
    unknown_states: tuple[str, ...] = ()
    """Lifecycle states this build does not name, as they were stored.

    Reported rather than refused: a record carries whatever standing the campaign that
    produced it recorded. It matters because ranking demotes by state, and a state the
    table does not name is worth :data:`~maver.memory.lifecycle.DEFAULT_LIFECYCLE_WEIGHT`
    — so a typo in an export is a silent re-weighting of the whole corpus.
    """


def parse_records(text: str) -> list[SessionMemory]:
    """Read every record in a JSON Lines file, or refuse the file.

    Args:
        text: The file's contents. Blank lines are skipped; every other line must be one
            JSON object whose keys are :class:`~maver.memory.ports.SessionMemory` field
            names, with ``labels`` as an object of strings.

    Returns:
        The records, in the order the file gives them.

    Raises:
        MalformedRecord: On the first line that cannot be read, naming it. Nothing is
            written by this function, so a refusal here leaves the corpus untouched.
    """
    records: list[SessionMemory] = []
    seen: dict[str, int] = {}
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        record = _record(_object(line, number), number)
        if (first := seen.get(record.session_id)) is not None:
            # Two rows for one session make the result depend on which one the store
            # happened to write last, which is the one thing an idempotent import may
            # not do.
            raise MalformedRecord(
                f"line {number}: session {record.session_id!r} was already given on "
                f"line {first}; one session is one record"
            )
        seen[record.session_id] = number
        records.append(record)
    return records


def import_records(
    index: TextSearch, records: Iterable[SessionMemory], *, replace: bool = False
) -> ImportReport:
    """Write records into a corpus, keyed by session.

    Args:
        index: The corpus to write to. Which campaign's corpus that is was decided when
            the store was opened.
        records: What :func:`parse_records` produced.
        replace: Whether a session the corpus already holds is overwritten. Off, it is
            left exactly as it is, so re-running an interrupted import is safe and cannot
            silently revise a record a run has already retrieved.

    Returns:
        The counts, for a caller to print.
    """
    imported = replaced = skipped = 0
    states: set[str] = set()
    for record in records:
        if index.open_record(record.session_id) is not None:
            if not replace:
                skipped += 1
                continue
            replaced += 1
        else:
            imported += 1
        index.upsert_summary(record)
        if not _known(record.lifecycle_state):
            states.add(record.lifecycle_state)
    _log.info(
        "imported %d record(s), replaced %d, skipped %d already present",
        imported,
        replaced,
        skipped,
    )
    return ImportReport(
        imported=imported, replaced=replaced, skipped=skipped, unknown_states=tuple(sorted(states))
    )


def _object(line: str, number: int) -> Mapping[str, Any]:
    """Read one line as a JSON object."""
    try:
        payload = loads(line)
    except JSONDecodeError as exc:
        raise MalformedRecord(f"line {number}: not JSON ({exc.msg})") from exc
    if not isinstance(payload, dict):
        raise MalformedRecord(
            f"line {number}: expected one JSON object per line, found {type(payload).__name__}"
        )
    return payload


def _record(payload: Mapping[str, Any], number: int) -> SessionMemory:
    """Build one record, refusing anything the store could not honestly hold."""
    if unknown := sorted(set(payload) - _FIELDS):
        raise MalformedRecord(
            f"line {number}: unknown field(s) {', '.join(unknown)}; the keys are the "
            f"record's own fields: {', '.join(sorted(_FIELDS))}"
        )
    values: dict[str, Any] = dict.fromkeys(_STRINGS, "")
    labels: dict[str, str] = {}
    for key, value in payload.items():
        if key in _STRINGS:
            values[key] = _text(key, value, number)
        elif key in _FLAGS:
            values[key] = _flag(key, value, number)
        elif key in _BOOLS:
            if not isinstance(value, bool):
                raise MalformedRecord(_wrong(key, value, number, "true or false"))
            values[key] = value
        elif key in _NUMBERS:
            values[key] = _number(key, value, number)
        else:
            labels = _labels(value, number)
    record = SessionMemory(**values, labels=labels)
    if not record.session_id.strip():
        raise MalformedRecord(f"line {number}: session_id is required and names the record")
    if not (record.evidence_text.strip() or record.lesson_text.strip()):
        # Both projections are what retrieval matches on. A record with neither is not a
        # record that ranks badly; it is one no query can ever return.
        raise MalformedRecord(
            f"line {number}: session {record.session_id!r} carries neither evidence_text "
            "nor lesson_text, so no retrieval could ever match it"
        )
    return record


def _text(key: str, value: Any, number: int) -> str:
    if not isinstance(value, str):
        raise MalformedRecord(_wrong(key, value, number, "a string"))
    return value


def _flag(key: str, value: Any, number: int) -> bool | None:
    if value is not None and not isinstance(value, bool):
        raise MalformedRecord(_wrong(key, value, number, "true, false or null"))
    return value


def _number(key: str, value: Any, number: int) -> float | None:
    # bool is an int in Python, and a score of `true` is a mistake worth naming rather
    # than a 1.0 nobody meant.
    if value is None or (isinstance(value, int | float) and not isinstance(value, bool)):
        return None if value is None else float(value)
    raise MalformedRecord(_wrong(key, value, number, "a number or null"))


def _labels(value: Any, number: int) -> dict[str, str]:
    """Read the environment's own keys, which are strings on both sides."""
    if not isinstance(value, dict):
        raise MalformedRecord(_wrong("labels", value, number, "an object"))
    for name, label in value.items():
        if not isinstance(label, str):
            raise MalformedRecord(_wrong(f"labels.{name}", label, number, "a string"))
    return {str(name): label for name, label in value.items()}


def _wrong(key: str, value: Any, number: int, expected: str) -> str:
    return f"line {number}: {key} must be {expected}, found {type(value).__name__}"


def _known(state: str) -> bool:
    """Whether the ranker's weight table can name this state."""
    return state in _STATES
