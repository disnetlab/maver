r"""Stripping NUL at the write boundary, for every backend that has one.

Shared rather than copied. A boundary with a live defect behind it is exactly the kind
that must not exist twice: two copies drift, and the one that drifts is the one nobody
re-reads. A fixture recorded against the embedded store and replayed against the server
one has to behave the same, and that is only true while both scrub the same way.

The defect is real and was paid for. ``\x00`` reaches artefacts, notes, metadata and
event payloads verbatim from tool output, and in the prototype one such byte failed a
PostgreSQL write, which made the episode skip every submission and produce a stub result
that read as an agent which never answered.
"""

import json
from typing import Any

__all__ = ["scrub", "scrub_mapping", "scrub_optional", "scrub_text"]


def scrub_text(text: str) -> str:
    """Strip NUL from a string on its way into a text column.

    A NUL is not merely ugly here. SQLite stores it but its own scalar functions treat
    TEXT as NUL-terminated, so a value containing one reports the wrong ``length()`` and
    stops matching ``LIKE`` past the byte; PostgreSQL cannot store it in ``text`` or
    ``jsonb`` at all. Removing it is a deliberate, lossy choice: one unprintable byte is
    worth less than the record it would otherwise cost.

    Args:
        text: Any string.

    Returns:
        The string without NUL.
    """
    return text.replace("\x00", "") if "\x00" in text else text


def scrub_optional(text: str | None) -> str | None:
    """Scrub a string that may be absent, leaving absence alone.

    Args:
        text: A string, or None.

    Returns:
        The scrubbed string, or None.
    """
    return None if text is None else scrub_text(text)


def scrub(value: object) -> object:
    """Scrub every string reachable inside a value.

    Args:
        value: Anything a payload can hold.

    Returns:
        The same structure with every string scrubbed. Non-strings pass through.
    """
    if isinstance(value, str):
        return scrub_text(value)
    if isinstance(value, dict):
        return {scrub(key): scrub(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [scrub(item) for item in value]
    return value


def scrub_mapping(value: dict[str, Any]) -> dict[str, Any]:
    """Render what a dict looks like once stored, for a record handed straight back.

    Through JSON rather than by scrubbing in place, because both backends store these as
    JSON and a caller comparing what it wrote against what it reads should not see the
    difference between a tuple and the list it became.

    Args:
        value: The payload as the caller supplied it.

    Returns:
        The payload as a reader will find it.
    """
    stored: dict[str, Any] = json.loads(json.dumps(scrub(value)))
    return stored
