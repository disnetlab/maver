"""Tolerant decoding of model-submitted artifact fields.

Every helper here exists because a specific serializer failure was observed to
destroy graded work, not because tolerance is a virtue. Each is deliberately narrow:
anything it cannot repair structurally is returned untouched so that Pydantic still
reports the real validation error.

These functions are carried over from the research prototype essentially verbatim,
along with the measurements that justify them. Do not relax or generalise them
without a replacement measurement.
"""

import json
from typing import Any

# Placeholder strings a model commonly emits for an optional field when told to
# "leave it null". Without this, a literal "null" reads as a truthy halt signal and
# wrongly short-circuits the pipeline before remediation runs.
NULLISH = frozenset({"", "null", "none", "n/a", "na", "nil", "nan", "false", "no"})

# Keywords that indicate a verification check read traffic or a service level
# indicator rather than control-plane object status.
_REQUEST_PATH_EVIDENCE = (
    "curl", "wget", "probe", "http", "request", "trace", "span", "endpoint",
    "error_rate", "error rate", "errorratio", "fail_ratio", "failure rate",
    "latency", "p95", "p99", "rps", "req/s", "sli", "slo", "status code",
    "2xx", "4xx", "5xx", "locust", "wrk", "throughput", "querie",
)  # fmt: skip


def normalize_nullish(value: object) -> object:
    """Map a placeholder string to ``None``, leaving everything else untouched.

    Args:
        value: A raw field value as submitted by the model.

    Returns:
        ``None`` when the value is a nullish placeholder, otherwise the value.
    """
    if isinstance(value, str) and value.strip().lower() in NULLISH:
        return None
    return value


def is_blank_reason(value: object) -> bool:
    """Report whether a free-text reason field is missing or a placeholder.

    A hand-back signal whose reason is empty, or is the string "null" or "n/a",
    gives the next role nothing to act on, so the loop must reject it rather than
    route a contentless retry.

    Args:
        value: A raw field value as submitted by the model.

    Returns:
        True when the value cannot serve as a reason.
    """
    return not isinstance(value, str) or value.strip().lower() in NULLISH


def cites_request_path(check: object) -> bool:
    """Report whether a verification check reads traffic or an SLI, not object status.

    Deliberately a keyword screen and not a parser: the field is free text written by
    the model, so the goal is to reject an evidence set consisting solely of
    ``kubectl get pods``-style status lines, while accepting any check that actually
    names a probe, a trace, or a request, latency or error metric.

    Args:
        check: One entry from a verification result's check list.

    Returns:
        True when the check text names request-path or SLI evidence.
    """
    text = str(check or "").lower()
    return any(keyword in text for keyword in _REQUEST_PATH_EVIDENCE)


def _last_structural_closer(text: str, before: int) -> int:
    """Find the last ``}`` or ``]`` before an index that is not inside a JSON string.

    Args:
        text: The candidate JSON document.
        before: Index to search up to, exclusive.

    Returns:
        The index of the closer, or -1 when there is none.
    """
    in_string = escaped = False
    last = -1
    for index, char in enumerate(text[:before]):
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == '"':
            in_string = not in_string
        elif not in_string and char in "}]":
            last = index
    return last


def loads_tolerant(text: str, max_repairs: int = 8) -> Any:
    """Parse JSON, repairing the two ways models malform a submitted container.

    Both come from the same slip: the serializer closes a nested object one level too
    early, then keeps writing the keys that belonged inside it, producing
    ``{"params": {...}}, "justification": ...`` where the step object is already shut.
    The parser stops at the stray colon reporting a missing comma, and every retry
    reproduces it. Deleting the closer that shut the object too early and re-parsing
    recovers the step list verbatim. The same walk also handles the sibling case,
    where trailing keys spill past a complete array, because ``raw_decode`` ignores
    what follows.

    Measured on the 315 string-typed ``steps`` payloads of the verify-dose run: 284
    parsed outright, and this recovers 25 of the remaining 31 while altering none of
    the 284. Repairs are bounded and structural, only characters outside strings are
    removed, so a payload malformed some other way still raises and Pydantic reports
    it.

    Args:
        text: The candidate JSON document.
        max_repairs: Maximum number of structural deletions to attempt.

    Returns:
        The decoded value.

    Raises:
        json.JSONDecodeError: When the document cannot be repaired structurally.
    """
    for _ in range(max_repairs):
        try:
            return json.JSONDecoder().raw_decode(text)[0]
        except json.JSONDecodeError as exc:
            cut = _last_structural_closer(text, exc.pos)
            if cut < 0:
                raise
            text = text[:cut] + text[cut + 1 :]
    return json.JSONDecoder().raw_decode(text)[0]


def decode_container(value: object) -> object:
    """Accept a JSON container that arrived wrapped in a string, at any nesting level.

    Models routinely emit ``"steps": "[{...}]"`` instead of a real array. Pydantic's
    rejection of the wrong container type is silent and total: every submit attempt
    fails and the stage produces nothing, discarding work that was substantively
    correct. It cost four graded mitigations in the era-2 baseline through
    ``Plan.steps``, and reappeared on ``ExecutionResult.steps`` on the first cell of
    the verify-dose run.

    Applied to every container-typed submission field rather than case by case,
    because the failure is a property of the serializer and not of any one schema.
    Anything that does not parse is returned untouched so Pydantic still reports the
    real error.

    Args:
        value: A raw field value as submitted by the model.

    Returns:
        The decoded container, or the value unchanged when it cannot be decoded.
    """
    if isinstance(value, str):
        try:
            value = loads_tolerant(value.strip())
        except (ValueError, TypeError):
            return value
    if isinstance(value, list):
        decoded: list[object] = []
        for item in value:
            if isinstance(item, str):
                try:
                    decoded.append(loads_tolerant(item.strip()))
                    continue
                except (ValueError, TypeError):
                    pass
            decoded.append(item)
        return decoded
    return value
