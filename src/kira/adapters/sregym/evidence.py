"""Keeping the harness's own machinery out of the evidence it serves.

This mirrors the hide list of the benchmark's Kubernetes proxy. It is benchmark
integrity rather than security: the fault injector and the load generator would
otherwise appear in what an agent reads, and turn a diagnosis into a lookup. That leak
was measured — 46 of 99 sessions saw the injector's own resources in tool output, 19 of
them without ever asking, and in one problem an agent diagnosed the injector's network
fault against a disk-sector ground truth and scored zero on every judge dimension.

The durable fix is on the harness's side, at ingestion, where the injector's series were
taken from 1641 to 0. This is the belt to that pair of braces, and it lives with the
adapter rather than in the tools because shipping it as a general rule in an operations
service would be dishonest: nothing in a real cluster is hidden from the people fixing it.
"""

import json
import re
from typing import Any

__all__ = ["EvidenceFilter"]

HIDDEN_NAMESPACES: frozenset[str] = frozenset({"chaos-mesh", "khaos"})
HIDDEN_LABELS: dict[str, frozenset[str]] = {
    "app": frozenset({"load-generator", "locust-fetcher"}),
    "job": frozenset({"workload"}),
    "opentelemetry.io/name": frozenset({"load-generator"}),
}

_SENSITIVE_TEXT_RE = re.compile(r"chaos|khaos|locust|load-generator|locust-fetcher", re.IGNORECASE)
"""Also matched against free text, because the proxy's namespace and label filtering
cannot catch an event in an application namespace whose subject is a hidden controller."""

_AUTH_ERROR_RE = re.compile(r"Error from server \((Forbidden|NotFound)\)")
_AUTH_ERROR_LIMIT = 512

_EXIT_PREFIX_RE = re.compile(r"^(exit_code=[^\n]*\n)(.*)\Z", re.DOTALL)


class EvidenceFilter:
    """The proxy's hide list, applied where a result reaches the model."""

    def filter_result(self, content: str) -> str:
        """Remove what the environment meant to hide.

        Structured output is filtered the way the proxy filters it and re-serialised,
        so it will not byte-compare against raw ``-o json``. Anything else has the
        lines naming a hidden term dropped. A permission or missing-resource error is
        kept, reduced to its first line: it is useful operational evidence and it does
        not confirm that anything was found.

        Args:
            content: A tool result.

        Returns:
            The result to hand to the model.
        """
        prefix, body = _split_exit_prefix(content)
        if _AUTH_ERROR_RE.search(body):
            first = body.splitlines()[0] if body else ""
            return prefix + first[:_AUTH_ERROR_LIMIT] + "\n"

        filtered = _filter_json_text(body.strip())
        if filtered is not None:
            return prefix + filtered + ("\n" if body.endswith("\n") else "")

        if not _SENSITIVE_TEXT_RE.search(body):
            return content
        visible = [
            line for line in body.splitlines(keepends=True) if not _SENSITIVE_TEXT_RE.search(line)
        ]
        if not visible:
            return prefix + "No resources found.\n"
        return prefix + "".join(visible)

    def filter_query(self, query: str) -> str:
        """Return the query unchanged.

        Deliberate, and carried: the measured configuration filtered results only. A
        query rewrite here would have to guess at three query languages, and the
        durable answer is the harness dropping these series at ingestion, where the
        agent cannot ask for what was never stored.

        Args:
            query: The query the model wrote.

        Returns:
            The same query.
        """
        return query


def _split_exit_prefix(output: str) -> tuple[str, str]:
    """Separate a rendered exit status from the body it introduces."""
    match = _EXIT_PREFIX_RE.match(output)
    return ("", output) if match is None else (match.group(1), match.group(2))


def _filter_json_text(text: str) -> str | None:
    """Filter a structured body, or report that it was not one."""
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    return json.dumps(_filter_objects(data), indent=2)


def _filter_objects(data: Any) -> Any:
    """Drop every hidden object from a Kubernetes API response."""
    if isinstance(data, list):
        return [_filter_objects(item) for item in data if not _is_hidden(item)]
    if not isinstance(data, dict):
        return data

    out = dict(data)
    if isinstance(out.get("items"), list):
        out["items"] = [_filter_objects(item) for item in out["items"] if not _is_hidden(item)]
    if isinstance(out.get("rows"), list):
        out["rows"] = [
            _filter_objects(row)
            for row in out["rows"]
            if not _is_hidden(row.get("object") if isinstance(row, dict) else None)
        ]
    return out


def _is_hidden(obj: Any) -> bool:
    """Report whether one object is part of the harness's own machinery."""
    if not isinstance(obj, dict):
        return False
    metadata = obj.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    return (
        metadata.get("namespace") in HIDDEN_NAMESPACES
        or _has_hidden_label(metadata)
        or _sensitive(metadata.get("name"))
        or _sensitive(obj.get("kind"))
        or _sensitive(obj.get("apiVersion"))
        or _hidden_subject(obj)
    )


def _has_hidden_label(metadata: dict[str, Any]) -> bool:
    labels = metadata.get("labels")
    if not isinstance(labels, dict):
        return False
    return any(labels.get(key) in values for key, values in HIDDEN_LABELS.items())


def _hidden_subject(obj: dict[str, Any]) -> bool:
    """Report whether an event points at something hidden, under either field name."""
    subject = obj.get("involvedObject")
    if not isinstance(subject, dict):
        subject = obj.get("regarding")
    if not isinstance(subject, dict):
        return False
    return any(
        _sensitive(subject.get(field))
        for field in ("apiVersion", "kind", "name", "namespace", "resource")
    )


def _sensitive(value: object) -> bool:
    return isinstance(value, str) and bool(_SENSITIVE_TEXT_RE.search(value))
