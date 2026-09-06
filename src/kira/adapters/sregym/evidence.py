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

import ast
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

_WALK_DEPTH = 12
"""How deep the schema-free search goes. A trace is nested; nothing useful is deeper."""

_EXIT_PREFIX_RE = re.compile(r"^(exit_code=[^\n]*\n)(.*)\Z", re.DOTALL)

_SUPPRESSED = "Result withheld. This is not a finding that nothing was there."
"""What replaces a result the filter removed in its entirety.

"No resources found." was a lie a model acts on, and it carries no ``error:`` prefix, so
the loop, the repeat breaker and the audit log all recorded a successful read of an
empty cluster."""


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
            return prefix + _SUPPRESSED + "\n"
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
    """Filter a structured body, or report that it was not one.

    Two parsers, as every other reader of these payloads has: neither transport emits
    JSON, both render structured telemetry as the Python repr of a list of dictionaries,
    and a repr is one line — so a single hidden series fell through to the line-based
    branch, lost that one line, and took the whole result with it.

    Re-serialised in the encoding it arrived in, because the alert partitioning
    downstream reads a repr with :func:`ast.literal_eval`, which a JSON rendering
    carrying ``true`` or ``null`` would not parse.
    """
    for parse in (json.loads, ast.literal_eval):
        try:
            data = parse(text)
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            continue
        filtered = _filter_objects(data)
        if filtered == _SUPPRESSED:
            # Bare, rather than a JSON string wrapped in quotes: the whole response was
            # the hidden object, so there is no structure left to render.
            return _SUPPRESSED
        return json.dumps(filtered, indent=2) if parse is json.loads else repr(filtered)
    return None


def _filter_objects(data: Any) -> Any:
    """Drop every hidden object from a structured response.

    Every rule below except the last reads a Kubernetes shape — ``items``, ``rows``,
    ``metadata.namespace``, a label, an event's subject. That was the whole of it, and it
    meant a body this could not recognise was re-serialised unchanged: a Prometheus
    result is a list under ``data.result`` whose namespace lives in ``metric``, and a
    Jaeger response is traces and spans, so neither lost anything. Worse, being valid
    JSON is what selected this filter over the line-based one, so an unrecognised shape
    got the filter that did nothing instead of the filter that would have worked.

    So the last rule is shape-free: an element of any list that names the harness's own
    machinery anywhere inside it is dropped. What is hidden is a set of names — the
    injector, the load generator — not a set of schemas, and a filter that has to know
    the schema fails silently on the first one nobody thought of.
    """
    if isinstance(data, list):
        return [_filter_objects(item) for item in data if not _is_hidden(item)]
    if not isinstance(data, dict):
        return data
    if _is_hidden_resource(data):
        # The object a response *is*, and not only the ones it holds. A single-resource
        # `-o json` read is a bare object, so the list rule never sees it, and being
        # valid JSON is what selected this filter — `get deploy load-generator -o json`
        # and `get networkchaos <name> -o json` came back with the injection verbatim.
        return _SUPPRESSED

    # Every value, not two named keys. Descending only into `items` and `rows` is what
    # confined this to Kubernetes responses: a Prometheus result hangs under `data`, and
    # nothing walked far enough to find it.
    out = {key: _filter_objects(value) for key, value in data.items()}
    if isinstance(out.get("rows"), list):
        # A table row wraps its object rather than being one.
        out["rows"] = [
            row
            for row in out["rows"]
            if not _is_hidden(row.get("object") if isinstance(row, dict) else None)
        ]
    return out


def _is_hidden(obj: Any) -> bool:
    """Report whether one object is part of the harness's own machinery."""
    if not isinstance(obj, dict):
        return False
    return _is_hidden_resource(obj) or _names_hidden_machinery(obj)


def _is_hidden_resource(obj: Any) -> bool:
    """Report whether an object is machinery by its own identity.

    Separate from :func:`_is_hidden` because the containment walk cannot be asked about
    a whole response: a Prometheus envelope holding one injector series names the
    machinery too, and dropping the body over it would destroy the incident's own
    telemetry. Dropping one element of a list costs one element; suppressing the object
    a response *is* costs the answer.
    """
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


def _names_hidden_machinery(value: Any, depth: int = 0) -> bool:
    """Report whether a value names the harness's machinery anywhere inside it.

    The schema-free rule, and the reason it is bounded: a response is arbitrary JSON from
    a service this code does not control, so the walk is depth-limited rather than
    trusted to terminate on shape.
    """
    if depth > _WALK_DEPTH:
        return False
    if isinstance(value, str):
        return bool(_SENSITIVE_TEXT_RE.search(value))
    if isinstance(value, dict):
        return any(
            _names_hidden_machinery(key, depth + 1) or _names_hidden_machinery(item, depth + 1)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_names_hidden_machinery(item, depth + 1) for item in value)
    return False


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
