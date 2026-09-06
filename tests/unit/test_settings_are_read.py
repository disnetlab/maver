"""Every setting either has a reader or is listed here as not having one yet.

A setting that nothing reads is a lie in the configuration surface: an operator sets
it, the process starts, and the behaviour it names does not change. Three of these have
already been found by accident rather than by looking — a corpus partition that
partitioned nothing, a model profile's endpoint and credential that were never sent to
the provider, and a tool kill switch that withheld nothing. Each one worked in the
measured deployments only because something else happened to supply the same effect.

The sweep is deliberately crude: it looks for the field name being read as an attribute
anywhere outside the module that declares it. That over-reports a field read through a
generic dump, which is why the allowlist below distinguishes the two cases and says
which is which for every entry.
"""

import pathlib
import re
from typing import get_args

from kira.settings.root import Settings

SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "kira"

READ_GENERICALLY = {
    # Dumped whole by SamplingOverrides.active(), which is the point: an override is
    # applied by name without anything naming it.
    "model.overrides.temperature",
    "model.overrides.top_p",
    "model.overrides.top_k",
    "model.overrides.min_p",
    "model.overrides.presence_penalty",
    "model.overrides.frequency_penalty",
    "model.overrides.repetition_penalty",
    "model.overrides.no_repeat_ngram_size",
    "model.overrides.reasoning_effort",
    "model.overrides.seed",
    "model.overrides.max_tokens",
}

DECLARED_FOR_UNBUILT_WORK = {
    # Each of these names a mechanism that is not built. They are kept rather than
    # deleted because the shape of the configuration is part of the design, but an
    # operator setting one today changes nothing, and that is worth failing over the
    # day someone believes otherwise.
    "intake.kafka.topic": "the queue intake is a phase-two extra",
    "intake.kafka.group_id": "the queue intake is a phase-two extra",
    "model.request_timeout_s": "the profile's own limit is what the client sends",
    "model.num_retries": "the profile's own limit is what the client sends",
    "observability.metric_interval_ms": "no metric exporter is installed",
    "observability.event_log_enabled": "the audit sink is chosen by the composition root",
    "observability.event_log_path": "the audit sink writes to the store, not a file",
    "tools.blind_namespaces": "no telemetry wrapper filters on it",
    "tools.kubectl_timeout_s": "the transport's own timeout bounds a cluster command",
}


def _paths(model: type, prefix: str = ""):
    for name, info in model.model_fields.items():
        # Into the element type as well as the field's own, because a tuple of settings
        # models exposes no model_fields of its own: the walk stopped at
        # `authority.rules` and no field of an authority rule was ever swept.
        for annotation in (info.annotation, *get_args(info.annotation)):
            if getattr(annotation, "model_fields", None):
                yield from _paths(annotation, f"{prefix}{name}.")
                break
        else:
            yield f"{prefix}{name}", name


def _readers() -> str:
    """Every module, searched for attribute access.

    A declaration reads ``name: Type = setting(...)`` and can never match ``.name``, so
    the group that declares a field is searched too — a group is allowed to read its own.
    """
    return "\n".join(
        path.read_text(encoding="utf-8")
        for path in SRC.rglob("*.py")
        if "__pycache__" not in str(path)
    )


def test_every_setting_is_read_or_declared_unread() -> None:
    text = _readers()
    unread = {
        path for path, leaf in _paths(Settings) if not re.search(rf"\.{re.escape(leaf)}\b", text)
    }

    undeclared = unread - READ_GENERICALLY - set(DECLARED_FOR_UNBUILT_WORK)
    assert not undeclared, (
        f"these settings have no reader and are not declared unread: {sorted(undeclared)}. "
        "Wire it, delete it, or add it above with the reason."
    )


def test_the_unread_list_does_not_outlive_what_it_excuses() -> None:
    """An entry that has since been wired is worse than a missing one: it says the
    mechanism is absent when it is there, and the next reader believes it."""
    text = _readers()
    stale = {
        path
        for path, leaf in _paths(Settings)
        if path in DECLARED_FOR_UNBUILT_WORK and re.search(rf"\.{re.escape(leaf)}\b", text)
    }

    assert not stale, f"these are read now and should leave the unread list: {sorted(stale)}"
