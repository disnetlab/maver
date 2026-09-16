"""Choose a configured pipeline once, before any agent or model call begins.

An explicit caller selection wins. Optional deployment middleware can choose only
when it is absent, and abstaining falls through to the next selector and then the
default. Checkpoints preserve the selected composition, not a rule to run again.
"""

import inspect
import json
from collections.abc import Awaitable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict

from maver.pipelines.definitions import ResolvedPipeline

__all__ = [
    "PipelineCatalog",
    "PipelineChangedError",
    "PipelineRefusal",
    "PipelineSelector",
    "SelectedPipeline",
    "restore_selection",
    "select_pipeline",
]


class PipelineRefusal(ValueError):
    """A deployment selector refused admission instead of silently selecting a default."""


class PipelineChangedError(ValueError):
    """The pinned composition is no longer available unchanged for a continuation."""


class PipelineSelector(Protocol):
    """Inspect private admission context and optionally return a configured name.

    Raising PipelineRefusal rejects admission; returning None abstains. A selector
    never runs for an explicit selection or when restoring an admitted incident.
    """

    def select(
        self, alert: Mapping[str, Any], metadata: Mapping[str, Any]
    ) -> str | Awaitable[str | None] | None:
        """Choose from configured names, with no direct access to model input."""
        ...


@dataclass(frozen=True, slots=True)
class PipelineCatalog:
    """The complete set of compositions admission is allowed to select."""

    pipelines: Mapping[str, ResolvedPipeline]
    default: str

    def __post_init__(self) -> None:
        """Copy bindings and validate the default before accepting incidents."""
        object.__setattr__(self, "pipelines", MappingProxyType(dict(self.pipelines)))
        if self.default not in self.pipelines:
            raise ValueError(f"default pipeline {self.default!r} is not configured")
        if any(name != pipeline.name for name, pipeline in self.pipelines.items()):
            raise ValueError("pipeline catalog keys must match their composition names")

    def get(self, name: str) -> ResolvedPipeline:
        """Resolve only configured names; import paths are never accepted."""
        try:
            return self.pipelines[name]
        except KeyError:
            raise ValueError(f"unknown configured pipeline {name!r}") from None


Source = Literal["explicit", "selector", "default"]


class _SavedSelection(BaseModel):
    """Validate the application-owned checkpoint payload before resolving it."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str
    source: Source
    digest: str
    composition: dict[str, Any]


@dataclass(frozen=True, slots=True)
class SelectedPipeline:
    """The admission decision and its immutable resolved composition."""

    pipeline: ResolvedPipeline
    source: Source

    def manifest(self) -> dict[str, Any]:
        """Return the complete JSON-safe selection to persist with the incident."""
        return {
            "name": self.pipeline.name,
            "source": self.source,
            "digest": self.pipeline.digest,
            "composition": self.pipeline.manifest(),
        }


def _readonly(value: Any) -> Any:
    """Give selectors recursively read-only JSON data, detached from caller objects."""
    if isinstance(value, dict):
        return MappingProxyType({key: _readonly(child) for key, child in value.items()})
    if isinstance(value, list):
        return tuple(_readonly(child) for child in value)
    return value


async def select_pipeline(
    catalog: PipelineCatalog,
    *,
    alert: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
    explicit: str | None = None,
    selectors: Sequence[PipelineSelector] = (),
) -> SelectedPipeline:
    """Apply explicit, selector, then default precedence exactly once at admission."""
    if explicit is not None:
        return SelectedPipeline(catalog.get(explicit), "explicit")
    if selectors:
        # JSON validation also prevents a live client or mutable custom object leaking
        # into the supposedly private, durable admission metadata.
        frozen_alert = _readonly(json.loads(json.dumps(dict(alert), allow_nan=False)))
        frozen_metadata = _readonly(json.loads(json.dumps(dict(metadata or {}), allow_nan=False)))
        for selector in selectors:
            selection = selector.select(frozen_alert, frozen_metadata)
            choice = await selection if inspect.isawaitable(selection) else selection
            if choice is not None:
                if not isinstance(choice, str):
                    raise TypeError("a pipeline selector must return a configured name or None")
                return SelectedPipeline(catalog.get(choice), "selector")
    return SelectedPipeline(catalog.get(catalog.default), "default")


def restore_selection(catalog: PipelineCatalog, saved: Mapping[str, Any]) -> SelectedPipeline:
    """Restore a pinned choice, refusing configuration drift instead of rerouting.

    Python workflow and agent versions belong in their manifests. Updating those
    implementations requires updating their versions; configuration and topology
    changes are detected automatically.
    """
    selection = _SavedSelection.model_validate(dict(saved))
    try:
        pipeline = catalog.get(selection.name)
    except ValueError as exc:
        raise PipelineChangedError(
            f"saved pipeline {selection.name!r} is no longer configured"
        ) from exc
    if pipeline.digest != selection.digest or pipeline.manifest() != selection.composition:
        raise PipelineChangedError(
            f"saved pipeline {selection.name!r} changed since admission; "
            "restore its original composition before resuming"
        )
    return SelectedPipeline(pipeline, selection.source)
