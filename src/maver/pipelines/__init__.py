"""Python workflows, YAML compositions, and once-per-incident selection."""

from maver.pipelines.builtin import DIAGNOSE, MULTI, SINGLE, builtin_registry
from maver.pipelines.definitions import (
    PipelineConfig,
    PipelineRegistry,
    ResolvedPipeline,
    StageDefinition,
    Transition,
    WorkflowConfig,
    WorkflowDefinition,
)
from maver.pipelines.selection import (
    PipelineCatalog,
    PipelineChangedError,
    PipelineRefusal,
    PipelineSelector,
    SelectedPipeline,
    restore_selection,
    select_pipeline,
)

__all__ = [
    "DIAGNOSE",
    "MULTI",
    "SINGLE",
    "PipelineCatalog",
    "PipelineChangedError",
    "PipelineConfig",
    "PipelineRefusal",
    "PipelineRegistry",
    "PipelineSelector",
    "ResolvedPipeline",
    "SelectedPipeline",
    "StageDefinition",
    "Transition",
    "WorkflowConfig",
    "WorkflowDefinition",
    "builtin_registry",
    "restore_selection",
    "select_pipeline",
]
