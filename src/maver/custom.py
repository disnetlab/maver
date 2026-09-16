"""The local Python registration point for a cloned MAVER deployment.

Import your subclasses, middleware factories and workflows here and register them
in the functions below. YAML selects only the names registered by this code.
See ``docs/customisation.md`` for a complete example.
"""

from maver.agents import AgentRegistry
from maver.pipelines import PipelineRegistry, PipelineSelector


def register_agents(registry: AgentRegistry) -> None:
    """Register local Agent subclasses and typed middleware factories here."""


def register_pipelines(registry: PipelineRegistry) -> None:
    """Register local WorkflowDefinition objects here."""


def pipeline_selectors() -> tuple[PipelineSelector, ...]:
    """Return optional admission selectors; explicit incident selection wins."""
    return ()
