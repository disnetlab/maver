"""The AIOpsLab environment adapter.

Installed with no extra of its own. The bridge is the standard library on the driver's
side and ``httpx`` on ours, and AIOpsLab itself is an operator-supplied checkout rather
than a package, so it is never declared as a dependency and the driver imports it inside
the function that needs it.

This module is deliberately import-light. The entry point that registers it is present in
every install, so nothing here may import the bridge client; that import happens when a
deployment actually enables the adapter, where a missing piece is a startup failure
naming what to fix.
"""

from kira.adapters.aiopslab.adapter import AiopsLabAdapter
from kira.adapters.aiopslab.config import ADAPTER_NAME, AiopsLabConfig
from kira.adapters.base import EnvironmentAdapter
from kira.adapters.registry import adapter as register
from kira.settings.root import Settings

__all__ = ["ADAPTER", "ADAPTER_NAME", "AiopsLabAdapter", "AiopsLabConfig"]


# Renamed on import so that the decorator does not shadow this package's own
# `adapter` submodule, which the entry point's target lives in.
@register(ADAPTER_NAME)
def _factory(settings: Settings) -> EnvironmentAdapter:
    """Build the process-wide adapter from the deployment's settings.

    No legacy switch. The prototype's own switch was the presence of a bridge URL and a
    token rather than a boolean, and reading a URL as a truthy flag would turn a
    half-configured deployment into an enabled one.
    """
    return AiopsLabAdapter(AiopsLabConfig.resolve(settings))


ADAPTER = _factory
"""What the entry point names.

Importing this module is what registers the adapter; naming the factory rather than the
module makes what the entry point resolves to explicit, and gives a caller holding the
entry point something to call.
"""
