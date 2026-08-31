"""The parts of a role's opening message that are the same for every role.

Three small things, kept here rather than in each role, because the duplicate is the
failure this module exists to prevent. The prototype repeated the alert-stripping
comprehension in four files, one edit away from rendering the benchmark's own answer
key into a prompt, and gated the two memory blocks from two different places, which is
how a role ended up telling a model to call a retrieval tool it had not been given.
"""

import json
from collections.abc import Mapping
from typing import Any

from kira.prompts.loader import read_runtime
from kira.roles.base import RoleContext

__all__ = ["alert_json", "memory_guidance", "retrieval_enabled", "visible_alert"]


def visible_alert(alert: Mapping[str, Any]) -> dict[str, Any]:
    """Return the alert with everything the model must not be shown removed.

    Args:
        alert: The incident's alert payload.

    Returns:
        The payload without its underscore-prefixed keys.
    """
    # Strip underscore-prefixed keys, as every other prompt builder does: `_problem_id`
    # carries SREGym's ground-truth fault label, and rendering it into a prompt would
    # let the agent read the answer instead of diagnosing it.
    return {k: v for k, v in alert.items() if not k.startswith("_")}


def alert_json(alert: Mapping[str, Any]) -> str:
    """Render the alert as the JSON block a role's opening message carries.

    Args:
        alert: The incident's alert payload.

    Returns:
        Indented JSON of the visible keys.
    """
    return json.dumps(visible_alert(alert), indent=2)


def retrieval_enabled(ctx: RoleContext) -> bool:
    """Whether this role, in this run, can actually retrieve a past incident.

    A conjunction rather than the memory setting alone. Both memory blocks prescribe
    the retrieval tools by name, so emitting either for a role whose resolved toolset
    withheld them tells the model to call a tool it does not have. The toolset is
    already arm- and transport-filtered when a role runs, so this reads as written.

    Args:
        ctx: The role run's context.

    Returns:
        True when memory is on and the retrieval tool survived toolset resolution.
    """
    return ctx.settings.memory.enabled and any(
        tool.name == "memory_search" for tool in ctx.toolset.definitions
    )


def memory_guidance(enabled: bool) -> str:
    """The one-line memory block a role emits into its own opening user turn.

    This is **not** the system-prompt block of the same marker name. Two independent
    delivery paths share the spelling ``{{MEMORY_GUIDANCE}}``:

    * the system-prompt marker, filled by
      :func:`~kira.prompts.loader.compose_role_prompt` from ``blocks/memory/``;
    * this user-turn marker, filled by the single line in
      ``runtime/shared/memory_enabled.txt``.

    Wiring one and leaving the other is a recorded failure mode, so both are gated on
    the same :func:`retrieval_enabled` conjunction.

    Args:
        enabled: Whether the role can retrieve.

    Returns:
        The line with the trailing newline the template expects, or an empty string.
    """
    return read_runtime("shared", "memory_enabled.txt") + "\n" if enabled else ""
