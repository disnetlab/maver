"""Registering what runs after an incident.

One hook, and the reason there is only one is worth stating: the remediating role
verifies its own change before reporting success, and the verification role independently
checks recovery, so there is no separate asynchronous verification pass to schedule. The
distilling pass is the only post-incident work, and it runs on every terminal outcome.

Including the halted ones. A run that stopped deliberately, with a stated reason, is one
of the more informative things a corpus can hold — it records a judgement that the
system should not act, which is exactly what a later incident facing the same evidence
needs to know was already considered.
"""

import logging

from maver.orchestration.hooks import PostIncidentHook, register_hook, registered_hooks
from maver.orchestration.learning import distil_incident
from maver.orchestration.outcomes import Outcome
from maver.settings.root import Settings

__all__ = ["register_post_incident_hooks"]

_log = logging.getLogger(__name__)


def register_post_incident_hooks(settings: Settings) -> None:
    """Register the hooks this configuration wants, once.

    Args:
        settings: The deployment's settings.
    """
    if not settings.memory.learning_enabled or "learning" in registered_hooks():
        return
    register_hook(
        PostIncidentHook(
            name="learning",
            job_fn=distil_incident,
            on_outcomes=frozenset(Outcome),
            delay_seconds=settings.memory.learning_delay_s,
        )
    )
    _log.info(
        "distillation will run %d seconds after an incident closes",
        settings.memory.learning_delay_s,
    )
