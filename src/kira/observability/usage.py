"""Counting every model call, including the ones no role made.

The rule is that a model call anywhere is counted, and the reason it is a rule is that
the prototype's second-pass helper calls bypassed its usage log — so its published cost
figures had to be disclosed with that gap in them. A gate, a nominator and a selector are
model calls, and they are the ones most easily forgotten because no role issues them.

This wrapper is where the counting happens, so it happens once and in a place no caller
can route around. It carries the incident and the stage that the client cannot know: the
client is one object for the process, while an event belongs to one run.
"""

from dataclasses import dataclass

from kira.audit.journal import Journal
from kira.contracts.events import ModelCallEvent
from kira.llm.sampling import SamplingResolution, applied_sampling
from kira.llm.types import Completion, ModelRequest, ModelResponse
from kira.settings.profiles import ModelProfile

__all__ = ["RecordingCompletion"]


@dataclass(frozen=True, slots=True)
class RecordingCompletion:
    """Passes a call through and writes what it cost to the incident's log."""

    inner: Completion
    journal: Journal
    profile: ModelProfile
    sampling: SamplingResolution
    """The agent's set as resolved against settings, not as written in the profile.

    Recording ``profile.sampling`` here meant an overridden run logged the file's value
    and a digest identical to an unoverridden one. That is the drift this event exists
    to make visible, reproduced inside the mechanism built to catch it.
    """

    async def complete(self, request: ModelRequest, *, caller: str) -> ModelResponse:
        """Run one completion and record it.

        Args:
            request: The call.
            caller: What issued it — a role, or a mechanism such as a quality gate.

        Returns:
            The response.
        """
        response = await self.inner.complete(request, caller=caller)
        self.journal.record(
            ModelCallEvent(
                caller=caller,
                model=self.profile.model,
                profile=self.profile.name,
                sampling=applied_sampling(
                    request.sampling, self.profile, self.sampling
                ).as_record(),
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                reasoning_tokens=response.usage.reasoning_tokens,
                cached_tokens=response.usage.cached_prompt_tokens,
                finish_reason=response.finish_reason,
                duration_ms=response.duration_ms,
                # One event per completed call, and the attempt count says how many
                # tries it took. The client's own sink is where a deployment that wants
                # per-attempt detail gets it; recording both here would double-count.
                attempt=response.attempts,
            )
        )
        return response
