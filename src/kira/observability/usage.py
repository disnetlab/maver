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
from kira.llm.sampling import sampling_digest
from kira.llm.types import Completion, ModelRequest, ModelResponse
from kira.settings.profiles import ModelProfile, SamplingSet

__all__ = ["RecordingCompletion"]


@dataclass(frozen=True, slots=True)
class RecordingCompletion:
    """Passes a call through and writes what it cost to the incident's log."""

    inner: Completion
    journal: Journal
    profile: ModelProfile

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
                sampling=_record(request.sampling or self.profile.sampling),
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


def _record(sampling: SamplingSet) -> dict[str, object]:
    """Render a sampling set for the audit event.

    The complete set and its digest together. A partial record is indistinguishable
    from a complete one after the fact, which is how a change to one knob went
    unnoticed across weeks of runs whose results were no longer comparable.
    """
    values = sampling.model_dump(mode="json")
    return {**values, "digest": sampling_digest(values)}
