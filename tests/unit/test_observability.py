"""Counting every model call, and the spans that go with them."""

import pathlib
from typing import Any

from kira.audit.journal import Journal, ListSink
from kira.llm.sampling import resolve_sampling
from kira.llm.types import ModelRequest, ModelResponse, Usage
from kira.observability.tracing import span
from kira.observability.usage import RecordingCompletion
from kira.settings.profiles import load_profile
from kira.settings.root import Settings

PROFILE = load_profile("qwen3.6-35b-a3b", pathlib.Path("configs/models"))
RESOLVED = resolve_sampling(PROFILE, Settings())


class FakeModel:
    async def complete(self, request: ModelRequest, *, caller: str) -> ModelResponse:
        return ModelResponse(
            text="{}",
            tool_calls=(),
            finish_reason="stop",
            reasoning=None,
            usage=Usage(prompt_tokens=120, completion_tokens=40),
            duration_ms=7,
            attempts=2,
        )


def request(**kw: Any) -> ModelRequest:
    return ModelRequest(system="s", messages=[{"role": "user", "content": "hi"}], tools=(), **kw)


async def test_a_helper_call_is_counted_like_a_role_s() -> None:
    """A gate, a nominator and a selector are model calls, and they are the ones most
    easily forgotten because no role issues them. The prototype's bypassed its usage
    log, and its published cost figures had to be disclosed with that gap in them."""
    sink = ListSink()
    recorder = RecordingCompletion(
        FakeModel(), Journal(sink, session_id="s1", stage="rca"), PROFILE, RESOLVED
    )

    await recorder.complete(request(), caller="secondpass.gate")

    [event] = [e for e in sink.events if e.payload.kind == "model_call"]
    assert event.payload.caller == "secondpass.gate"
    assert event.payload.prompt_tokens == 120
    assert event.session_id == "s1"


async def test_the_recorded_sampling_is_complete_and_has_a_digest() -> None:
    """A partial record is indistinguishable from a complete one after the fact, which
    is how a change to one knob went unnoticed across weeks of runs."""
    sink = ListSink()
    recorder = RecordingCompletion(
        FakeModel(), Journal(sink, session_id="s1", stage="rca"), PROFILE, RESOLVED
    )

    await recorder.complete(request(sampling=PROFILE.aux_sampling), caller="secondpass.gate")

    sampling = sink.events[0].payload.sampling
    assert sampling["temperature"] == 0.0
    assert set(PROFILE.sampling.model_dump()) <= set(sampling)
    assert len(sampling["digest"]) == 12


async def test_one_event_per_completed_call_not_per_attempt() -> None:
    """Recording both here and in the client's own per-attempt sink would double-count.
    The attempt number is how many tries it took."""
    sink = ListSink()
    recorder = RecordingCompletion(
        FakeModel(), Journal(sink, session_id="s1", stage="rca"), PROFILE, RESOLVED
    )

    await recorder.complete(request(), caller="rca")

    assert len([e for e in sink.events if e.payload.kind == "model_call"]) == 1
    assert sink.events[0].payload.attempt == 2


def test_a_span_survives_an_unconfigured_deployment() -> None:
    """Instrumentation is unconditional and only export is configured, so a deployment
    with no collector runs the same code path rather than a quieter one."""
    with span("kira.test", {"a": 1, "dropped": None}) as current:
        assert current is not None


async def test_an_overridden_run_is_not_recorded_as_the_profile_it_overrode() -> None:
    """The July temperature flip, reproduced inside the mechanism built to catch it.

    A run script exporting ``LLM_TEMPERATURE`` displaces the profile's set, and the
    recorder read the profile file instead of the resolved set: the event named the
    YAML's temperature and carried a digest identical to an unoverridden run's. Two
    campaigns that differed in the one variable that matters were, in the record that
    exists to tell them apart, the same campaign.
    """
    sink = ListSink()
    overridden = resolve_sampling(
        PROFILE, Settings.model_validate({"model": {"overrides": {"temperature": 0.0}}})
    )
    recorder = RecordingCompletion(
        FakeModel(), Journal(sink, session_id="s1", stage="rca"), PROFILE, overridden
    )

    await recorder.complete(request(), caller="rca")

    sampling = sink.events[0].payload.sampling
    assert sampling["temperature"] == 0.0
    assert sampling["overrides"] == ["temperature"]
    assert sampling["digest"] != RESOLVED.digest


async def test_a_call_that_brought_its_own_set_is_recorded_on_that_set() -> None:
    """The other direction of the same rule: an auxiliary call runs deterministically
    on its own set precisely so it does not run on the agent's, and recording the
    agent's would say it did."""
    sink = ListSink()
    recorder = RecordingCompletion(
        FakeModel(), Journal(sink, session_id="s1", stage="rca"), PROFILE, RESOLVED
    )

    await recorder.complete(request(sampling=PROFILE.aux_sampling), caller="secondpass.gate")

    assert sink.events[0].payload.sampling["digest"] != RESOLVED.digest
