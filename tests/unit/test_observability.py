"""Counting every model call, and the spans that go with them."""

import asyncio
import json
import pathlib
from types import SimpleNamespace
from typing import Any

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import SecretStr

from maver.audit.journal import AuditEvent, Journal, ListSink
from maver.llm.client import LLMClient, prompt_digest
from maver.llm.sampling import resolve_sampling
from maver.llm.types import ModelRequest, ModelResponse, ToolSpec, Usage
from maver.observability.tracing import span
from maver.observability.usage import RecordingCompletion
from maver.settings.groups import ModelSettings
from maver.settings.profiles import load_profile
from maver.settings.root import Settings

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


class _Raw:
    """One completion as the provider library hands it back."""

    def __init__(self) -> None:
        message = type("M", (), {"content": "hi", "tool_calls": None})()
        self.choices = [type("C", (), {"message": message, "finish_reason": "stop"})()]
        self.usage = None


def request(**kw: Any) -> ModelRequest:
    return ModelRequest(system="s", messages=[{"role": "user", "content": "hi"}], tools=(), **kw)


def cost_of(sink: ListSink) -> AuditEvent:
    """The completion event, found by kind: a call writes a row before it as well."""
    [event] = [e for e in sink.events if e.payload.kind == "model_call"]
    return event


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


async def test_a_call_is_recorded_before_it_is_made() -> None:
    """A process killed mid-generation writes no completion event, so the turn it died
    in left no trace at all and the run read as having stopped between turns. The row
    written first is the only durable evidence a call was in flight, and it is where the
    request is described: a completion event says what came back, never what went out.
    """
    sink = ListSink()
    recorder = RecordingCompletion(
        FakeModel(), Journal(sink, session_id="s1", stage="rca"), PROFILE, RESOLVED
    )

    await recorder.complete(request(), caller="secondpass.gate")

    started, completed = sink.events
    assert started.payload.kind == "model_call_started"
    assert completed.payload.kind == "model_call"
    assert started.payload.call_id == completed.payload.call_id
    assert started.payload.call_id is not None
    manifest = started.payload.request["inline"]
    assert manifest["capture_layer"] == "neutral_request"
    assert manifest["neutral"]["system"]["inline"] == "s"
    assert completed.payload.response["inline"]["normalised"]["text"] == "{}"
    assert started.payload.caller == "secondpass.gate"
    assert started.payload.message_count == 1
    assert started.payload.digest == prompt_digest("s", [{"role": "user", "content": "hi"}])


async def test_the_digest_says_which_messages_a_turn_was_sent() -> None:
    """It is checked against the run's own transcript, which is why it is over the
    prompt alone. A turn that sent the transcript digests to one of its prefixes; a turn
    the harness sent something else — a rescue brief, say — matches no prefix, and that
    mismatch is what sends a reader to the intervention holding what was sent."""
    transcript = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}]
    sink = ListSink()
    recorder = RecordingCompletion(
        FakeModel(), Journal(sink, session_id="s1", stage="rca"), PROFILE, RESOLVED
    )

    await recorder.complete(
        ModelRequest(system="s", messages=transcript[:1], tools=()), caller="rca"
    )

    [started] = [e for e in sink.events if e.payload.kind == "model_call_started"]
    assert started.payload.digest == prompt_digest("s", transcript[:1])
    assert started.payload.digest != prompt_digest("s", transcript)
    assert started.payload.digest != prompt_digest("elsewhere", transcript[:1])


async def test_the_recorded_sampling_is_complete_and_has_a_digest() -> None:
    """A partial record is indistinguishable from a complete one after the fact, which
    is how a change to one knob went unnoticed across weeks of runs."""
    sink = ListSink()
    recorder = RecordingCompletion(
        FakeModel(), Journal(sink, session_id="s1", stage="rca"), PROFILE, RESOLVED
    )

    await recorder.complete(request(sampling=PROFILE.aux_sampling), caller="secondpass.gate")

    sampling = cost_of(sink).payload.sampling
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
    assert cost_of(sink).payload.attempt == 2


def test_a_span_survives_an_unconfigured_deployment() -> None:
    """Instrumentation is unconditional and only export is configured, so a deployment
    with no collector runs the same code path rather than a quieter one."""
    with span("maver.test", {"a": 1, "dropped": None}) as current:
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

    sampling = cost_of(sink).payload.sampling
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

    assert cost_of(sink).payload.sampling["digest"] != RESOLVED.digest


async def test_a_call_that_failed_is_still_a_call_that_was_counted() -> None:
    """Found by running the service against an endpoint that was not there.

    The recorder wrote its completion event after the call returned, so a call that
    raised left the started row and nothing else — and the client's own sink, which does
    record an error, is not wired in the deployed composition. A run that ended on model
    failures therefore had no record of what it had cost, and a started row with nothing
    after it is supposed to mean the process died, which a terminal failure is not.
    """

    class Failing:
        async def complete(self, request: ModelRequest, *, caller: str) -> ModelResponse:
            raise RuntimeError("Connection error.")

    sink = ListSink()
    recorder = RecordingCompletion(
        Failing(), Journal(sink, session_id="s1", stage="rca"), PROFILE, RESOLVED
    )

    with pytest.raises(RuntimeError):
        await recorder.complete(request(), caller="rca")

    kinds = [e.payload.kind for e in sink.events]
    assert kinds == ["model_call_started", "model_call"]
    failed = sink.events[-1].payload
    assert failed.error is not None and "Connection error." in failed.error
    assert failed.sampling["digest"] == RESOLVED.digest


async def test_a_cancelled_call_is_counted_like_any_other() -> None:
    """The selector and the distilling pass are each bounded by a ``wait_for``, and a
    cancelled call has already burned a full prefill. Caught as ``Exception``, those two
    left a started row and nothing after it — which is how the log says the process
    died, so a routine timeout read as an OOM kill and its cost was never counted."""

    class Slow:
        async def complete(self, request: ModelRequest, *, caller: str) -> ModelResponse:
            await asyncio.sleep(10)
            raise AssertionError("the bound should have cancelled this")

    sink = ListSink()
    recorder = RecordingCompletion(
        Slow(), Journal(sink, session_id="s1", stage="rca"), PROFILE, RESOLVED
    )

    with pytest.raises(TimeoutError):
        await asyncio.wait_for(
            recorder.complete(request(), caller="secondpass.selector"), timeout=0.05
        )

    assert [e.payload.kind for e in sink.events] == ["model_call_started", "model_call"]
    assert cost_of(sink).payload.error == "cancelled"


async def test_a_repair_the_client_made_reaches_the_event() -> None:
    """A repeated tool-call identifier drops a result wherever results are matched to
    calls by identifier, and the transcript then carries an identifier the provider
    never sent. The repair is a harness intervention on the record, and the field that
    documents itself as recorded on the audit event reached no event at all."""

    class Repairing:
        async def complete(self, request: ModelRequest, *, caller: str) -> ModelResponse:
            return ModelResponse(
                text=None,
                tool_calls=(),
                finish_reason="stop",
                reasoning=None,
                usage=Usage(),
                duration_ms=1,
                attempts=1,
                notes=("synthesised duplicate tool_call_id -> call_1_get_promql",),
            )

    sink = ListSink()
    recorder = RecordingCompletion(
        Repairing(), Journal(sink, session_id="s1", stage="rca"), PROFILE, RESOLVED
    )

    await recorder.complete(request(), caller="rca")

    assert cost_of(sink).payload.notes == [
        "synthesised duplicate tool_call_id -> call_1_get_promql"
    ]


async def test_a_retried_call_records_what_the_caller_waited(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """907 ms over three attempts was recorded as 301, because the success path took
    the last attempt's duration while the failure path took the whole call. Anyone
    summing the column over an incident lost every retried attempt, and lost most of it
    exactly where the endpoint was worst."""
    monkeypatch.setattr("maver.llm.client._backoff", lambda *_: 0.0)
    attempts = 0

    async def provider(**kwargs: Any) -> Any:
        nonlocal attempts
        attempts += 1
        await asyncio.sleep(0.05)
        if attempts < 3:
            failure = ConnectionError("endpoint refused the connection")
            failure.usage = SimpleNamespace(prompt_tokens=7, completion_tokens=3)
            raise failure
        return _Raw()

    sink = ListSink()
    recorder = RecordingCompletion(
        LLMClient(PROFILE, Settings(), completion=provider),
        Journal(sink, session_id="s1", stage="rca"),
        PROFILE,
        RESOLVED,
    )

    await recorder.complete(request(), caller="rca")

    recorded = cost_of(sink).payload
    assert recorded.attempt == 3
    assert recorded.last_attempt_ms is not None
    # Two failed attempts each slept at least 50 ms before the one that succeeded, and
    # a sleep never returns early, so the whole call is at least 100 ms longer than
    # its last attempt. Comparing the two measurements as a ratio flaked whenever the
    # scheduler stretched the last attempt more than the earlier ones.
    assert recorded.duration_ms >= recorded.last_attempt_ms + 95
    attempts_recorded = [
        event.payload for event in sink.events if event.payload.kind == "model_attempt"
    ]
    assert [event.status for event in attempts_recorded] == ["failed", "failed", "succeeded"]
    assert [event.attempt for event in attempts_recorded] == [1, 2, 3]
    assert all(event.call_id == recorded.call_id for event in attempts_recorded)
    assert attempts_recorded[0].usage["prompt_tokens"] == 7
    assert attempts_recorded[0].usage["completion_tokens"] == 3
    assert recorded.usage_scope == "final_attempt"
    assert recorded.prompt_tokens is None


async def test_effective_request_response_and_cancellation_are_linked_without_credentials() -> None:
    """Schemas and provider options are evidence; credentials are not model input."""
    sink = ListSink()
    entered = asyncio.Event()
    settings = Settings(model=ModelSettings(api_key=SecretStr("audit-test-credential")))

    async def provider(**kwargs: Any) -> Any:
        assert kwargs["api_key"] == "audit-test-credential"
        assert sink.events[-1].payload.kind == "model_call_started"
        entered.set()
        await asyncio.sleep(30)

    client = LLMClient(PROFILE, settings, completion=provider)
    recorder = RecordingCompletion(
        client, Journal(sink, session_id="s1", stage="rca"), PROFILE, RESOLVED
    )
    tool = ToolSpec(
        "inspect",
        "Inspect the application",
        {"type": "object", "properties": {"namespace": {"type": "string"}}},
    )
    model_request = ModelRequest(
        system="private local prompt",
        messages=({"role": "user", "content": "investigate"},),
        tools=(tool,),
        tool_choice={"type": "function", "function": {"name": "inspect"}},
        suppress_reasoning=True,
    )
    task = asyncio.create_task(recorder.complete(model_request, caller="rca"))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    started, attempt, completed = [event.payload for event in sink.events]
    assert started.call_id == attempt.call_id == completed.call_id
    assert attempt.kind == "model_attempt" and attempt.status == "cancelled"
    assert completed.error == "cancelled"
    manifest = started.request["inline"]
    assert manifest["capture_layer"] == "litellm_invocation"
    invocation = manifest["provider"]
    assert invocation["tool_choice"] == model_request.tool_choice
    assert invocation["tools"][0]["inline"]["function"]["parameters"] == tool.parameters
    assert invocation["messages"][0]["inline"] == {
        "role": "system",
        "content": "private local prompt",
    }
    assert invocation["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    assert invocation["credentials_excluded"] == ["api_key"]
    assert "audit-test-credential" not in json.dumps(
        [event.model_dump(mode="json") for event in sink.events]
    )


async def test_concurrent_calls_keep_provider_attempts_with_their_incident() -> None:
    """The shared provider client must not capture another incident's response."""
    ready = asyncio.Event()
    arrived = 0

    async def provider(**kwargs: Any) -> Any:
        nonlocal arrived
        arrived += 1
        if arrived == 2:
            ready.set()
        await ready.wait()
        raw = _Raw()
        raw.choices[0].message.content = kwargs["messages"][-1]["content"]
        return raw

    client = LLMClient(PROFILE, Settings(), completion=provider)
    sinks = [ListSink(), ListSink()]
    recorders = [
        RecordingCompletion(
            client, Journal(sink, session_id=f"s{index}", stage="rca"), PROFILE, RESOLVED
        )
        for index, sink in enumerate(sinks)
    ]
    await asyncio.gather(
        *(
            recorder.complete(
                ModelRequest(
                    system="s", messages=({"role": "user", "content": f"incident-{index}"},)
                ),
                caller="rca",
            )
            for index, recorder in enumerate(recorders)
        )
    )
    ids = set()
    for index, sink in enumerate(sinks):
        started, attempt, completed = [event.payload for event in sink.events]
        assert started.call_id == attempt.call_id == completed.call_id
        ids.add(started.call_id)
        captured = completed.response["inline"]
        assert captured["normalised"]["text"] == f"incident-{index}"
        assert captured["provider_body"]["choices"][0]["message"]["content"] == f"incident-{index}"
        assert attempt.response == completed.response
    assert len(ids) == 2


@pytest.mark.parametrize(
    "failure",
    [
        RuntimeError("local-only secret payload"),
        asyncio.CancelledError("private cancellation context"),
    ],
)
def test_exported_spans_keep_exception_type_without_payload(monkeypatch, failure) -> None:
    """OTLP is an external export; detailed exception evidence stays in local audit."""

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(
        "maver.observability.tracing.trace.get_tracer", lambda _: provider.get_tracer("test")
    )
    with pytest.raises(type(failure)), span("maver.test"):
        raise failure
    [recorded] = exporter.get_finished_spans()
    assert recorded.status.description == type(failure).__name__
    assert recorded.events == ()
    assert str(failure) not in recorded.to_json()
    provider.shutdown()


async def test_provider_failure_keeps_details_without_the_resolved_api_key() -> None:
    """Provider exceptions sometimes echo authentication arguments into their message."""
    secret = "failure-audit-test-credential"
    settings = Settings(model=ModelSettings(api_key=SecretStr(secret)))
    sink = ListSink()

    async def provider(**kwargs):
        raise ValueError(f"request rejected using api_key={kwargs['api_key']}")

    recorder = RecordingCompletion(
        LLMClient(PROFILE, settings, completion=provider),
        Journal(sink, session_id="s1", stage="rca"),
        PROFILE,
        RESOLVED,
    )
    with pytest.raises(Exception, match="REDACTED"):
        await recorder.complete(request(), caller="rca")
    recorded = cost_of(sink).payload
    assert recorded.attempts_observed == 1
    assert recorded.error_detail["inline"]["type"] == "ValueError"
    assert "[REDACTED]" in recorded.error_detail["inline"]["message"]
    assert secret not in json.dumps([event.model_dump(mode="json") for event in sink.events])


async def test_normalisation_failure_retains_provider_usage_without_retrying() -> None:
    """A local parsing error must retain its old exception and never become a provider retry."""
    calls = 0

    async def provider(**kwargs):
        nonlocal calls
        calls += 1
        raw = _Raw()
        raw.choices = []
        raw.usage = SimpleNamespace(prompt_tokens=11, completion_tokens=4)
        return raw

    sink = ListSink()
    recorder = RecordingCompletion(
        LLMClient(PROFILE, Settings(), completion=provider),
        Journal(sink, session_id="s1", stage="rca"),
        PROFILE,
        RESOLVED,
    )
    with pytest.raises(IndexError):
        await recorder.complete(request(), caller="rca")
    assert calls == 1
    [attempt] = [event.payload for event in sink.events if event.payload.kind == "model_attempt"]
    assert attempt.error["inline"]["phase"] == "normalisation"
    assert attempt.response["inline"]["choices"] == []
    assert attempt.usage["prompt_tokens"] == 11
    assert cost_of(sink).payload.prompt_tokens == 11
    assert cost_of(sink).payload.completion_tokens == 4
