"""The provider client, and one regression test per compensated provider quirk.

Every workaround here exists because of measured behaviour. A workaround with no test is
a comment waiting to be deleted by someone who does not believe it.
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import pytest

from maver.contracts.events import ModelCallEvent
from maver.llm import (
    ErrorClass,
    LLMClient,
    ModelCallError,
    ModelRequest,
    ToolCall,
    ToolResult,
    classify,
    extract_reasoning_text,
    is_token_limit_finish,
    resolve_sampling,
    serialise_assistant,
    serialise_tool_results,
)
from maver.llm.client import _BACKOFF_CAP_S, _backoff, carry_reasoning_forward
from maver.llm.sampling import assert_sampling_is_pinned
from maver.llm.types import ModelResponse, ToolSpec, Usage
from maver.settings import load_profile, load_settings

MODELS = Path("configs/models")


@pytest.fixture(autouse=True)
def _no_real_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the retry tests deterministic and fast.

    The backoff schedule itself is unit-tested separately; sleeping through it here
    would make the suite slow and its duration random.
    """
    monkeypatch.setattr("maver.llm.client._backoff", lambda *_: 0.0)


class _Fn:
    def __init__(self, name: str, arguments: Any) -> None:
        self.name = name
        self.arguments = arguments


class _Call:
    def __init__(self, call_id: Any, name: str, arguments: Any) -> None:
        self.id = call_id
        self.function = _Fn(name, arguments)


class _Msg:
    def __init__(self, content: Any = None, tool_calls: Any = None, **extra: Any) -> None:
        self.content = content
        self.tool_calls = tool_calls
        for key, value in extra.items():
            setattr(self, key, value)


class _Choice:
    def __init__(self, message: _Msg, finish_reason: str | None) -> None:
        self.message = message
        self.finish_reason = finish_reason


class _Raw:
    def __init__(
        self, message: _Msg, finish_reason: str | None = "stop", usage: Any = None
    ) -> None:
        self.choices = [_Choice(message, finish_reason)]
        self.usage = usage


def _client(profile: str = "qwen3.6-35b-a3b", **kw: Any) -> tuple[LLMClient, list[ModelCallEvent]]:
    events: list[ModelCallEvent] = []
    client = LLMClient(load_profile(profile, MODELS), load_settings(), sink=events.append, **kw)
    return client, events


async def _complete(client: LLMClient, caller: str = "rca") -> Any:
    return await client.complete(ModelRequest(system="s", messages=()), caller=caller)


class TestSampling:
    def test_the_whole_set_is_resolved_and_digested(self) -> None:
        """A partial record is indistinguishable from a complete one after the fact."""
        profile = load_profile("qwen3.6-35b-a3b", MODELS)
        resolution = resolve_sampling(profile, load_settings())
        assert set(resolution.values) == set(profile.sampling.model_dump())
        assert len(resolution.digest) == 12
        assert resolution.as_record()["profile"] == profile.name

    def test_an_override_is_recorded_rather_than_absorbed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One knob moved and unrecorded made months of runs incomparable."""
        monkeypatch.setenv("LLM_TEMPERATURE", "0.0")
        resolution = resolve_sampling(load_profile("qwen3.6-35b-a3b", MODELS), load_settings())
        assert resolution.values["temperature"] == 0.0
        assert resolution.overrides == {"temperature": 0.0}
        assert resolution.as_record()["overrides"] == ["temperature"]

    def test_the_digest_changes_when_any_knob_changes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        profile = load_profile("qwen3.6-35b-a3b", MODELS)
        before = resolve_sampling(profile, load_settings()).digest
        monkeypatch.setenv("LLM_TOP_P", "0.5")
        assert resolve_sampling(profile, load_settings()).digest != before

    def test_native_knobs_are_routed_to_the_extension(self) -> None:
        """Q6: they have no field in the schema a hosted route accepts."""
        resolution = resolve_sampling(load_profile("qwen3.6-35b-a3b", MODELS), load_settings())
        assert set(resolution.extra_body) == {"top_k", "min_p", "repetition_penalty"}
        assert "top_k" not in resolution.top_level

    def test_the_startup_assertion_accepts_a_shipped_profile(self) -> None:
        for name in ("qwen3.6-35b-a3b", "claude-sonnet-4-6"):
            profile = load_profile(name, MODELS)
            assert_sampling_is_pinned(profile, resolve_sampling(profile, load_settings()))

    def test_a_native_knob_bound_for_a_hosted_route_fails_at_startup(self) -> None:
        """Q7: dropping unsupported parameters does not police the native extension."""
        profile = load_profile("claude-sonnet-4-6", MODELS)
        resolution = resolve_sampling(profile, load_settings())
        resolution.extra_body["top_k"] = 20
        with pytest.raises(ValueError, match="no native extension fields"):
            assert_sampling_is_pinned(profile, resolution)


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        ("length", True),
        ("max_tokens", True),
        ("MAX_TOKENS_EXCEEDED", True),
        ("stop", False),
        ("tool_calls", False),
        (None, False),
    ],
)
def test_token_limit_finish_detection(reason: str | None, expected: bool) -> None:
    """Defined once. The prototype had it byte-identical in two modules."""
    assert is_token_limit_finish(reason) is expected


class TestToolCallNormalisation:
    async def test_arguments_arrive_as_a_json_string(self) -> None:
        client, _ = _client(
            completion=_returns(_Raw(_Msg(tool_calls=[_Call("a", "t", '{"x": 1}')])))
        )
        response = await _complete(client)
        assert response.tool_calls[0].arguments == {"x": 1}

    async def test_a_zero_argument_call_is_legal(self) -> None:
        for empty in (None, ""):
            client, _ = _client(
                completion=_returns(_Raw(_Msg(tool_calls=[_Call("a", "t", empty)])))
            )
            response = await _complete(client)
            assert response.tool_calls[0].arguments == {}
            assert response.tool_calls[0].parse_error is None

    async def test_already_decoded_arguments_pass_through(self) -> None:
        """Some provider routes pre-decode them."""
        client, _ = _client(completion=_returns(_Raw(_Msg(tool_calls=[_Call("a", "t", {"x": 1})]))))
        assert (await _complete(client)).tool_calls[0].arguments == {"x": 1}

    async def test_malformed_arguments_become_a_parse_error_not_an_exception(self) -> None:
        """The model can be told what was wrong and try again; a raise ends the run."""
        client, _ = _client(completion=_returns(_Raw(_Msg(tool_calls=[_Call("a", "t", "{oops")]))))
        call = (await _complete(client)).tool_calls[0]
        assert call.parse_error and "invalid JSON" in call.parse_error

    async def test_a_json_scalar_is_rejected_as_a_parse_error(self) -> None:
        client, _ = _client(completion=_returns(_Raw(_Msg(tool_calls=[_Call("a", "t", "42")]))))
        call = (await _complete(client)).tool_calls[0]
        assert call.parse_error == "arguments JSON must decode to an object"

    async def test_missing_and_duplicate_ids_are_normalised(self) -> None:
        """Q12: a null identifier writes a null into the transcript, and a repeated one
        silently drops a result wherever results are matched to calls by identifier."""
        client, _ = _client(
            completion=_returns(
                _Raw(
                    _Msg(
                        tool_calls=[
                            _Call(None, "a", "{}"),
                            _Call("x", "b", "{}"),
                            _Call("x", "c", "{}"),
                        ]
                    )
                )
            )
        )
        response = await _complete(client)
        ids = [c.id for c in response.tool_calls]
        assert len(set(ids)) == 3
        assert all(ids)
        assert response.notes == (
            "synthesised missing tool_call_id -> call_0_a",
            "synthesised duplicate tool_call_id -> call_2_c",
        )
        assert any("synthesised" in note for note in response.notes)


class TestResponseRepair:
    def test_a_truncated_tool_call_is_dropped_and_noted(self) -> None:
        """Q1: echoing incomplete argument JSON makes the provider reject the NEXT
        request, and a result for a dropped call would dangle."""
        response = ModelResponse(
            text="partial",
            tool_calls=(ToolCall(id="a", name="t", arguments={}, raw_arguments='{"x": '),),
            finish_reason="length",
            reasoning=None,
            usage=Usage(),
            duration_ms=1,
            attempts=1,
        )
        message = serialise_assistant(response)
        assert "tool_calls" not in message
        assert "cut off" in message["content"]

    def test_a_reasoning_only_turn_carries_its_reasoning_as_content(self) -> None:
        """Q2: otherwise the reasoning is dropped and the no-tool nudge loop stalls."""
        message = serialise_assistant(
            ModelResponse(
                text=None,
                tool_calls=(),
                finish_reason="length",
                reasoning="I was working out that the Service selector is wrong",
                usage=Usage(),
                duration_ms=1,
                attempts=1,
            )
        )
        assert "Service selector" in message["content"]

    def test_an_empty_truncated_turn_gets_an_explanatory_note(self) -> None:
        message = serialise_assistant(
            ModelResponse(
                text=None,
                tool_calls=(),
                finish_reason="length",
                reasoning=None,
                usage=Usage(),
                duration_ms=1,
                attempts=1,
            )
        )
        assert "truncated" in message["content"].lower()

    def test_a_parse_errored_call_keeps_its_place_with_sanitised_arguments(self) -> None:
        """Its error result must still resolve, but a strict server would reject the
        next request over the malformed string."""
        message = serialise_assistant(
            ModelResponse(
                text=None,
                tool_calls=(
                    ToolCall(
                        id="a",
                        name="t",
                        arguments={},
                        raw_arguments="{oops",
                        parse_error="invalid JSON arguments: x",
                    ),
                ),
                finish_reason="tool_calls",
                reasoning=None,
                usage=Usage(),
                duration_ms=1,
                attempts=1,
            )
        )
        assert message["tool_calls"][0]["function"]["arguments"] == "{}"

    def test_a_good_call_is_echoed_verbatim(self) -> None:
        """The next request carries what the provider produced, not a re-encoding."""
        raw = '{"query":  "up"}'
        message = serialise_assistant(
            ModelResponse(
                text=None,
                tool_calls=(
                    ToolCall(id="a", name="t", arguments={"query": "up"}, raw_arguments=raw),
                ),
                finish_reason="tool_calls",
                reasoning=None,
                usage=Usage(),
                duration_ms=1,
                attempts=1,
            )
        )
        assert message["tool_calls"][0]["function"]["arguments"] == raw


class TestReasoningExtraction:
    def test_covers_three_field_shapes(self) -> None:
        """Q3: a flat string, and two different names for structured blocks."""
        assert extract_reasoning_text(_Msg(reasoning_content="flat")) == "flat"
        assert extract_reasoning_text(_Msg(thinking_blocks=[{"thinking": "block"}])) == "block"
        assert extract_reasoning_text(_Msg(reasoning_items=[{"text": "item"}])) == "item"
        assert extract_reasoning_text(_Msg()) == ""

    def test_carry_forward_keeps_the_tail(self) -> None:
        """The useful part of a reasoning trace is where it ended up."""
        message = carry_reasoning_forward(
            {"role": "assistant", "content": ""}, "a" * 50 + "END", 10
        )
        assert message["content"].endswith("END")
        assert message["content"].startswith("…")


class TestUsageAccounting:
    async def test_an_unreported_cached_count_stays_null(self) -> None:
        """Q8: one endpoint runs prefix caching without reporting it, so a zero here
        would be an invented measurement."""
        usage = type("U", (), {"prompt_tokens": 100, "completion_tokens": 10})()
        client, events = _client(completion=_returns(_Raw(_Msg("hi"), usage=usage)))
        response = await _complete(client)
        assert response.usage.cached_prompt_tokens is None
        assert response.usage.uncached_prompt_tokens is None
        assert events[0].cached_tokens is None

    async def test_uncached_input_is_derived_only_when_the_cached_subset_is_known(self) -> None:
        details = type("D", (), {"cached_tokens": 40})()
        usage = type(
            "U",
            (),
            {"prompt_tokens": 100, "completion_tokens": 10, "prompt_tokens_details": details},
        )()
        client, _ = _client(completion=_returns(_Raw(_Msg("hi"), usage=usage)))
        response = await _complete(client)
        assert response.usage.cached_prompt_tokens == 40
        assert response.usage.uncached_prompt_tokens == 60

    async def test_every_call_names_its_caller(self) -> None:
        """The prototype counted only role-issued calls and disclosed the gap."""
        client, events = _client(completion=_returns(_Raw(_Msg("hi"))))
        await _complete(client, caller="quality_gate")
        assert events[0].caller == "quality_gate"
        assert events[0].profile == "qwen3.6-35b-a3b"
        assert events[0].sampling["digest"]

    async def test_a_failed_call_still_writes_an_event(self) -> None:
        """A call that burned a prefill before being rejected recorded nothing."""
        client, events = _client(completion=_raises(_status_error(400, "bad parameter")))
        with pytest.raises(ModelCallError):
            await _complete(client)
        assert len(events) == 1
        assert events[0].error and "bad parameter" in events[0].error

    async def test_each_retry_attempt_is_its_own_event(self) -> None:
        """With the library retrying internally, latency conflates attempts and the
        attempt count is lost."""
        client, events = _client(completion=_raises(_status_error(503, "unavailable")))
        with pytest.raises(ModelCallError) as excinfo:
            await _complete(client)
        assert excinfo.value.attempts == 4  # one call plus three retries
        assert [e.attempt for e in events] == [1, 2, 3, 4]


class TestProviderPacing:
    async def test_profile_interval_serialises_provider_requests(self) -> None:
        """A sequential incident can still exceed Google's free-tier RPM cap."""
        profile = load_profile("gemini-3.5-flash-lite", MODELS)
        profile = profile.model_copy(
            update={"limits": {**profile.limits, "min_request_interval_ms": 40}}
        )
        starts: list[float] = []
        active = 0
        peak = 0

        async def completion(**_kwargs: Any) -> Any:
            nonlocal active, peak
            starts.append(time.monotonic())
            active += 1
            peak = max(peak, active)
            try:
                await asyncio.sleep(0.02)
                return _Raw(_Msg("ok"))
            finally:
                active -= 1

        client = LLMClient(profile, load_settings(), completion=completion)
        await asyncio.gather(_complete(client), _complete(client))
        assert peak == 1
        assert len(starts) == 2
        assert starts[1] - starts[0] >= 0.035


class TestRequestAssembly:
    async def test_a_tool_choice_is_never_sent_without_tools(self) -> None:
        captured: dict[str, Any] = {}
        client, _ = _client(completion=_capture(captured, _Raw(_Msg("hi"))))
        await client.complete(
            ModelRequest(system="s", messages=(), tool_choice="required"), caller="rca"
        )
        assert "tool_choice" not in captured["kwargs"]

    async def test_the_profile_supplies_the_tool_choice_mode(self) -> None:
        """Q4/Q10: forcing a choice returns zero parsed calls on one parser, and some
        thinking endpoints reject an explicit choice outright."""
        captured: dict[str, Any] = {}
        client, _ = _client(completion=_capture(captured, _Raw(_Msg("hi"))))
        await client.complete(
            ModelRequest(system="s", messages=(), tools=(ToolSpec("t", "d", {}),)), caller="rca"
        )
        # The mechanism is under test, not the shipped value, which describes an era.
        assert (
            captured["kwargs"]["tool_choice"]
            == load_profile("qwen3.6-35b-a3b", MODELS).quirks.tool_choice
        )

    async def test_the_retry_loop_is_ours(self) -> None:
        captured: dict[str, Any] = {}
        client, _ = _client(completion=_capture(captured, _Raw(_Msg("hi"))))
        await _complete(client)
        assert captured["kwargs"]["num_retries"] == 0

    async def test_a_rescue_turn_applies_the_rescue_quirks(self) -> None:
        """Q11: a greedy local model otherwise spends the whole budget reasoning."""
        captured: dict[str, Any] = {}
        client, _ = _client("claude-sonnet-4-6", completion=_capture(captured, _Raw(_Msg("hi"))))
        await client.complete(
            ModelRequest(system="s", messages=(), lite_reasoning=True), caller="rca"
        )
        assert captured["kwargs"]["reasoning_effort"] == "low"

    async def test_a_helper_asks_for_no_reasoning_channel_on_any_profile(self) -> None:
        """The second-pass helpers each want one JSON object under a few hundred tokens.
        Left with a reasoning channel, this model spends the whole budget in it and
        returns empty content, which the gate reads as unparseable and the selector as a
        parse failure — so both fall back and the arm's mechanism never runs. Every
        measured run made these calls with the channel off, and it is off here whatever
        the profile says about rescue turns."""
        captured: dict[str, Any] = {}
        client, _ = _client(completion=_capture(captured, _Raw(_Msg("hi"))))
        await client.complete(
            ModelRequest(system="s", messages=(), suppress_reasoning=True),
            caller="secondpass.gate",
        )
        assert captured["kwargs"]["extra_body"]["chat_template_kwargs"] == {
            "enable_thinking": False
        }
        assert "reasoning_effort" not in captured["kwargs"]

    async def test_a_rescue_turn_is_recorded_on_the_set_it_was_sent_on(self) -> None:
        """The rescue levers are applied after the rule that resolves the set, so a
        record derived from the rule alone said the turn ran on the profile's reasoning
        effort — and carried the same digest as an ordinary turn, so the one field that
        exists to tell two configurations apart could not. The rescue is the turn that
        produces the graded submission in the cells it fires on."""
        captured: dict[str, Any] = {}
        client, events = _client(
            "claude-sonnet-4-6", completion=_capture(captured, _Raw(_Msg("hi")))
        )

        await client.complete(
            ModelRequest(system="s", messages=(), lite_reasoning=True), caller="rca"
        )
        rescue, sent = events[-1].sampling, captured["kwargs"]["reasoning_effort"]
        await _complete(client)
        ordinary = events[-1].sampling

        assert rescue["reasoning_effort"] == sent == "low"
        assert rescue["digest"] != ordinary["digest"]


def test_backoff_grows_and_is_capped() -> None:
    """Matches what the library did internally, so a run's timing profile is unchanged."""
    for attempt in range(1, 12):
        for cls in (ErrorClass.TRANSIENT, ErrorClass.RATE_LIMIT):
            assert 0.0 <= _backoff(attempt, cls) <= _BACKOFF_CAP_S


class TestErrorClassification:
    @pytest.mark.parametrize(
        ("status", "expected"),
        [
            (429, ErrorClass.RATE_LIMIT),
            (401, ErrorClass.AUTH),
            (403, ErrorClass.AUTH),
            (400, ErrorClass.BAD_REQUEST),
            (500, ErrorClass.TRANSIENT),
            (503, ErrorClass.TRANSIENT),
        ],
    )
    def test_status_codes_classify(self, status: int, expected: ErrorClass) -> None:
        assert classify(_status_error(status, "x")) is expected

    def test_context_overflow_is_held_apart_from_other_bad_requests(self) -> None:
        """It is the one failure whose correct response is to send less, not again."""
        assert classify(_status_error(400, "maximum context length exceeded")) is (
            ErrorClass.CONTEXT_OVERFLOW
        )

    def test_a_timeout_is_transient(self) -> None:
        assert classify(TimeoutError("read timed out")) is ErrorClass.TRANSIENT

    async def test_a_terminal_failure_is_not_retried(self) -> None:
        client, events = _client(completion=_raises(_status_error(401, "no key")))
        with pytest.raises(ModelCallError) as excinfo:
            await _complete(client)
        assert excinfo.value.attempts == 1
        assert len(events) == 1


def test_tool_result_audit_metadata_is_not_echoed_to_the_model() -> None:
    """It is information a tool attached for the audit log, not context anyone wrote."""
    messages = serialise_tool_results(
        [ToolResult(tool_call_id="a", name="t", content="ok", metadata={"secret": "x"})]
    )
    assert messages == [{"role": "tool", "tool_call_id": "a", "content": "ok"}]
    assert "secret" not in json.dumps(messages)


def _status_error(status: int, message: str) -> Exception:
    exc = RuntimeError(message)
    exc.status_code = status  # type: ignore[attr-defined]
    return exc


def _returns(raw: Any) -> Any:
    async def _call(**_: Any) -> Any:
        return raw

    return _call


def _raises(exc: Exception) -> Any:
    async def _call(**_: Any) -> Any:
        raise exc

    return _call


def _capture(sink: dict[str, Any], raw: Any) -> Any:
    async def _call(**kwargs: Any) -> Any:
        sink["kwargs"] = kwargs
        return raw

    return _call


def test_a_model_failure_survives_being_re_raised() -> None:
    """Python assigns `__traceback__` on an exception as it propagates. A frozen
    dataclass refuses that, and on a slotted one the refusal surfaces as a TypeError
    that then replaces the real failure — a run against an unreachable endpoint
    recorded that TypeError as its outcome instead of the model error."""
    error = ModelCallError(
        error_class=ErrorClass.TERMINAL,
        provider_type="APIConnectionError",
        message="connection refused",
        attempts=1,
    )

    error.__traceback__ = None

    assert "connection refused" in str(error)


def test_the_profile_determines_where_the_call_goes_and_as_whom(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both fields were declared, validated and never sent, so a keyless local server
    worked only where the operator happened to export the provider library's own
    environment variables — which is the profile not determining the call at all."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = LLMClient(load_profile("qwen3.6-35b-a3b", MODELS), load_settings())

    kwargs, _ = client._build_kwargs(
        ModelRequest(system="s", messages=[{"role": "user", "content": "hi"}], tools=())
    )

    assert kwargs["api_base"] == "http://vllm.internal:8000/v1"
    assert kwargs["api_key"] == "EMPTY"


def test_a_quoted_key_in_the_environment_is_not_sent_with_its_quotes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The one credential the settings model never sees: the profile names the variable,
    so a quoted or newline-terminated key read straight from the environment reached the
    provider verbatim and came back as a 401 minutes into a run."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", '"sk-ant-quoted-by-an-env-file"\n')
    client = LLMClient(load_profile("claude-sonnet-4-6", MODELS), load_settings())

    kwargs, _ = client._build_kwargs(
        ModelRequest(system="s", messages=[{"role": "user", "content": "hi"}], tools=())
    )

    assert kwargs["api_key"] == "sk-ant-quoted-by-an-env-file"


def test_an_override_displaces_the_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pointing a run at another server is what the override is for, and a key given
    explicitly outranks one the profile names."""
    monkeypatch.setenv("MAVER_MODEL__ENDPOINT_OVERRIDE", "http://localhost:9/v1")
    monkeypatch.setenv("MAVER_MODEL__API_KEY", "sk-from-the-operator")
    client = LLMClient(load_profile("qwen3.6-35b-a3b", MODELS), load_settings())

    kwargs, _ = client._build_kwargs(
        ModelRequest(system="s", messages=[{"role": "user", "content": "hi"}], tools=())
    )

    assert kwargs["api_base"] == "http://localhost:9/v1"
    assert kwargs["api_key"] == "sk-from-the-operator"
