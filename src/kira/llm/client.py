"""The provider client: request assembly, response normalisation, retries, accounting.

This is the only module that imports the provider library, and the boundary is enforced
by a banned-import rule. Everything above it sees the neutral types in ``types.py``.

Most of what this module does is compensate for specific, measured provider behaviour.
Each workaround carries the evidence for its existence and has a regression test naming
the failure it prevents, because a workaround with no test is a comment waiting to be
deleted by someone who does not believe it.
"""

import asyncio
import hashlib
import json
import logging
import os
import random
import time
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from kira.contracts.events import ModelCallEvent
from kira.llm.errors import RETRYABLE, ErrorClass, ModelCallError, classify
from kira.llm.sampling import (
    SamplingResolution,
    applied_sampling,
    is_token_limit_finish,
    resolve_sampling,
)
from kira.llm.types import Message, ModelRequest, ModelResponse, ToolCall, ToolSpec, Usage
from kira.prompts.loader import read_asset
from kira.settings.base import clean_credential
from kira.settings.profiles import ModelProfile
from kira.settings.root import Settings

__all__ = [
    "LLMClient",
    "carry_reasoning_forward",
    "extract_reasoning_text",
    "prompt_digest",
    "sent_sampling",
    "serialise_assistant",
]

_log = logging.getLogger(__name__)

_BACKOFF_BASE_S = 1.0
_BACKOFF_CAP_S = 30.0


def sent_sampling(
    request: ModelRequest, profile: ModelProfile, resolved: SamplingResolution
) -> SamplingResolution:
    """The set a call is sent on, rescue levers included.

    ``applied_sampling`` answers which set a call runs on, and a rescue turn then
    displaces two of its knobs — after that rule has been consulted. Derived from the
    rule alone, the record said a rescue turn ran on the profile's reasoning effort,
    with a digest identical to an ordinary turn's, so the one field that exists to tell
    two configurations apart could not. The rescue is the turn that produces the graded
    submission in the cells it fires on.

    Args:
        request: The call.
        profile: The profile, for the rescue levers and the native extension fields.
        resolved: The agent's set, already resolved against settings.

    Returns:
        The resolution that reaches the provider.
    """
    applied = applied_sampling(request.sampling, profile, resolved)
    if not (request.lite_reasoning or request.suppress_reasoning):
        return applied
    # A rescue turn asks only for a schema-valid submission, and a greedy local model
    # will otherwise spend the whole budget in its reasoning channel and be cut off
    # before emitting one. A helper call asks for one JSON object under a small cap and
    # has the same problem for the same reason, but on every model rather than the ones
    # the profile names, so it says so itself.
    top_level = dict(applied.top_level)
    extra_body = dict(applied.extra_body)
    if request.suppress_reasoning or profile.quirks.rescue_disable_thinking:
        extra_body.setdefault("chat_template_kwargs", {})["enable_thinking"] = False
    if request.lite_reasoning and profile.quirks.rescue_reasoning_effort is not None:
        top_level["reasoning_effort"] = profile.quirks.rescue_reasoning_effort
    return SamplingResolution(
        applied.profile_name,
        {**applied.values, **top_level, **extra_body},
        top_level,
        extra_body,
        applied.overrides,
    )


def extract_reasoning_text(message: Any) -> str:
    """Pull the reasoning channel out of a response, whatever the provider calls it.

    Providers disagree on the field. Some expose a flat string, some a list of
    structured blocks under one of two names. Trying all three is the whole fix.

    Args:
        message: The provider's message object.

    Returns:
        The reasoning text, empty when the provider reported none.
    """
    reasoning = getattr(message, "reasoning_content", None)
    if reasoning:
        return str(reasoning)
    blocks = getattr(message, "thinking_blocks", None) or getattr(message, "reasoning_items", None)
    if blocks:
        parts: list[str] = []
        for block in blocks:
            if isinstance(block, dict):
                parts.append(
                    str(block.get("thinking") or block.get("text") or block.get("content") or "")
                )
            else:
                parts.append(str(block))
        return "\n".join(p for p in parts if p).strip()
    return ""


def _normalise_tool_calls(raw_calls: list[Any]) -> tuple[list[ToolCall], list[str]]:
    """Turn provider tool calls into neutral ones, repairing what can be repaired.

    Arguments arrive as a JSON string, an already-decoded mapping, or occasionally
    something else. A malformed payload becomes a parse error carried on the call rather
    than an exception, so the model can be told what was wrong and try again.

    Identifiers are synthesised when missing or repeated. A null identifier puts a null
    into the transcript, and a repeated one silently drops a result wherever results are
    matched to calls by identifier.

    Args:
        raw_calls: The provider's tool-call objects.

    Returns:
        The normalised calls and any notes worth recording on the audit event.
    """
    calls: list[ToolCall] = []
    notes: list[str] = []
    seen: set[str] = set()

    for index, raw in enumerate(raw_calls):
        function = getattr(raw, "function", None)
        name = str(getattr(function, "name", "") or "")
        raw_args = getattr(function, "arguments", None)
        if raw_args is None or raw_args == "":
            raw_args = "{}"

        arguments: dict[str, Any] = {}
        parse_error: str | None = None
        if isinstance(raw_args, dict):
            arguments = raw_args
            raw_args = json.dumps(raw_args)
        elif isinstance(raw_args, str):
            try:
                loaded = json.loads(raw_args)
            except json.JSONDecodeError as exc:
                parse_error = f"invalid JSON arguments: {exc.msg}"
            else:
                if isinstance(loaded, dict):
                    arguments = loaded
                else:
                    parse_error = "arguments JSON must decode to an object"
        else:
            parse_error = f"unsupported arguments type: {type(raw_args).__name__}"
            raw_args = "{}"

        call_id = str(getattr(raw, "id", "") or "")
        if not call_id or call_id in seen:
            reason = "missing" if not call_id else "duplicate"
            call_id = f"call_{index}_{name or 'unnamed'}"
            notes.append(f"synthesised {reason} tool_call_id -> {call_id}")
        seen.add(call_id)

        calls.append(
            ToolCall(
                id=call_id,
                name=name,
                arguments=arguments,
                raw_arguments=str(raw_args),
                parse_error=parse_error,
            )
        )
    return calls, notes


def serialise_assistant(response: ModelResponse) -> Message:
    """Build the assistant message that goes into the transcript.

    Two repairs apply, both consequences of a generation ending at its cap.

    A tool call cut off mid-emission has incomplete argument JSON. Echoing it back
    poisons the very next request: the server fails to parse the truncated arguments and
    rejects the whole call, and a result for a dropped call would dangle. Every call is
    dropped on a token-limit finish, because a fresh regeneration is what is being asked
    for, and a note tells the model what happened.

    A turn with no content and no tool call usually means a reasoning model spent its
    whole budget thinking and was cut off before emitting either. Carrying that reasoning
    forward as content lets a retry continue from it instead of re-deriving it and
    truncating again; without it, the reasoning is dropped and the loop stalls.

    Args:
        response: The normalised response.

    Returns:
        The assistant message.
    """
    truncated = is_token_limit_finish(response.finish_reason)
    content = (response.text or "").strip()
    calls = list(response.tool_calls)

    if truncated and calls:
        calls = []
        # Fold the reasoning forward here too, not only on the no-call branch below. The
        # loop answers this turn with a nudge telling the model to continue from the
        # state it was cut off in — and if the visible content was empty, that state is
        # otherwise just the discard note, which is the stall the fold exists to prevent.
        if not content:
            content = (response.reasoning or "").strip()
        note = read_asset("control/truncated_tool_call_note.txt").strip()
        content = f"{content}\n\n{note}".strip() if content else note

    if not content and not calls:
        reasoning = (response.reasoning or "").strip()
        if reasoning:
            content = reasoning
        elif truncated:
            content = read_asset("control/generation_truncated_note.txt").strip()

    message: Message = {"role": "assistant", "content": content}
    if calls:
        message["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    # A call whose arguments would not parse keeps its place, so its
                    # error result still resolves, but the malformed string is not
                    # echoed: a strict server would reject the next request over it,
                    # which is the same failure the truncation repair exists to avoid.
                    "arguments": "{}" if call.parse_error else call.raw_arguments,
                },
            }
            for call in calls
        ]
    return message


def prompt_digest(system: str, messages: Sequence[Message]) -> str:
    """Hash what a request sends the model, so a log row can be checked against it.

    Over the system prompt and the messages only, and not the tools or the sampling: a
    run keeps its transcript, and this is the digest a reader recomputes over a prefix
    of it to confirm that the prefix is what a given turn actually sent. A turn that
    matches no prefix was sent something else — which is the case worth finding.

    Args:
        system: The system prompt, as sent.
        messages: The messages, in order.

    Returns:
        Twelve hex characters, the length a sampling digest uses.
    """
    canonical = json.dumps(
        [system, list(messages)], sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def carry_reasoning_forward(message: Message, reasoning: str | None, limit: int) -> Message:
    """Append a bounded tail of the reasoning channel to an assistant message.

    The tail rather than the head: the useful part of a reasoning trace is where it
    ended up, not where it started.

    Args:
        message: The assistant message.
        reasoning: The reasoning text, if any.
        limit: Maximum characters to carry.

    Returns:
        The message, with reasoning appended when there was any.
    """
    text = (reasoning or "").strip()
    if not text:
        return message
    if len(text) > limit:
        text = "…" + text[-limit:]
    existing = str(message.get("content") or "").strip()
    message["content"] = f"{existing}\n\n{text}".strip() if existing else text
    return message


def _cached_prompt_tokens(usage: Any) -> int | None:
    """Read the cached prompt-token count, when the backend reports one.

    Self-hosted servers populate this from automatic prefix caching, which applies to
    the input only. Returns None when the backend omits the field, and a sink must treat
    that as unknown rather than as zero: at least one endpoint runs prefix caching
    without reporting it, so a zero here would be an invented measurement.

    Args:
        usage: The provider's usage object.

    Returns:
        The cached prompt tokens, or None when not reported.
    """
    details = getattr(usage, "prompt_tokens_details", None)
    for source in (details, usage):
        if source is None:
            continue
        value = getattr(source, "cached_tokens", None)
        if value is None and isinstance(source, dict):
            value = source.get("cached_tokens")
        if isinstance(value, int):
            return value
    return None


def _extract_usage(raw_usage: Any) -> Usage:
    """Normalise a provider usage object.

    Args:
        raw_usage: What the provider returned, possibly None.

    Returns:
        Token counts, with None wherever the provider reported nothing.
    """
    if raw_usage is None:
        return Usage()
    prompt = getattr(raw_usage, "prompt_tokens", None)
    completion = getattr(raw_usage, "completion_tokens", None)
    details = getattr(raw_usage, "completion_tokens_details", None)
    reasoning = getattr(details, "reasoning_tokens", None) if details is not None else None
    cached = _cached_prompt_tokens(raw_usage)
    uncached = prompt - cached if isinstance(prompt, int) and isinstance(cached, int) else None
    return Usage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        reasoning_tokens=reasoning,
        cached_prompt_tokens=cached,
        uncached_prompt_tokens=uncached,
    )


def _to_openai_tool(spec: ToolSpec) -> dict[str, Any]:
    """Render a tool for the provider's schema.

    Args:
        spec: The tool as the model should see it.

    Returns:
        The provider-shaped tool definition.
    """
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.parameters,
        },
    }


class LLMClient:
    """Calls one model, according to one profile.

    The retry loop is ours rather than the library's. With the library retrying
    internally, a recorded latency covers every attempt, the attempt count is lost, and
    a call that burned a prefill before being rejected records no tokens at all. One
    audit event per attempt is what makes "every call, including retries, is costed"
    true rather than aspirational.
    """

    def __init__(
        self,
        profile: ModelProfile,
        settings: Settings,
        *,
        sink: Callable[[ModelCallEvent], None] | None = None,
        completion: Callable[..., Awaitable[Any]] | None = None,
    ) -> None:
        """Build a client for one model profile.

        Args:
            profile: The model profile in use.
            settings: Resolved settings, for sampling overrides and runtime bounds.
            sink: Receives one event per attempt. Defaults to discarding them, which
                only a test should do.
            completion: The underlying completion callable. Injected by tests; the
                provider library is imported lazily so importing this module does not
                pull it in.
        """
        self.profile = profile
        self.settings = settings
        self.sampling: SamplingResolution = resolve_sampling(profile, settings)
        self._sink = sink
        self._completion = completion
        # Resolved here rather than per call, because here is startup. A credential the
        # environment mangled is then refused while the process is coming up, instead of
        # on the first model call, minutes into a run, where a provider's 401 reads as
        # the model failing.
        self._api_key = self._resolve_api_key()

    async def _call_provider(self, **kwargs: Any) -> Any:
        if self._completion is None:
            from kira.llm._provider import litellm  # noqa: PLC0415

            self._completion = litellm.acompletion
        return await self._completion(**kwargs)

    def _resolve_api_key(self) -> str | None:
        """Resolve the credential this client authenticates with.

        In precedence order: an explicit override, then the environment variable the
        profile names, then its placeholder for a keyless server. None means send
        nothing and let the provider library look for its own, which is what a hosted
        route configured entirely through the environment expects.

        The variable the profile names is cleaned exactly as a settings secret is. It is
        the one credential the settings model never sees — the profile decides which
        variable it is — so without this the quoted key that a settings override would
        have survived reaches the provider verbatim.

        Returns:
            The key, or None when the profile names no way to find one.

        Raises:
            ValueError: When the named variable holds something no request can carry.
        """
        override = self.settings.model.api_key
        if override is not None:
            return override.get_secret_value()
        named = self.profile.endpoint.api_key_env
        if named:
            from_env = clean_credential(os.environ.get(named))
            if isinstance(from_env, str):
                return from_env
        return self.profile.endpoint.api_key_literal

    def _build_kwargs(self, request: ModelRequest) -> tuple[dict[str, Any], SamplingResolution]:
        """Assemble the provider request.

        Args:
            request: The neutral request.

        Returns:
            Keyword arguments for the completion call, and the sampling set they carry.
        """
        profile = self.profile
        messages: list[Message] = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.extend(request.messages)

        # A request may carry its own sampling set, which is how an auxiliary call —
        # a quality gate, a nominator, a selector — runs deterministically while the
        # agent's own profile stays as measured. Declared and never read, this field
        # would silently run every gate on the agent's sampling, which is the failure it
        # exists to prevent: a gate at temperature 1.0 with a presence penalty answers
        # differently each time it is asked the same question.
        applied = sent_sampling(request, profile, self.sampling)
        top_level = applied.top_level
        extra_body = applied.extra_body

        kwargs: dict[str, Any] = {
            "model": profile.model,
            "messages": messages,
            "num_retries": 0,
            "timeout": profile.limits["request_timeout_s"],
            **top_level,
        }
        # The profile says how this model is called, and until now the two fields that
        # say *where* and *as whom* were declared, validated and never sent. A keyless
        # local server then worked only where the operator happened to export the
        # provider library's own environment variables, which is the profile not
        # determining the call at all.
        base_url = self.settings.model.endpoint_override or profile.endpoint.base_url
        if base_url:
            kwargs["api_base"] = base_url
        api_key = self._api_key
        if api_key is not None:
            kwargs["api_key"] = api_key
        if extra_body:
            kwargs["extra_body"] = extra_body
        if profile.quirks.drop_unsupported_params:
            kwargs["drop_params"] = True

        if request.tools:
            kwargs["tools"] = [_to_openai_tool(t) for t in request.tools]
            # A tool choice is never sent without tools: some providers reject the
            # combination outright, and it cannot mean anything anyway.
            choice = request.tool_choice
            if choice is None and profile.quirks.tool_choice != "omit":
                choice = profile.quirks.tool_choice
            if choice is not None:
                kwargs["tool_choice"] = choice
        return kwargs, applied

    async def complete(self, request: ModelRequest, *, caller: str) -> ModelResponse:
        """Run one completion, retrying only what retrying can fix.

        Args:
            request: The messages, tools and sampling for this call.
            caller: What issued the call. Required so that no call escapes accounting.

        Returns:
            The normalised response.

        Raises:
            ModelCallError: When the call failed and retrying did not help.
        """
        # The object the kwargs were built from, not a second resolution of the same
        # rule: reported from the rule alone, a rescue turn recorded the reasoning
        # effort the profile declares rather than the one it displaced.
        kwargs, applied = self._build_kwargs(request)
        attempts = self.profile.limits["num_retries"] + 1
        last: ModelCallError | None = None
        # What the caller waits, which is what a per-incident latency is summed from.
        # Every row's duration is measured from here and the attempt's own from below.
        call_started = time.monotonic()

        for attempt in range(1, attempts + 1):
            started = time.monotonic()
            try:
                raw = await self._call_provider(**kwargs)
            except Exception as exc:
                duration_ms = int((time.monotonic() - started) * 1000)
                error_class = classify(exc)
                self._emit(
                    caller=caller,
                    sampling=applied,
                    usage=_extract_usage(getattr(exc, "usage", None)),
                    finish_reason=None,
                    duration_ms=int((time.monotonic() - call_started) * 1000),
                    last_attempt_ms=duration_ms,
                    attempt=attempt,
                    error=f"{type(exc).__name__}: {exc}",
                )
                last = ModelCallError(
                    error_class=error_class,
                    provider_type=type(exc).__name__,
                    message=str(exc),
                    attempts=attempt,
                    status_code=getattr(exc, "status_code", None),
                )
                if error_class not in RETRYABLE or attempt == attempts:
                    raise last from exc
                await asyncio.sleep(_backoff(attempt, error_class))
                continue

            duration_ms = int((time.monotonic() - started) * 1000)
            response = self._normalise(raw, duration_ms=duration_ms, attempts=attempt)
            self._emit(
                caller=caller,
                sampling=applied,
                usage=response.usage,
                finish_reason=response.finish_reason,
                duration_ms=int((time.monotonic() - call_started) * 1000),
                last_attempt_ms=duration_ms,
                attempt=attempt,
                notes=response.notes,
            )
            return response

        raise (
            last
            if last
            else ModelCallError(
                error_class=ErrorClass.TERMINAL,
                provider_type="NoAttempt",
                message="no completion attempt was made",
                attempts=0,
            )
        )

    def _normalise(self, raw: Any, *, duration_ms: int, attempts: int) -> ModelResponse:
        choice = raw.choices[0]
        message = choice.message
        calls, notes = _normalise_tool_calls(list(getattr(message, "tool_calls", None) or []))
        usage = _extract_usage(getattr(raw, "usage", None))

        if usage.cached_prompt_tokens is None and self.profile.quirks.reports_cached_tokens:
            # A null under a profile that claims to report is worth acting on; a null
            # under one that does not is expected and must not be read as zero.
            _log.warning(
                "profile %s declares reports_cached_tokens but the endpoint returned no "
                "cached count",
                self.profile.name,
            )

        return ModelResponse(
            text=getattr(message, "content", None),
            tool_calls=tuple(calls),
            finish_reason=getattr(choice, "finish_reason", None),
            reasoning=extract_reasoning_text(message) or None,
            usage=usage,
            duration_ms=duration_ms,
            attempts=attempts,
            estimated_cost_usd=None,
            notes=tuple(notes),
            raw=raw,
        )

    def _emit(
        self,
        *,
        caller: str,
        sampling: SamplingResolution,
        usage: Usage,
        finish_reason: str | None,
        duration_ms: int,
        last_attempt_ms: int,
        attempt: int,
        error: str | None = None,
        notes: Sequence[str] = (),
    ) -> None:
        if self._sink is None:
            return
        self._sink(
            ModelCallEvent(
                caller=caller,
                model=self.profile.model,
                profile=self.profile.name,
                sampling=sampling.as_record(),
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                reasoning_tokens=usage.reasoning_tokens,
                cached_tokens=usage.cached_prompt_tokens,
                finish_reason=finish_reason,
                duration_ms=duration_ms,
                last_attempt_ms=last_attempt_ms,
                attempt=attempt,
                error=error,
                notes=list(notes),
            )
        )


def _backoff(attempt: int, error_class: ErrorClass) -> float:
    """Compute the wait before the next attempt.

    Exponential with full jitter, matching what the library did internally, so the
    timing profile of a run is unchanged by taking the retry loop over.

    Args:
        attempt: The attempt that just failed, counting from one.
        error_class: What kind of failure it was.

    Returns:
        Seconds to wait.
    """
    ceiling = min(_BACKOFF_BASE_S * (2 ** (attempt - 1)), _BACKOFF_CAP_S)
    if error_class is ErrorClass.RATE_LIMIT:
        ceiling = min(ceiling * 2, _BACKOFF_CAP_S)
    return random.uniform(0, ceiling)
