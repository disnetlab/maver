"""Capture every logical model call and its provider attempts in the incident audit."""

import asyncio
import time
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from maver.audit.journal import Journal
from maver.contracts.events import ModelAttemptEvent, ModelCallEvent, ModelCallStartedEvent
from maver.llm.client import LLMClient, prompt_digest, sent_sampling
from maver.llm.observations import error_record, observe_calls, public_response, serialisable
from maver.llm.sampling import SamplingResolution
from maver.llm.types import Completion, ModelRequest, ModelResponse
from maver.settings.profiles import ModelProfile

__all__ = ["RecordingCompletion"]


def _response(journal: Journal, response: ModelResponse) -> dict[str, Any]:
    """Capture the model-visible response and the SDK body before normalisation."""
    return journal.capture(
        "model_response",
        {
            "capture_layer": "normalised_response_and_public_provider_body",
            "normalised": {
                "text": response.text,
                "reasoning": response.reasoning,
                "tool_calls": serialisable(response.tool_calls),
                "finish_reason": response.finish_reason,
                "usage": serialisable(response.usage),
                "notes": list(response.notes),
                "estimated_cost_usd": response.estimated_cost_usd,
            },
            "provider_body": public_response(response.raw) if response.raw is not None else None,
            "provider_capture": (
                "sdk_public_body"
                if isinstance(response.raw, dict)
                or callable(getattr(response.raw, "model_dump", None))
                else "public_field_projection"
                if response.raw is not None
                else "unavailable"
            ),
        },
    )


def _request(
    journal: Journal, request: ModelRequest, invocation: dict[str, Any] | None
) -> dict[str, Any]:
    """Deduplicate messages and schemas; the small manifest describes their order.

    The provider capture is the redacted SDK invocation, not the HTTP payload the SDK
    subsequently constructs. Custom Completion implementations expose the neutral layer.
    """
    neutral = {
        "system": journal.capture("model_system", request.system),
        "messages": [journal.capture("model_message", message) for message in request.messages],
        "tools": [journal.capture("model_tool", serialisable(tool)) for tool in request.tools],
        "tool_choice": request.tool_choice,
        "sampling": serialisable(request.sampling),
        "lite_reasoning": request.lite_reasoning,
        "suppress_reasoning": request.suppress_reasoning,
    }
    provider = None
    if invocation is not None:
        provider = {
            **{key: value for key, value in invocation.items() if key not in {"messages", "tools"}},
            "messages": [journal.capture("model_message", item) for item in invocation["messages"]],
            "tools": [journal.capture("model_tool", item) for item in invocation.get("tools", [])],
        }
    return journal.capture(
        "model_request",
        {
            "capture_layer": "litellm_invocation" if provider is not None else "neutral_request",
            "neutral": neutral,
            "provider": provider,
        },
    )


@dataclass(frozen=True, slots=True)
class RecordingCompletion:
    """Capture calls, with incident identity and locally stored full request evidence."""

    inner: Completion
    journal: Journal
    profile: ModelProfile
    sampling: SamplingResolution

    async def complete(self, request: ModelRequest, *, caller: str) -> ModelResponse:
        """Run one logical call, linking its request, attempts, response and failures."""
        call_id = str(uuid4())
        recorded_start = False
        attempts_observed = 0
        last_attempt: dict[str, Any] = {}
        response_ref: dict[str, Any] = {}

        def start(invocation: dict[str, Any] | None = None) -> None:
            nonlocal recorded_start
            if recorded_start:
                return
            self.journal.record(
                ModelCallStartedEvent(
                    call_id=call_id,
                    caller=caller,
                    model=self.profile.model,
                    profile=self.profile.name,
                    message_count=len(request.messages),
                    digest=prompt_digest(request.system, request.messages),
                    request=_request(self.journal, request, invocation),
                )
            )
            recorded_start = True

        def observe(kind: str, detail: dict[str, Any]) -> None:
            nonlocal last_attempt, response_ref, attempts_observed
            if kind == "request":
                start(detail["invocation"])
                return
            attempts_observed += 1
            last_attempt = detail
            response = detail.get("response")
            captured_response = {}
            if isinstance(response, ModelResponse):
                response_ref = captured_response = _response(self.journal, response)
            elif detail.get("raw") is not None:
                captured_response = self.journal.capture(
                    "model_provider_response", public_response(detail["raw"])
                )
            self.journal.record(
                ModelAttemptEvent(
                    call_id=call_id,
                    caller=caller,
                    model=self.profile.model,
                    profile=self.profile.name,
                    attempt=detail["attempt"],
                    status=detail["status"],
                    duration_ms=detail["duration_ms"],
                    usage=serialisable(detail["usage"]),
                    response=captured_response,
                    error=self.journal.capture("model_error", detail["error"])
                    if detail.get("error")
                    else {},
                )
            )

        # A real client reports its actual assembled invocation before sending it.
        # A custom client cannot promise an SDK/HTTP body, so name the neutral layer.
        if getattr(self.inner.complete, "__func__", None) is not LLMClient.complete:
            start()
        started = time.monotonic()
        try:
            with observe_calls(observe):
                response = await self.inner.complete(request, caller=caller)
        except BaseException as exc:
            start()  # Request assembly can itself fail before reaching the provider.
            detail = last_attempt.get("error") or error_record(exc)
            if (
                isinstance(exc, asyncio.CancelledError)
                and last_attempt.get("status") != "cancelled"
            ):
                detail = {**error_record(exc), "phase": "between_attempts"}
            usage = serialisable(last_attempt.get("usage")) or {}
            self.journal.record(
                ModelCallEvent(
                    attempts_observed=attempts_observed,
                    call_id=call_id,
                    caller=caller,
                    model=self.profile.model,
                    profile=self.profile.name,
                    sampling=sent_sampling(request, self.profile, self.sampling).as_record(),
                    duration_ms=int((time.monotonic() - started) * 1000),
                    last_attempt_ms=last_attempt.get("duration_ms"),
                    attempt=last_attempt.get("attempt", getattr(exc, "attempts", 1)),
                    prompt_tokens=usage.get("prompt_tokens"),
                    completion_tokens=usage.get("completion_tokens"),
                    reasoning_tokens=usage.get("reasoning_tokens"),
                    cached_tokens=usage.get("cached_prompt_tokens"),
                    error="cancelled"
                    if isinstance(exc, asyncio.CancelledError)
                    else f"{detail['type']}: {detail['message']}"[:1000],
                    error_detail=self.journal.capture("model_error", detail),
                )
            )
            raise
        start()
        self.journal.record(
            ModelCallEvent(
                attempts_observed=attempts_observed,
                call_id=call_id,
                caller=caller,
                model=self.profile.model,
                profile=self.profile.name,
                sampling=sent_sampling(request, self.profile, self.sampling).as_record(),
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                reasoning_tokens=response.usage.reasoning_tokens,
                cached_tokens=response.usage.cached_prompt_tokens,
                duration_ms=int((time.monotonic() - started) * 1000),
                last_attempt_ms=response.duration_ms,
                finish_reason=response.finish_reason,
                attempt=response.attempts,
                notes=list(response.notes),
                response=response_ref or _response(self.journal, response),
            )
        )
        return response
