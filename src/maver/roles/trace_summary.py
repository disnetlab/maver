"""A bounded, deterministic account of what a role did.

No model is involved. The offline learning pass reads these rather than the transcripts,
because a transcript is unbounded and mostly tool output, while what a later incident
needs is what was looked at, what was concluded and what went wrong.

Every list is capped and every string clipped. The caps are the point: an unbounded
summary is a transcript with extra steps, and the whole reason this exists is to give the
learning pass something it can hold several of at once.
"""

import json
from typing import Any

from maver.contracts.artifacts import RoleTraceSummary
from maver.llm.types import Message
from maver.tools.errors import call_failed

__all__ = ["build_trace_summary"]

_GOALS: dict[str, str] = {
    "rca": "Diagnose the incident from telemetry and produce a RootCause artefact.",
    "planning": "Turn the root cause and relevant memory into a remediation Plan.",
    "execution": (
        "Apply the approved plan safely, verify it took effect at the resource level, "
        "and report what was applied."
    ),
    "verify": (
        "Independently check end-to-end system recovery and decide whether to end the "
        "pipeline or hand back."
    ),
    "single": "Diagnose the incident and remediate it, without a separate planning stage.",
    "single.diagnosis": "Diagnose the incident, for a benchmark that grades detection on its own.",
    "single.mitigation": "Remediate the incident the same run has just diagnosed.",
    "learning": "Distil the incident into a record a later incident can use.",
}

_OMITTED = (
    "Raw transcript is retained for audit only; this summary keeps tool names, "
    "bounded result excerpts, and the structured output artefact summary."
)


def build_trace_summary(
    role: str,
    *,
    transcript: list[Message],
    output: Any | None = None,
    error: str | None = None,
) -> RoleTraceSummary:
    """Summarise one role's run.

    Args:
        role: The role that ran.
        transcript: Its conversation, used for the tool-call account.
        output: The artifact it submitted, if any.
        error: Why it ended without submitting, if it did.

    Returns:
        The bounded summary.
    """
    tool_calls, tool_errors = _summarise_tool_calls(transcript)
    payload = _normalise_output(output)
    observations, decisions, output_summary = _summarise_output(role, payload)

    uncertainties = list(tool_errors)
    if error:
        uncertainties.append(_clip(error, 220))
    if not payload and not error:
        # Neither an artifact nor a stated reason. Recorded rather than left blank,
        # because a stage that produced nothing and said nothing is the case a reader
        # most needs pointed out.
        uncertainties.append("No structured output artefact was recorded for this stage.")

    return RoleTraceSummary(
        role=role,
        goal=_GOALS.get(role, ""),
        key_observations=observations[:8],
        decisions=decisions[:8],
        tool_calls=tool_calls[:16],
        errors_or_uncertainties=uncertainties[:8],
        output_summary=output_summary,
        omitted_details_hint=_OMITTED,
    )


def _normalise_output(output: Any | None) -> dict[str, Any] | None:
    if output is None:
        return None
    if hasattr(output, "model_dump"):
        dumped = output.model_dump(mode="json")
        return dict(dumped) if isinstance(dumped, dict) else {"value": str(dumped)}
    if isinstance(output, dict):
        return output
    return {"value": str(output)}


def _summarise_output(role: str, output: dict[str, Any] | None) -> tuple[list[str], list[str], str]:
    if not output:
        return [], [], ""

    if role == "single.diagnosis":
        # Free text rather than a structured diagnosis: this configuration submits what
        # the benchmark's own reference agents submit, which is a sentence.
        return [], [], _clip(str(output.get("solution", "")), 320)

    if role == "single.mitigation":
        return (
            [],
            [f"succeeded={output.get('succeeded', '?')}"],
            _clip(str(output.get("summary", "")), 320),
        )

    if role in ("rca", "single"):
        observations = [_clip(str(v), 220) for v in output.get("evidence", [])]
        decisions = [
            f"fault_type={output.get('fault_type', '?')}",
            f"affected_resource={output.get('affected_resource', '?')}",
            f"confidence={output.get('confidence', '?')}",
        ]
        return observations, decisions, _clip(str(output.get("summary", "")), 320)

    if role == "planning":
        steps = output.get("steps") or []
        observations = (
            [_clip(str(output.get("expected_outcome", "")), 260)]
            if output.get("expected_outcome")
            else []
        )
        decisions = [
            _clip(
                f"{index + 1}. {step.get('action', '?')} -> {step.get('target', '?')}: "
                f"{step.get('justification', '')}",
                280,
            )
            for index, step in enumerate(steps)
            if isinstance(step, dict)
        ]
        if output.get("rollback"):
            decisions.append(_clip(f"rollback: {output['rollback']}", 260))
        return observations, decisions, _clip(f"{len(steps)} planned step(s).", 160)

    if role == "execution":
        steps = output.get("steps") or []
        observations = [
            _clip(str(step.get("observation", "")), 240)
            for step in steps
            if isinstance(step, dict) and step.get("observation")
        ]
        decisions = [
            f"succeeded={output.get('succeeded', '?')}",
            f"drift_detected={output.get('drift_detected', '?')}",
        ]
        return observations, decisions, _clip(str(output.get("final_state_summary", "")), 320)

    if role == "verify":
        checks = [_clip(str(v), 220) for v in output.get("checks", [])]
        decisions = [
            f"verdict={output.get('verdict', '?')}",
            f"recovered={output.get('recovered', '?')}",
        ]
        if output.get("needs_replan"):
            decisions.append("needs_replan")
        if output.get("needs_reassessment"):
            decisions.append("needs_reassessment")
        return checks, decisions, _clip(str(output.get("summary", "")), 320)

    return [], [], ""


def _summarise_tool_calls(messages: list[Message]) -> tuple[list[str], list[str]]:
    """Render each call as one line, pairing a result back to the call that asked for it."""
    names: dict[str, str] = {}
    arguments: dict[str, str] = {}
    lines: list[str] = []
    errors: list[str] = []

    for message in messages:
        if message.get("role") == "assistant":
            for call in message.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                function = call.get("function") or {}
                call_id = str(call.get("id") or "")
                if call_id:
                    names[call_id] = str(function.get("name") or "?")
                    arguments[call_id] = _format_args(function.get("arguments"))
        elif message.get("role") == "tool":
            call_id = str(message.get("tool_call_id") or "")
            name = str(message.get("name") or names.get(call_id) or "?")
            content = str(message.get("content", ""))
            status = "ERROR" if call_failed(content) else "OK"
            line = f"[{status}] {name}"
            if arguments.get(call_id):
                line += f"({arguments[call_id]})"
            excerpt = _clip(content.replace("\n", " "), 240)
            if excerpt:
                line += f" -> {excerpt}"
            lines.append(line)
            if status == "ERROR":
                errors.append(f"{name}: {_clip(content, 240)}")

    return lines, errors


def _format_args(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, str):
        try:
            loaded = json.loads(raw)
        except json.JSONDecodeError:
            return _clip(raw.replace("\n", " "), 160)
    else:
        loaded = raw
    if not isinstance(loaded, dict):
        return _clip(str(loaded), 160)
    return ", ".join(
        f"{key}={_clip(str(value).replace(chr(10), ' '), 60)}" for key, value in loaded.items()
    )


def _clip(value: str, limit: int) -> str:
    text = " ".join(value.split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."
