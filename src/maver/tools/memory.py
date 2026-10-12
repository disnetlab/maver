"""Reading the experience store: search it, open one record, read a role's trace.

Two things here are about what a small model can actually do, and both were observed
live rather than anticipated.

Search results expose short ``memory-N`` handles. A model reproduces a short handle
reliably and hallucinates a session UUID: a run once passed a placeholder identifier
straight out of documentation to open a record. The handles are resolved back to real
identifiers here, and the handle is what appears in model-facing output so a later turn
copying it back still works.

The scopes hide different things on purpose. Evidence scope returns what an incident
looked like and withholds what was concluded about it, because a role that reads a prior
conclusion tends to adopt it, and the whole value of a precedent is as comparable
observations rather than as an answer.

Because the handle is what the model sees, the real identifier survives the call only in
the run's alias map, which dies with the process. Each of these tools therefore returns
the record it served alongside its text, and the loop writes that to the call's event
row. Retrieval the roles do is otherwise the one delivery path whose provenance is
unrecorded, and what a role was actually served is the question every analysis of this
mechanism starts from.

Every store call here goes to a thread. The ports are synchronous by design, and the
deployment behind them is a database on another host: called inline, one of these blocks
the event loop running the incident, and blocks it for as long as the database takes.
Nothing above can cut that short, because a timeout is a coroutine and a coroutine cannot
run while the loop is inside a synchronous call. What actually bounds the wait is the
backend's own connect and statement timeouts; the hop off the loop is what lets the run
keep its clock while it waits.
"""

import asyncio
import json
from dataclasses import asdict
from typing import Any, cast

from maver.memory.ports import SearchScope
from maver.memory.retrieval import Retriever
from maver.tools.context import ToolContext
from maver.tools.errors import tool_error
from maver.tools.policy import read_only
from maver.tools.registry import ToolOutput, tool

__all__ = ["memory_search", "open_memory", "session_trace_summary"]

_EVIDENCE_FIELDS = ("alert_name", "namespace")
_LESSON_FIELDS = ("alert_name", "namespace", "fault_category", "affected_resource")

_DISABLED = "memory retrieval is disabled for this configuration"
_BAD_SCOPE = "scope must be 'evidence' or 'lesson'"

_EVIDENCE_NOTE = (
    "Evidence scope intentionally hides causal lessons and the mixed L1 summary. "
    "These are comparable reported observations, not necessarily correct or causal. "
    "Inspect lesson scope only if relying on the prior incident for causal or "
    "remediation support."
)


def _scope_ok(scope: str) -> bool:
    return scope in ("evidence", "lesson")


def _provenance(record_id: str, handle: str | None, score: float | None) -> dict[str, Any]:
    """Describe one served record for the audit log, never for the model.

    A null handle is a role that named the record itself rather than copying a handle
    out of a search result, which is the shape a leaked identifier has.
    """
    return {"record_id": record_id, "handle": handle, "score": score}


@tool(
    name="memory_search",
    safety="read_only",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Free-text description of the current problem.",
            },
            "scope": {
                "type": "string",
                "description": ("`evidence` for observations or `lesson` for causal/action reuse."),
            },
            "n_results": {
                "type": "integer",
                "description": "Max records to return (default 5).",
            },
        },
        "required": ["query"],
    },
)
async def memory_search(
    ctx: ToolContext,
    query: str,
    scope: SearchScope = "evidence",
    n_results: int | None = None,
) -> str | ToolOutput:
    """Search one projection of the distilled past.

    Args:
        ctx: The run's context.
        query: What the current incident looks like.
        scope: Which projection to match.
        n_results: How many records to return.

    Returns:
        The matches, or a statement that there are none.
    """
    if ctx.memory is None:
        return _DISABLED
    if not _scope_ok(scope):
        return tool_error(_BAD_SCOPE)
    if not query.strip():
        # An empty query matches everything, so it discharges any obligation to search
        # without doing any work and returns the least related records in the corpus.
        return tool_error(
            "query must not be empty — search on the concrete symptoms and "
            "error signatures you observed (exact error strings, resource kinds, "
            "observed values)"
        )

    # A model asking for more than the configuration allows still gets the
    # configured cap, so the number of precedents in context stays a controlled
    # variable rather than something the model can vary mid-run.
    # Tool contexts can carry duck-typed custom stores at runtime as well as Retriever.
    memory = cast(object, ctx.memory)
    if isinstance(memory, Retriever) and memory.retrieval_policy == "historical":
        # No new model arguments: use only the current incident's known identifiers.
        # This is a context-based subset of the prototype's generic role metadata API.
        alert: dict[str, Any] = {}
        if ctx.sessions is not None:
            session = await asyncio.to_thread(ctx.sessions.get_session, ctx.session_id)
            if session is not None:
                alert = session.alert
        query_metadata = {
            name: value
            for name, value in {
                "alert_name": str(alert.get("alertname") or "").strip(),
                "namespace": str(alert.get("namespace") or ctx.scope.namespace or "").strip(),
            }.items()
            if value
        }
        hits = await asyncio.to_thread(
            memory.search,
            query,
            scope=scope,
            exclude=ctx.memory_exclude,
            query_metadata=query_metadata or None,
        )
    else:
        hits = await asyncio.to_thread(
            ctx.memory.search, query, scope=scope, exclude=ctx.memory_exclude
        )
    if n_results is not None and n_results > 0:
        hits = hits[:n_results]
    if not hits:
        # Carried verbatim. It now occurs for a second reason — nothing cleared the
        # relevance floor — and that is the point: retrieval that can only ever return
        # its top-k cannot say this, and five unrelated incidents get read as evidence.
        return "no relevant past sessions found"

    fields = _EVIDENCE_FIELDS if scope == "evidence" else _LESSON_FIELDS
    lines: list[str] = []
    served: list[dict[str, Any]] = []
    for index, hit in enumerate(hits, start=1):
        handle = f"memory-{index}"
        ctx.memory_aliases[handle] = hit.session_id
        served.append(_provenance(hit.session_id, handle, hit.score))
        described = "  ".join(
            f"{name}={hit.metadata.get(name)}" for name in fields if hit.metadata.get(name)
        )
        score = f"{hit.score:.3f}" if hit.score is not None else "n/a"
        lines.append(f"[{handle}]  score={score}  {described}".rstrip())
        lines.append(hit.summary)
        lines.append("")
    lines.append(f'Open one with `open_memory(session_id="memory-N", scope="{scope}")`.')
    return ToolOutput("\n".join(lines), tuple(served))


@tool(
    name="open_memory",
    safety="read_only",
    parameters={
        "type": "object",
        "properties": {
            "session_id": {
                "type": "string",
                "description": (
                    'The `memory-N` handle shown in a memory_search result (e.g. "memory-1"). '
                    "A raw session_id is also accepted."
                ),
            },
            "scope": {
                "type": "string",
                "description": "`evidence` or `lesson`.",
            },
        },
        "required": ["session_id"],
    },
)
async def open_memory(
    ctx: ToolContext, session_id: str, scope: SearchScope = "evidence"
) -> str | ToolOutput:
    """Expand one stored record.

    Args:
        ctx: The run's context.
        session_id: A ``memory-N`` handle or a raw session id.
        scope: Which projection to expand.

    Returns:
        The record as JSON, or a not-found marker.
    """
    if ctx.memory is None:
        return _DISABLED
    if not _scope_ok(scope):
        return tool_error(_BAD_SCOPE)

    reference = (session_id or "").strip()
    resolved = ctx.memory_aliases.get(reference, reference)
    # Recorded on the refusal too. A withheld record is reported as not found, so the
    # journal is the only place the leave-one-out set can be shown to have held.
    served = (_provenance(resolved, reference if reference != resolved else None, None),)
    record = (
        None if _withheld(ctx, resolved) else await asyncio.to_thread(ctx.memory.open, resolved)
    )
    if record is None:
        return ToolOutput(json.dumps({"error": "not found", "ref": session_id}), served)

    payload: dict[str, Any] = asdict(record)
    # The handle is what the model can reproduce, so that is what it gets back. The
    # real identifier stays in the audit metadata.
    payload["session_id"] = reference
    # The environment's own keys, and the only place a whole record is rendered for a
    # model. Under an evaluation harness one of them is the ground-truth fault label.
    payload.pop("labels", None)

    payload.pop("authoritative_correction", None)

    if not payload.get("verification_verdict"):
        payload.pop("verification_verdict", None)
    if not payload.get("evaluation_label"):
        for name in (
            "evaluation_label",
            "evaluation_source",
            "evaluation_diagnosis_success",
            "evaluation_diagnosis_score",
            "evaluation_mitigation_success",
        ):
            payload.pop(name, None)

    # Held out of search results so it cannot be skimmed and copied; surfaced here
    # because opening one record is a deliberate act about a precedent the role has
    # already judged relevant. Still a conclusion about a different incident, so it is
    # labelled as one rather than as an answer.
    corrected = " ".join(str(payload.pop("corrected_cause", "") or "").split())
    if corrected and scope == "lesson":
        payload["corrected_cause"] = (
            "Cause established for THAT incident, by external evaluation — not a "
            "finding about yours. Use it to judge whether this precedent resembles "
            f"your case; it is not evidence for your diagnosis. {corrected}"
        )

    if scope == "evidence":
        for name in ("lesson_text", "incident_summary", "fault_category", "affected_resource"):
            payload.pop(name, None)
        payload["note"] = _EVIDENCE_NOTE
    else:
        payload.pop("evidence_text", None)

    return ToolOutput(json.dumps(payload, indent=2), served)


@tool(
    name="session_trace_summary",
    safety="read_only",
    parameters={
        "type": "object",
        "properties": {
            "session_id": {
                "type": "string",
                "description": (
                    "The `memory-N` handle from a memory_search result, or a raw session_id."
                ),
            },
            "agent": {
                "type": "string",
                "description": "Restrict to one role's trace, e.g. `rca`.",
            },
        },
        "required": ["session_id"],
    },
)
async def session_trace_summary(
    ctx: ToolContext, session_id: str, agent: str | None = None
) -> str | ToolOutput:
    """Read what a role actually did during a past incident.

    The record says what was concluded; this says what was tried on the way. They are
    different kinds of evidence, and a precedent whose conclusion was wrong can still
    have a useful account of what was ruled out.

    Args:
        ctx: The run's context.
        session_id: A ``memory-N`` handle or a raw session id.
        agent: Restrict to one role.

    Returns:
        The trace summaries as JSON, or a not-found marker.
    """
    if ctx.sessions is None:
        return _DISABLED

    reference = (session_id or "").strip()
    resolved = ctx.memory_aliases.get(reference, reference)
    served = (_provenance(resolved, reference if reference != resolved else None, None),)
    record = (
        None
        if _withheld(ctx, resolved)
        else await asyncio.to_thread(ctx.sessions.get_session, resolved)
    )
    if record is None:
        return ToolOutput(json.dumps({"error": "not found", "ref": session_id}), served)

    runs = await asyncio.to_thread(ctx.runs.runs_for, resolved) if ctx.runs is not None else []
    summaries = [
        {"agent": run.agent_name, **run.trace_summary} for run in runs if run.trace_summary
    ]
    if agent:
        summaries = [s for s in summaries if s.get("agent") == agent or s.get("role") == agent]
    if not summaries:
        return ToolOutput(
            json.dumps({"session_id": reference, "traces": [], "note": "no trace recorded"}),
            served,
        )
    return ToolOutput(json.dumps({"session_id": reference, "traces": summaries}, indent=2), served)


def _withheld(ctx: ToolContext, session_id: str) -> bool:
    """Whether this record is one retrieval must not serve.

    Exclusion has to hold on the way in as well as on the way out. Search filters the
    set, but both of the tools that take an identifier accept a raw one, so a role that
    has the id from anywhere — a trace, an alert field, an earlier turn — reads the
    withheld record straight out of the store and the leave-one-out set means nothing.

    Refused as not-found rather than as withheld. Naming it would confirm that a record
    for this incident exists and which identifier it has, which on a generalisation arm
    is itself a signal about the answer. The call and its arguments are in the journal,
    so the refusal stays auditable without being legible to the model.
    """
    return session_id in ctx.memory_exclude


read_only("memory_search")
read_only("open_memory")
read_only("session_trace_summary")
