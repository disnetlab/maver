"""What a role is: a declaration, plus one function.

Everything uniform lives in the node wrapper above — opening the run row, writing the
audit events, summarising the trace, merging the state, applying a hand-back. So a role
module contains only what makes that role different from the others, which is the point:
the differences between RCA and Planning should be readable in a hundred lines, not
inferred from two files that are ninety percent the same.

A role never reads a global. Everything it needs arrives in its context, which is also
what makes one testable without a cluster, a database or a model.
"""

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from kira.audit.journal import Journal
from kira.contracts.artifacts import LearningPacket
from kira.contracts.routing import HandBack
from kira.contracts.state import IncidentState
from kira.llm.types import Completion, Message
from kira.memory.retrieval import Retriever
from kira.middleware.base import MiddlewareStack
from kira.settings.profiles import ModelProfile
from kira.settings.root import Settings
from kira.tools.context import ToolContext
from kira.tools.toolsets import ResolvedToolset

__all__ = [
    "RoleContext",
    "RoleFn",
    "RoleOutcome",
    "RoleSpec",
    "StateDelta",
    "stage_role",
]

StateDelta = Mapping[str, Any]
"""A partial :class:`IncidentState`, merged onto the accumulated state by the graph."""


def stage_role(stage: str) -> str:
    """The role that runs a stage label.

    A role announcing more than one graded question labels each phase ``role.phase``,
    and that label is what the write policy is declared under and what a checkpoint
    records. Everything keyed by role — the toolset, the node, the tool context —
    needs the role back. Resolving it by convention in one place is what stops a
    resume from looking up ``single.mitigation`` in a map keyed by ``single``.

    Args:
        stage: The stage label, either a role name or ``role.phase``.

    Returns:
        The role name.
    """
    return stage.split(".", 1)[0]


@dataclass(frozen=True, slots=True)
class RoleContext:
    """One role run's inputs, built by the node wrapper."""

    spec: "RoleSpec"
    state: IncidentState
    """Already hand-back-applied, so a role reads the same shape on first entry and re-entry."""
    handback: HandBack | None
    """Why this is a re-entry, or None for a first entry."""
    settings: Settings
    completion: Completion
    profile: ModelProfile
    """The model profile in use.

    A role needs it for the two quirks that belong to the model rather than to the
    harness: whether this endpoint accepts a forced tool choice, and whether reasoning
    is carried between turns. Without it both silently take the loop's defaults, so a
    profile declaring `tool_choice: omit` still gets `required` on the wire — which is
    the entire reason that quirk exists.
    """
    middleware: MiddlewareStack
    """The hook stack every tool call passes through.

    Not optional. Absent it, the approval gate is registered and never consulted, and
    every cluster write a role issues is dispatched ungated — the exact failure the gate
    was made structural to prevent.
    """
    write_lock: asyncio.Lock
    """Held for the duration of any role run that mutates the cluster.

    Multiple incidents can be accepted concurrently, and diagnosis and planning run in
    parallel safely because they are read-only. Remediation is the sole writer, and two
    remediations running at once can issue conflicting changes — both scaling the same
    deployment, or one rolling back while the other rolls forward. One lock, held per
    run rather than per call, serialises them; the queue is FIFO by asyncio's
    implementation.

    Required rather than optional, like :attr:`middleware`, because a run that silently
    received no lock would be indistinguishable from a run that held one. The limitation
    is that this is in-process only: several replicas need a distributed lock, which is
    why horizontal scale-out is on the deferred list rather than claimed.
    """
    toolset: ResolvedToolset
    tool_context: ToolContext
    journal: Journal
    run_id: str
    session_id: str
    memory: Retriever | None = None
    """None when this configuration has no experience store."""
    assignment: str = ""
    """A brief the gated second pass hands this run, rendered into the role's own
    template. Empty on every ordinary run: only that mechanism has anything to assign,
    and a role that invented one would be second-guessing itself."""
    run_label: str = ""
    """Which of a mechanism's several runs of one role this is, for the record.

    Empty on an ordinary run, where the role name is the whole answer. The gated second
    pass runs the diagnosing role twice — once retrieval-free, once with precedents — and
    both rows say `rca`, so without this nothing downstream can tell the first opinion
    from the second, which is the comparison that arm exists to make."""
    resume: list[Message] | None = None
    """The transcript this role was paused in, with the decided call already answered.

    A role builds its own opening turn on a first entry. On a continuation there is
    already a conversation, and starting over would discard everything the role had
    established before it asked for permission.
    """
    resume_stage: str = ""
    """The stage label the pause was recorded under, when `resume` is set.

    A role that answers more than one graded question labels each phase `role.phase`,
    and a continuation has to re-enter the phase that paused. The role alone knows what
    its labels mean, so the label travels with the transcript rather than being resolved
    for it.
    """
    packet: LearningPacket | None = None
    """The offline pass's input, assembled from stored records rather than live state.

    Learning runs minutes after the incident closed, from a scheduler job on a fresh
    task, so there is no live state to read: what it gets is what was written down. The
    other roles leave this None.
    """


@dataclass(frozen=True, slots=True)
class RoleOutcome:
    """What a role hands back. The wrapper turns this into a state delta."""

    artifact: BaseModel | None
    """The validated submission, or None when the role did not produce one."""
    transcript: list[Message]
    error: str | None = None
    """The loop's terminal reason, when it ended without submitting."""
    delta: StateDelta = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    """Persisted on the run row rather than shown to any later role."""


RoleFn = Callable[[RoleContext], Awaitable[RoleOutcome]]


@dataclass(frozen=True, slots=True)
class RoleSpec:
    """Everything the orchestration layer needs to know about a role."""

    name: str
    submit_tool: str
    submit_model: type[BaseModel]
    state_slot: str | None
    """Which artifact slot this role fills, or None when it fills none."""
    prompt: str
    """Asset path under ``prompts/roles``."""
    run: RoleFn
    serial_tool_calls: bool = False
    """Run a turn's calls one at a time. Required wherever calls mutate a cluster or
    probe a live request path, since concurrent writes to one namespace are not ordered."""
    failure_sets_error: bool = True
    """Whether failing to submit halts the incident.

    An asymmetry the prototype had in code and nowhere in prose, made declarative here.
    The diagnostic and remediating roles set the incident's error when they cannot
    submit, so the outcome resolves against them. Verification does not: a check that
    could not run must not invalidate a remediation that applied cleanly, so it returns
    no result and the incident is decided on the remediation's own claim.
    """
