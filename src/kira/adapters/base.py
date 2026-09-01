"""The environment adapter protocol and its payload types.

An adapter owns everything that knows an *episode* exists: how a problem is taken and
given back, how an incident signal is manufactured from it, in what encoding an answer
is handed out for grading, how grades come back, and which transport the cluster tools
speak. The core owns everything that knows an *incident* exists, and nothing more.

That line is the load-bearing structural claim of this system. The prototype crossed it
in one place, inside the function every cluster and telemetry tool called, and adding a
second harness meant editing the transport every module depended on. Here the core
cannot name an environment at all: an adapter supplies capability, and the core takes
what it is given.

This module is contract, not implementation. It is the one module under
``kira.adapters`` that core code may import; everything under ``kira/adapters/<name>/``
is banned from core imports by the ``TID251`` rule in ``pyproject.toml``.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from kira.contracts.state import IncidentState, TargetStage
from kira.tools.context import EvidencePolicy
from kira.tools.registry import ToolSpec
from kira.tools.transport.base import ToolTransport

__all__ = [
    "AdapterTool",
    "EnvironmentAdapter",
    "EpisodeContext",
    "ExternalFeedback",
    "SubmissionRecord",
]


@dataclass(frozen=True, slots=True)
class EpisodeContext:
    """Everything the adapter learned about one episode before the pipeline starts.

    Named for an episode rather than a problem because the core carries no benchmark
    noun, and an adapter that opens one episode per alert firing reads correctly.
    """

    namespace: str | None
    app_name: str | None = None
    app_description: str | None = None
    observation_window_start: str | None = None
    target_stage: TargetStage = "full"
    submission_kinds: tuple[str, ...] = ()
    external_key: str | None = None
    """The adapter's own identifier for this episode, and storage-only.

    For an evaluation harness it is usually the problem id, which *is* the ground-truth
    fault label, so one leak into a prompt invalidates every comparison with a blind
    baseline. The driver writes it under an underscore-prefixed key and the prompt
    builders drop every such key, which is what makes the blinding structural rather
    than a convention each new role has to remember.
    """
    raw: Any = None
    """The adapter's own blob, for its own logging. Never rendered, never persisted."""


@dataclass(frozen=True, slots=True)
class SubmissionRecord:
    """The outcome of one submission attempt. Never raised, always recorded."""

    kind: str
    payload: Any
    response: Any = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ExternalFeedback:
    """An external source of truth about one session's outcome.

    The distilling pass treats this as ground truth and it supersedes the system's own
    verdict, so it exists only where an environment has an oracle. That makes any
    configuration that supplies it an upper-bound ablation arm rather than the
    deployable system.
    """

    source: str
    diagnosis: Mapping[str, Any] | None = None
    mitigation: Mapping[str, Any] | None = None
    ground_truth_root_cause: str | None = None

    def to_packet(self) -> dict[str, Any]:
        """Render the feedback in the shape the distilling packet carries.

        ``LearningPacket.external_feedback`` is an untyped mapping, so a renamed key
        degrades the lesson silently rather than failing. The shape is pinned by a test
        against the prototype's for that reason.

        Returns:
            The packet's ``external_feedback`` value.
        """
        packet: dict[str, Any] = {"source": self.source}
        if self.diagnosis is not None:
            packet["diagnosis"] = dict(self.diagnosis)
        if self.mitigation is not None:
            packet["mitigation"] = dict(self.mitigation)
        if self.ground_truth_root_cause is not None:
            packet["ground_truth_root_cause"] = self.ground_truth_root_cause
        return packet


@dataclass(frozen=True, slots=True)
class AdapterTool:
    """A tool that exists only under one environment, with the roles allowed to see it.

    The core's role allowlists must not name an environment's tool, so an
    adapter-supplied tool carries its own role set and its own descriptions rather than
    being added to the shipped catalogue.
    """

    spec: ToolSpec
    roles: frozenset[str]


@runtime_checkable
class EnvironmentAdapter(Protocol):
    """Binds Kira to one incident environment for the lifetime of one episode.

    An instance is created per episode by the adapter's factory and is not reused.
    Nothing here may raise into the episode driver except :meth:`acquire`: an episode
    that never started is a clean refusal, but one that has started must always reach
    :meth:`release`.
    """

    name: str

    async def acquire(self) -> EpisodeContext:
        """Take ownership of one episode and report what is known about it.

        Returns:
            What the pipeline needs to run against this episode.
        """
        ...

    def bind_session(self, session_id: str) -> None:
        """Bind this episode to the incident session the core just created.

        Called before any role runs. Everything the adapter correlates by session id is
        fixed here and stays stable for the episode's whole lifetime. The prototype did
        this with a context variable, whose propagation into tasks was the mechanism
        that covered calls fanned out with ``asyncio.gather``; a per-episode instance
        holding the id gives the same property structurally.

        Args:
            session_id: The incident session this episode runs as.
        """
        ...

    async def release(self, *, state: IncidentState | None) -> None:
        """Hand the environment back.

        Must be idempotent, must not raise, and the driver calls it from a ``finally``:
        a slow or crashed distilling pass must never strand a deployed application.

        Args:
            state: The incident state as the pipeline left it, or None when it never
                produced one.
        """
        ...

    def build_signal(self, ctx: EpisodeContext) -> dict[str, Any]:
        """Manufacture the blinded intake trigger this episode's pipeline runs on.

        Must carry no ground-truth fault label. Anything stored for later analysis goes
        under an underscore-prefixed key, which no prompt builder can render.

        Args:
            ctx: What :meth:`acquire` reported.

        Returns:
            The signal, in the shape the generic intake takes.
        """
        ...

    async def await_ready(self, kind: str) -> bool:
        """Block until the harness will accept a submission of this kind.

        Args:
            kind: The submission kind about to be sent.

        Returns:
            True when it will be graded. False stops the plan, because firing at a
            stage machine that will reject it leaves the answer silently ungraded.
        """
        ...

    def encode(self, kind: str, state: IncidentState, ctx: EpisodeContext) -> Any:
        """Map the final incident state to one harness wire value.

        Pure and synchronous, so it is testable without a harness. It must always
        produce a value: a halted or absent answer is a *failed* answer and must still
        be submitted, or the cell vanishes from the denominator and biases the rate.

        Args:
            kind: Which submission.
            state: The incident state as the pipeline left it.
            ctx: What :meth:`acquire` reported.

        Returns:
            The value to submit.
        """
        ...

    async def submit(self, kind: str, payload: Any) -> SubmissionRecord:
        """Make one submission available to the harness.

        Modelled as making available rather than posting, because an environment that
        drives the agent asks for the answer instead of receiving it.

        Args:
            kind: Which submission.
            payload: What :meth:`encode` produced.

        Returns:
            What happened, including a failure. Never raises.
        """
        ...

    def holds_environment_for_grading(self) -> bool:
        """Report whether the environment stays up between submission and release.

        Returns:
            True when the distilling pass can still inspect the post-remediation
            cluster and there is a grade to fetch before releasing.
        """
        ...

    async def fetch_feedback(self) -> ExternalFeedback | None:
        """Collect external grades, best effort.

        Must not block the release and must not raise.

        Returns:
            The grades, or None when unsupported, ungraded, or timed out.
        """
        ...

    def tool_transport(self) -> ToolTransport | None:
        """Supply the transport this episode's tools speak.

        Returns:
            A transport built for this episode, carrying its own correlation id, or
            None to keep the deployment's configured default.
        """
        ...

    def evidence_policy(self) -> EvidencePolicy | None:
        """Supply a last-mile filter over what tools return.

        Returns:
            The policy, or None for the no-op default.
        """
        ...

    def extra_tools(self) -> tuple[AdapterTool, ...]:
        """Supply tools that exist only under this environment.

        Returns:
            The tools and the roles allowed to see them.
        """
        ...

    def routes(self) -> Any | None:
        """Supply an optional router mounted under the API prefix.

        This is how a path naming an environment stays out of the core.

        Returns:
            A ``fastapi.APIRouter``, or None. None is normal.
        """
        ...
