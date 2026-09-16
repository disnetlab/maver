"""The agent half of an AIOpsLab run.

AIOpsLab is the other way round from a harness we drive. It imports its own orchestrator
in its own process, deploys the problem, injects the fault, and then pulls exactly one
action back out of us: the driver posts one incident to the generic intake, polls it, and
encodes the answer into what its orchestrator returns. Nothing here opens an episode,
because the environment is held over there.

So this class implements the capability half of the protocol and refuses the lifecycle
half. It is bound once for the process — it mounts no episode route, which is what says
so — and every incident that arrives through the ordinary intake then runs on the
bridge's transport, with the tools and the role overlays this environment supplies.
"""

from collections.abc import Sequence
from importlib import import_module
from typing import Any

from maver.adapters.aiopslab.config import ADAPTER_NAME, AiopsLabConfig
from maver.adapters.aiopslab.overlay import BENCHMARK, role_overlay
from maver.adapters.aiopslab.tools import adapter_tools
from maver.adapters.base import (
    AdapterTool,
    EpisodeContext,
    ExternalFeedback,
    LifecycleRunsElsewhere,
    RoleOverlay,
    SubmissionRecord,
)
from maver.contracts.state import IncidentState
from maver.memory.ports import TextSearch
from maver.tools.context import EvidencePolicy
from maver.tools.transport.base import ToolTransport

__all__ = ["AiopsLabAdapter"]

_ELSEWHERE = (
    "the aiopslab driver holds the environment: it deploys the problem, submits the "
    "answer and tears the application down in its own process, and this one runs the "
    "pipeline alone"
)


class AiopsLabAdapter:
    """Supplies what a role needs to work through the driver's bridge."""

    name = ADAPTER_NAME

    def __init__(self, config: AiopsLabConfig, *, authority_mode: str = "prompt") -> None:
        """Build the process-wide adapter.

        Args:
            config: This deployment's adapter configuration.
            authority_mode: How this deployment decides a cluster write. Read once, at
                composition, because it decides which of two approval sentences the
                remediating role is given — and a role told to wait for a human on an
                unattended run spends its budget waiting for nobody.
        """
        self._config = config
        self._authority_mode = authority_mode
        self._tools = adapter_tools(config)
        self._transport: Any = None

    @property
    def config(self) -> AiopsLabConfig:
        """The resolved configuration, for the halves that read it."""
        return self._config

    def tool_transport(self) -> ToolTransport | None:
        """Supply the transport every tool in this process speaks.

        Deliberately less capable than the default one: the agent holds no kubeconfig
        and no shell on the benchmark host, so every cluster read and write goes through
        the driver's scoped bridge and is screened there.

        Returns:
            The bridge transport.

        Raises:
            ValueError: When the adapter is bound without being switched on, or its
                transport module is not installed. Both are startup failures naming what
                to fix, because the alternative is a process that starts, hands every
                role an empty toolset and reports a model that could not investigate.
        """
        if not self._config.enabled:
            raise ValueError(
                "the aiopslab adapter is bound but adapters.aiopslab.enabled is false, "
                "so it has no bridge credential to speak through; set it, or drop "
                "aiopslab from intake.adapters"
            )
        try:
            module = import_module("maver.adapters.aiopslab.transport")
        except ImportError as exc:
            raise ValueError(f"the aiopslab bridge transport is not available: {exc}") from exc
        token = self._config.bridge_token
        transport: ToolTransport = module.AiopsLabTransport(
            self._config.bridge_url,
            token.get_secret_value() if token is not None else "",
            timeout_s=self._config.bridge_timeout_s,
        )
        # Held, because this is the same object every incident's tools speak through and
        # the overlay below has to point it at the incident it is about to serve.
        self._transport = transport
        return transport

    def evidence_policy(self) -> EvidencePolicy | None:
        """Supply no last-mile filter.

        Returns:
            None. AIOpsLab injects its faults through its own orchestrator rather than
            through a controller that shows up in the telemetry it serves, so there is
            nothing to hide from the evidence.
        """
        return None

    def extra_tools(self) -> tuple[AdapterTool, ...]:
        """Supply the tools that exist only against this bridge.

        Returns:
            The Service probe and the MongoDB recovery action, each carrying the one
            role allowed to see it.
        """
        return self._tools

    def role_overlay(self, role: str, state: IncidentState) -> RoleOverlay | None:
        """Point the bridge at this incident, then narrow the role to its graded task.

        The binding happens here because this adapter is bound for the whole process:
        one transport serves every incident, and the bridge refuses any action whose
        namespace is not the active problem's. AIOpsLab deploys one problem at a time,
        so pointing the transport at the incident whose role is about to run is correct
        for the deployment it supports, and a second concurrent incident fails closed at
        the bridge rather than reading across into another application's telemetry. The
        deployment is documented as ``max_concurrent_incidents=1`` for that reason.

        Args:
            role: The role about to run.
            state: The incident as it stands, which carries the task type.

        Returns:
            The overlay for this role and task, or None when the incident is not an
            AIOpsLab one — in which case nothing here is touched at all.
        """
        alert = state.get("alert") or {}
        if alert.get("benchmark") != BENCHMARK:
            return None
        if self._transport is not None:
            self._transport.namespace = str(alert.get("namespace") or "")
            self._transport.stage = role
            # Assigned here for the same reason: the transport is built at startup, and
            # the session it will run as is only known once an incident exists. Without
            # it the agent's audit events and the bridge's access log share nothing but
            # a wall clock, and a timestamp join mis-mapped five sessions of six once.
            self._transport.session_id = str(state.get("session_id") or "")
        return role_overlay(
            role,
            state,
            config=self._config,
            transport=self._transport,
            authority_mode=self._authority_mode,
        )

    def routes(self) -> Any | None:
        """Mount no episode route.

        Returns:
            None. The driver drives us, so there is no episode for anything here to
            open, and this adapter is bound for the whole process instead.
        """
        return None

    def withheld_records(self, ctx: EpisodeContext, records: TextSearch) -> Sequence[str]:
        """Withhold nothing.

        Args:
            ctx: Unused; there is no episode context in this process.
            records: Unused.

        Returns:
            Empty. A leave-one-out protocol needs an episode key, and this process is
            never given one.
        """
        return ()

    async def acquire(self) -> EpisodeContext:
        """Refuse: the driver takes the problem.

        Raises:
            LifecycleRunsElsewhere: Always.
        """
        raise LifecycleRunsElsewhere(_ELSEWHERE)

    def bind_session(self, session_id: str) -> None:
        """Refuse: the driver correlates the session it posted.

        Args:
            session_id: Unused.

        Raises:
            LifecycleRunsElsewhere: Always.
        """
        raise LifecycleRunsElsewhere(_ELSEWHERE)

    def build_signal(self, ctx: EpisodeContext) -> dict[str, Any]:
        """Refuse: the driver builds the blinded trigger it posts.

        Args:
            ctx: Unused.

        Raises:
            LifecycleRunsElsewhere: Always.
        """
        raise LifecycleRunsElsewhere(_ELSEWHERE)

    async def await_ready(self, kind: str) -> bool:
        """Refuse: there is no stage machine on this side.

        Args:
            kind: Unused.

        Raises:
            LifecycleRunsElsewhere: Always.
        """
        raise LifecycleRunsElsewhere(_ELSEWHERE)

    def encode(self, kind: str, state: IncidentState, ctx: EpisodeContext) -> Any:
        """Refuse: the driver encodes the answer its orchestrator asked for.

        Args:
            kind: Unused.
            state: Unused.
            ctx: Unused.

        Raises:
            LifecycleRunsElsewhere: Always.
        """
        raise LifecycleRunsElsewhere(_ELSEWHERE)

    async def submit(self, kind: str, payload: Any) -> SubmissionRecord:
        """Refuse: the answer is pulled out of the driver, not posted from here.

        Args:
            kind: Unused.
            payload: Unused.

        Raises:
            LifecycleRunsElsewhere: Always.
        """
        raise LifecycleRunsElsewhere(_ELSEWHERE)

    def holds_environment_for_grading(self) -> bool:
        """Refuse: whether the application stays up is the driver's to know.

        Raises:
            LifecycleRunsElsewhere: Always.
        """
        raise LifecycleRunsElsewhere(_ELSEWHERE)

    async def fetch_feedback(self) -> ExternalFeedback | None:
        """Refuse: the grade is returned to the orchestrator, not to us.

        Raises:
            LifecycleRunsElsewhere: Always.
        """
        raise LifecycleRunsElsewhere(_ELSEWHERE)

    async def release(self, *, state: IncidentState | None) -> None:
        """Refuse: the orchestrator tears its own application down.

        Args:
            state: Unused.

        Raises:
            LifecycleRunsElsewhere: Always.
        """
        raise LifecycleRunsElsewhere(_ELSEWHERE)
