"""The SREGym environment adapter. Installed with the ``sregym`` extra.

One episode is one benchmark problem: the adapter takes it from the conductor, builds a
blinded incident signal out of the application's own metadata, supplies the transport
the tools speak, hands the final answers over for grading once the pipeline has settled,
and releases the held environment for teardown.

This module is deliberately import-light. The entry point that registers it is present
in every install, including one without the extra, so nothing here may import the client
library; that import lives in :mod:`kira.adapters.sregym.transport` and happens when an
episode actually starts, where a missing extra is a clean refusal naming what to install.

Two of the configuration values are oracle-grounded upper-bound ablation arms rather
than the deployable system. A ``feedback_mode`` other than ``none`` feeds the benchmark's
own grades to the distilling pass, so a memory corpus built under it is grounded in an
oracle no real deployment has.
"""

import asyncio
import logging
import os
from collections.abc import Mapping, Sequence
from importlib import import_module
from pathlib import Path
from typing import Any, Literal

from pydantic import AnyHttpUrl, BaseModel, ConfigDict

from kira.adapters.base import (
    EXTERNAL_KEY,
    AdapterTool,
    EnvironmentAdapter,
    EpisodeContext,
    ExternalFeedback,
    RoleOverlay,
    SubmissionRecord,
)
from kira.adapters.registry import adapter
from kira.adapters.sregym.conductor import Conductor
from kira.adapters.sregym.episode import reduce_feedback, submit_stage, wait_for_stage
from kira.adapters.sregym.evidence import EvidenceFilter
from kira.adapters.sregym.signal import build_diagnosis_solution, build_signal, problem_id
from kira.adapters.sregym.tools import adapter_tools
from kira.contracts.state import IncidentState
from kira.memory.distil import label_name
from kira.memory.ports import TextSearch
from kira.settings.root import Settings
from kira.tools.context import EvidencePolicy
from kira.tools.kubectl_proxy import ProxyRoute
from kira.tools.transport.base import ToolTransport

__all__ = ["ADAPTER_NAME", "MCP_KUBECTL_TOOLS", "SREGymAdapter", "SregymConfig"]

_log = logging.getLogger(__name__)

ADAPTER_NAME = "sregym"

DIAGNOSIS = "diagnosis"
MITIGATION = "mitigation"
ACCEPT = "accept"

_MISSING_EXTRA = "the sregym adapter needs its MCP transport: install kira[sregym]"

MCP_KUBECTL_TOOLS = frozenset(
    {"exec_kubectl_cmd_safely", "rollback_command", "get_previous_rollbackable_cmd"}
)
"""The tools that reach the cluster through this harness's own kubectl server.

Withheld together because they are one mechanism: the rollback pair replays the journal
that server keeps of what the kubectl tool applied, so offering either without it leaves
a tool whose whole subject is a history nothing is writing.
"""


class SregymConfig(BaseModel):
    """The adapter's own slice of ``Settings.adapters``.

    Every default is the one the archive was measured under. Unknown keys are refused
    rather than ignored, which is what makes a typo inside the slice a startup failure
    instead of a silent default that changes what an arm means.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    conductor_base_url: AnyHttpUrl = AnyHttpUrl("http://localhost:8000")
    mcp_base_url: AnyHttpUrl = AnyHttpUrl("http://localhost:9954")
    kubectl_proxy_url: AnyHttpUrl = AnyHttpUrl("http://127.0.0.1:16444")
    """Where a local kubectl reaches the cluster, credential-free.

    The harness fronts its API server with a filtering proxy and expects a real kubectl
    to be run against it. That is how every archived cell reached the cluster, for the
    agent's own reads and for the model's kubectl inside a shell alike, so it is the
    route rather than an option: the MCP kubectl server truncates, and a truncated
    ``-o json`` is unparseable without saying so."""
    kubectl_proxy_required: bool = True
    """Whether a shell's kubectl waits for the proxy before running anything.

    The proxy is a port-forward the harness restarts between problems. Off, a command
    that raced the restart reports a connection refused the model cannot act on."""
    kubectl_proxy_check_timeout_s: float = 5.0
    """How long one readiness probe of the proxy may take."""
    kubectl_proxy_ready_timeout_s: float = 60.0
    """How long the readiness probing continues before a command is refused."""
    timeout_s: float = 120.0
    conductor_timeout_s: float = 30.0
    stage_wait_timeout_s: float = 600.0
    stage_poll_s: float = 5.0
    feedback_timeout_s: float = 120.0
    feedback_poll_s: float = 3.0
    feedback_mode: Literal["none", "scores", "details"] = "none"
    """Whether the benchmark's own grades reach the distilling pass. Anything but
    ``none`` is an oracle-grounded upper-bound arm, not the deployable system."""
    problem_id: str | None = None
    """The runner's own name for this problem, used only when the conductor no longer
    serves one. Storage-only: it is the ground-truth fault label."""
    mcp_kubectl_tools: bool = False
    """Whether roles reach the cluster through the harness's own kubectl server.

    Off, as every measured campaign ran it: those runs typed kubectl into the sandbox,
    where it resolved to a stage-aware shim against the filtering proxy, and the three
    tools this switch governs were withheld from every role. The transport still serves
    ``kubernetes`` with this off — the pipeline's own reads, the namespace snapshot and
    the trace inventory among them, go through it and are not the model's calls."""
    leave_one_out: bool = True
    """Whether an episode withholds the corpus records of its own problem.

    The published protocol, and the default: the exact problem is withheld and its
    cross-application twins stay retrievable, so a cell is scored on what generalises
    rather than on having seen this problem before. Off retrieves the whole corpus,
    which is a measurement of something else."""

    @classmethod
    def resolve(cls, settings: Settings, env: Mapping[str, str] | None = None) -> "SregymConfig":
        """Build the configuration, honouring the names a prototype deployment uses.

        The legacy names are read here rather than by the settings layer because two of
        them mean something else there. ``API_HOSTNAME`` and ``API_PORT`` are this
        service's own bind address in a modern deployment file and the *conductor's*
        address in the harness's, and honouring both readings in one place made the
        service bind the conductor's port while the harness posted elsewhere and the run
        failed as a health-check timeout naming nothing.

        Args:
            settings: The resolved settings, whose ``adapters`` slice takes precedence.
            env: Environment to read legacy names from. Defaults to the process's.

        Returns:
            The validated configuration.

        Raises:
            pydantic.ValidationError: When the slice carries an unknown or malformed key.
        """
        environ = os.environ if env is None else env
        values = dict(settings.adapters.get(ADAPTER_NAME, {}))
        for field, value in _legacy_values(environ).items():
            values.setdefault(field, value)
        return cls.model_validate(values)


class SREGymAdapter:
    """One benchmark episode, from taking the problem to releasing the environment."""

    name = ADAPTER_NAME

    def __init__(
        self,
        config: SregymConfig,
        *,
        kubectl_binary: str = "kubectl",
        kubectl_timeout_s: int = 120,
        workdir: Path = Path(".local-workspace"),
    ) -> None:
        """Build an adapter for one episode.

        The three settings beyond the adapter's own slice are passed in rather than read
        here, because they are the deployment's and not this environment's: the same
        kubectl binary and the same ceiling apply whatever environment is bound. The
        defaults match the shipped settings, so an adapter built by hand behaves as a
        deployment that changed nothing would.

        Args:
            config: This deployment's adapter configuration.
            kubectl_binary: The real kubectl the proxy route runs.
            kubectl_timeout_s: How long one cluster command may take.
            workdir: The deployment's working directory, which holds the generated
                kubeconfig.
        """
        self._config = config
        self._kubectl = ProxyRoute(
            url=str(config.kubectl_proxy_url).rstrip("/"),
            required=config.kubectl_proxy_required,
            check_timeout_s=config.kubectl_proxy_check_timeout_s,
            ready_timeout_s=config.kubectl_proxy_ready_timeout_s,
            binary=kubectl_binary,
            exec_timeout_s=float(kubectl_timeout_s),
        )
        self._kubeconfig_dir = workdir / ".kubectl-proxy"
        self._conductor = Conductor(
            str(config.conductor_base_url), timeout_s=config.conductor_timeout_s
        )
        self._transport: Any = None
        self._signal: dict[str, Any] = {}
        self._session_id = ""
        self._released = False

    async def acquire(self) -> EpisodeContext:
        """Take the running problem and read what the agent is allowed to know about it.

        The problem blob is tolerated as absent: newer conductors dropped that route and
        answer 404, and it reaches only this adapter's logging and the storage-only
        episode key. The application metadata is not optional, because it is the only
        source for the signal the pipeline runs on.

        Returns:
            The episode's context.

        Raises:
            ValueError: When the transport's extra is not installed. Nothing has
                started, so this is a refusal rather than an abandoned environment.
            httpx.HTTPError: When the conductor could not describe the application.
        """
        self._transport = _build_transport(
            self._config, proxy=self._kubectl, kubeconfig_dir=self._kubeconfig_dir
        )
        self._transport.session_id = self._session_id
        problem = await self._conductor.get_problem()
        app = await self._conductor.get_app()
        # Manufactured once, here, where the metadata is. One blinding seam is the whole
        # mechanism: rebuilding the signal anywhere else is a second place for something
        # problem-derived to get in.
        self._signal = build_signal(problem, app)
        return EpisodeContext(
            namespace=self._signal.get("namespace"),
            app_name=self._signal.get("app_name"),
            app_description=self._signal.get("description"),
            target_stage="full",
            submission_kinds=(DIAGNOSIS, MITIGATION),
            # The fallback matters. A conductor that no longer serves the problem route
            # otherwise stores a null here, leaving nothing to join a session to the
            # grade it earned except a time window, which mis-mapped five sessions of
            # six in one measured case.
            external_key=problem_id(problem) or self._config.problem_id,
            raw=problem,
        )

    def bind_session(self, session_id: str) -> None:
        """Fix the correlation id this episode is known by, on both connections.

        Args:
            session_id: The incident session this episode runs as.
        """
        self._session_id = session_id
        self._conductor.session_id = session_id
        if self._transport is not None:
            self._transport.session_id = session_id

    def build_signal(self, ctx: EpisodeContext) -> dict[str, Any]:
        """Hand out the blinded intake trigger this episode's pipeline runs on.

        Args:
            ctx: What :meth:`acquire` reported. Unused: the signal was built there, from
                the application metadata, and nothing else may add to it.

        Returns:
            A copy of the signal.
        """
        return dict(self._signal)

    async def await_ready(self, kind: str) -> bool:
        """Wait for the grader to reach the stage this answer belongs to.

        Args:
            kind: The submission about to be sent.

        Returns:
            True when the stage came live.
        """
        return await wait_for_stage(
            self._conductor,
            kind,
            timeout_s=self._config.stage_wait_timeout_s,
            poll_s=self._config.stage_poll_s,
        )

    def encode(self, kind: str, state: IncidentState, ctx: EpisodeContext) -> str:
        """Render one answer as the harness's wire value.

        The remediation's grade comes from the harness's own live cluster-state oracle
        and the text is ignored, so it carries none.

        Args:
            kind: Which submission.
            state: The incident state as the pipeline left it.
            ctx: What :meth:`acquire` reported.

        Returns:
            The value to submit.
        """
        if kind == DIAGNOSIS:
            return build_diagnosis_solution(state.get("root_cause"))
        return ""

    async def submit(self, kind: str, payload: Any) -> SubmissionRecord:
        """Post one answer for grading.

        Args:
            kind: Which submission.
            payload: The text to grade.

        Returns:
            What happened, including a failure. Never raises.
        """
        return await submit_stage(self._conductor, str(payload), kind)

    def holds_environment_for_grading(self) -> bool:
        """Report that the environment stays up until it is released.

        Returns:
            True. The conductor holds the deployed application between the last
            submission and the release, and that window is what lets the distilling pass
            inspect what the remediation actually did.
        """
        return True

    async def fetch_feedback(self) -> ExternalFeedback | None:
        """Wait out the grading, then read the verdicts this arm is allowed to see.

        The wait happens in every mode, including the one that reads nothing: reaching
        the release hold is what says the grades have landed and the environment is
        still up, which is the state the distilling pass runs against.

        Returns:
            The grades, or None when this arm reads none, the conductor serves none, or
            grading did not finish in time.
        """
        try:
            held = await wait_for_stage(
                self._conductor,
                ACCEPT,
                timeout_s=self._config.stage_wait_timeout_s,
                poll_s=self._config.stage_poll_s,
            )
        except Exception:
            _log.exception("waiting for the release hold failed")
            return None
        if not held:
            _log.error("episode never reached the release hold; grading is incomplete")
            return None
        if self._config.feedback_mode == "none":
            return None
        return await self._poll_results()

    async def release(self, *, state: IncidentState | None) -> None:
        """Release the held environment for teardown.

        Not modelled as a third submission, though it is posted like one. It has to run
        in a ``finally`` after the distilling pass while the submissions run before it,
        and conflating the two once posted this while the conductor was still parked on
        an earlier stage, swallowed the rejection, and tore down with empty results.

        Args:
            state: Unused. The harness grades the cluster, not what we say about it.
        """
        if self._released:
            return
        self._released = True
        record = await submit_stage(self._conductor, "", ACCEPT)
        if record.error:
            _log.error("releasing the environment failed: %s", record.error)
        try:
            status = await self._conductor.status()
        except Exception as exc:
            _log.warning("could not read the conductor's status after release: %s", exc)
            return
        _log.info("episode %s released; conductor reports %s", self._session_id, status)

    def withheld_records(self, ctx: EpisodeContext, records: TextSearch) -> Sequence[str]:
        """Withhold every record this problem has already produced.

        Leave-one-out, as the campaign pre-registered it: the exact problem is withheld
        and its cross-application twins are not, so what a cell retrieves is a precedent
        from a different incident rather than its own answer. The archive applied this
        per cell from the harness, by handing the process a list of session ids before
        it started; here it is derived per episode from the problem the conductor served,
        which is the same set without a list anybody has to keep in step.

        The records name their own problem because the distilling pass copies the
        signal's underscore-prefixed keys onto them as labels, and the driver writes this
        episode's key under one of those.

        Args:
            ctx: What :meth:`acquire` reported.
            records: The corpus.

        Returns:
            The session ids of every record from this problem, and nothing else. Empty
            when the protocol is off or the conductor served no problem to key on.
        """
        if not self._config.leave_one_out:
            return ()
        if not ctx.external_key:
            # Nothing to key on and nothing withheld. Loud, because on a generalisation
            # arm this is the difference between a cell and a void one.
            _log.error(
                "leave-one-out is on and this episode has no problem id: nothing can be "
                "withheld, and a memory arm's cell is void"
            )
            return ()
        withheld = records.session_ids_with_label(label_name(EXTERNAL_KEY), ctx.external_key)
        _log.info("withholding %d record(s) from this episode's own problem", len(withheld))
        return withheld

    def tool_transport(self) -> ToolTransport | None:
        """Supply the transport built for this episode.

        Returns:
            The MCP transport, already carrying this episode's correlation id.
        """
        transport: ToolTransport | None = self._transport
        return transport

    def evidence_policy(self) -> EvidencePolicy | None:
        """Supply the filter that keeps the harness's own machinery out of evidence.

        Returns:
            The hide-list mirror.
        """
        return EvidenceFilter()

    def extra_tools(self) -> tuple[AdapterTool, ...]:
        """Supply the tools that exist only while this harness holds the session state.

        Returns:
            The rollback pair.
        """
        return adapter_tools()

    def role_overlay(self, role: str, state: IncidentState) -> RoleOverlay | None:
        """Withhold the harness's kubectl tools, unless this deployment asked for them.

        The benchmark asks the two questions the pipeline was written to answer, so the
        prompts and the budgets are left as each role built them. What is narrowed is
        the route to the cluster: with the kubectl server off, roles run kubectl in the
        sandbox against the filtering proxy, and leaving these three offered would give
        the model a second route that answers every call with a refusal — which reads to
        it as a fault to work around rather than as a tool it does not have.

        Every role, not just the writing ones: the diagnostic roles hold the kubectl
        tool to read with, and a read through a server this deployment is not using
        fails exactly as a write does.

        Args:
            role: The role about to run. The same three tools go for all of them.
            state: The incident as it stands.

        Returns:
            The withholding, or None when the deployment enabled the kubectl server.
        """
        if self._config.mcp_kubectl_tools:
            return None
        return RoleOverlay(drop_tools=MCP_KUBECTL_TOOLS)

    def routes(self) -> Any | None:
        """Supply the episode routes the external driver posts to.

        Returns:
            The router. Imported by name rather than at module scope so the web
            framework is not pulled in by anything that only wants the signal builder.
        """
        return import_module("kira.adapters.sregym.api").router

    async def _poll_results(self) -> ExternalFeedback | None:
        """Poll until both verdicts are in, or give up quietly.

        Grading is asynchronous on the conductor's side, so the two halves rarely land
        at once. Best effort throughout: feedback must never delay the release, and an
        episode without it is one the distilling pass reasons about from its own record.
        """
        details = self._config.feedback_mode == "details"
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._config.feedback_timeout_s
        results: Any = None
        while loop.time() < deadline:
            try:
                results = await self._conductor.results(include_ground_truth=details)
            except Exception as exc:
                _log.warning("reading the grades failed, will retry: %s", exc)
                results = None
            if isinstance(results, dict) and "Diagnosis" in results and "Mitigation" in results:
                break
            await asyncio.sleep(self._config.feedback_poll_s)

        feedback = reduce_feedback(results, details=details)
        if feedback is None:
            _log.warning(
                "feedback mode %r asked for grades and the conductor served none; the "
                "distilling pass runs without them",
                self._config.feedback_mode,
            )
        return feedback


def _build_transport(config: SregymConfig, *, proxy: ProxyRoute, kubeconfig_dir: Path) -> Any:
    """Load the MCP transport, which is where the optional extra is actually needed.

    Imported by name rather than with a statement because the module must not sit on the
    import path of an install without the extra, and a function-local import statement
    is a lint error in this repository.

    Args:
        config: The adapter's configuration.
        proxy: How a local kubectl reaches the cluster.
        kubeconfig_dir: Where the generated kubeconfig is written.

    Returns:
        A transport for one episode.

    Raises:
        ValueError: When the extra is not installed, naming what to install.
    """
    try:
        module = import_module("kira.adapters.sregym.transport")
    except ImportError as exc:
        raise ValueError(f"{_MISSING_EXTRA} ({exc})") from exc
    return module.McpTransport(
        str(config.mcp_base_url),
        timeout_s=config.timeout_s,
        proxy=proxy,
        kubeconfig_dir=kubeconfig_dir,
    )


def _legacy_values(env: Mapping[str, str]) -> dict[str, Any]:
    """Read the prototype's own names for this adapter's settings.

    Each is reported once, like every other legacy name, so an operator can see which
    spelling decided a value.

    Args:
        env: The environment to read.

    Returns:
        Field name to value, for the names that are actually set.
    """
    found: dict[str, Any] = {}
    for name, field in (
        ("MCP_TIMEOUT_SECONDS", "timeout_s"),
        ("MCP_KUBECTL_TOOLS_ENABLED", "mcp_kubectl_tools"),
        ("KUBECTL_PROXY_URL", "kubectl_proxy_url"),
        ("KUBECTL_PROXY_REQUIRED", "kubectl_proxy_required"),
        ("KUBECTL_PROXY_CHECK_TIMEOUT_SECONDS", "kubectl_proxy_check_timeout_s"),
        ("KUBECTL_PROXY_READY_TIMEOUT_SECONDS", "kubectl_proxy_ready_timeout_s"),
        ("SREGYM_FEEDBACK_MODE", "feedback_mode"),
        ("SREGYM_PROBLEM_ID", "problem_id"),
        ("MCP_SERVER_URL", "mcp_base_url"),
        ("API_BASE_URL", "conductor_base_url"),
    ):
        value = env.get(name, "").strip()
        if value:
            found[field] = value
            _log.info("legacy name in use: %s -> adapters.sregym.%s", name, field)

    host = env.get("API_HOSTNAME", "").strip()
    if host:
        found.setdefault("conductor_base_url", f"http://{host}:{env.get('API_PORT', '8000')}")
        found.setdefault("mcp_base_url", f"http://{host}:{env.get('MCP_SERVER_PORT', '9954')}")
        _log.info("legacy name in use: API_HOSTNAME -> adapters.sregym.conductor_base_url")
    return found


@adapter(ADAPTER_NAME, legacy_switch="MCP_ENABLED")
def _factory(settings: Settings) -> EnvironmentAdapter:
    """Build one episode's adapter from the deployment's settings."""
    return SREGymAdapter(
        SregymConfig.resolve(settings),
        kubectl_binary=settings.transport.kubectl_binary,
        kubectl_timeout_s=settings.tools.kubectl_timeout_s,
        workdir=settings.server.workdir,
    )
