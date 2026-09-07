"""One problem, start to graded finish: deploy, blind, delegate, submit.

AIOpsLab is the inverse of a harness that spawns us. We import its orchestrator, drive
it in this process, and it pulls exactly one action back out of the facade below. That
action is produced by posting a blinded incident to the agent over HTTP and waiting for
the run to finish, so the whole multi-agent pipeline sits inside a single
``get_action()`` call.

Everything the agent is told about the problem travels in the alert, and the problem id
is not in it. The one thing that does cross is a timestamp: the moment fault injection
began, recorded by wrapping a lifecycle method AIOpsLab already calls.

One problem leaves two records of its own beside AIOpsLab's result file: ``bridge.log``,
which is the credential-holding process's account of every cluster action, and
``kira-submission.json``, which is what was asked of the agent and what was submitted for
it. The second exists because the encoded answer is not always the agent's own: the
analysis mapper's hand-written rules override the model's classification, and an accuracy
figure that cannot separate the two reports the rules' accuracy as the agent's.
"""

import argparse
import asyncio
import json
import logging
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError

from .bridge import BridgeBackend, serve_bridge
from .mapping import (
    _analysis_from_root_cause,
    _is_terminal_record,
    _root_cause,
    build_alert,
    submission_action,
)

__all__ = [
    "BRIDGE_LOG_NAME",
    "SUBMISSION_RECORD_NAME",
    "AIOpsLabAgentAdapter",
    "instrument_observation_window",
    "kubernetes_namespace",
    "resolve_bridge_token",
    "run",
    "task_type",
    "utc_now",
]

log = logging.getLogger("aiopslab-driver")

_TASKS = ("detection", "localization", "analysis", "mitigation")
_MINIMUM_TOKEN_LENGTH = 16

BRIDGE_LOG_NAME = "bridge.log"
"""The host side's own account of every cluster action, in the problem's results dir."""

SUBMISSION_RECORD_NAME = "kira-submission.json"
"""What was asked and what was answered, in the problem's results dir.

Named here rather than where it is written because the sweep has to skip it when it looks
for the newest result file: it is written after AIOpsLab's, and grading it would read
`unknown` for a problem the harness has already graded.
"""


def utc_now() -> str:
    """Return UTC to the second, spelled the way the alert and both records spell it.

    Returns:
        An ISO-8601 timestamp ending in Z.
    """
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def instrument_observation_window(registry: Any, marker: dict[str, str]) -> None:
    """Record the neutral pre-injection/workload boundary without exposing an ID.

    AIOpsLab deploys the application before it invokes ``inject_fault``. Wrapping
    that generic lifecycle method gives the agent a timestamp for separating
    deployment startup noise from the evaluated observation window. No problem
    identifier, injector arguments, or expected answer crosses into the alert.

    Args:
        registry: AIOpsLab's problem registry.
        marker: Mapping to write ``started_at`` into.
    """
    original_get_problem = registry.get_problem_instance

    def instrumented_get_problem(problem_id: str) -> Any:
        problem = original_get_problem(problem_id)
        original_inject = problem.inject_fault

        def instrumented_inject() -> Any:
            marker["started_at"] = utc_now()
            return original_inject()

        problem.inject_fault = instrumented_inject
        return problem

    registry.get_problem_instance = instrumented_get_problem


def task_type(problem: Any) -> str:
    """Name the graded task from the problem's class hierarchy.

    Name matching against another project's classes, which is fragile and is the only
    handle AIOpsLab exposes. It fails loudly rather than defaulting, because a task type
    guessed wrong submits a well-formed answer to the wrong question.

    Args:
        problem: The instantiated AIOpsLab problem.

    Returns:
        One of detection, localization, analysis, mitigation.

    Raises:
        ValueError: If no task class is recognised.
    """
    for cls in type(problem).mro():
        name = cls.__name__.lower()
        for task in _TASKS:
            if name == f"{task}task":
                return task
    raise ValueError(f"cannot determine AIOpsLab task type from {type(problem).__name__}")


def kubernetes_namespace(problem: Any) -> str:
    """Return the namespace this problem deployed into, refusing Docker-backed ones.

    Rejected rather than skipped quietly: granting the Docker socket or a host shell to
    reach them would be an unsafe extension of the tool, RBAC and approval model the
    measurements were taken under, and the sweep records them as excluded so the
    denominator stays honest.

    Args:
        problem: The instantiated AIOpsLab problem.

    Returns:
        The active namespace.

    Raises:
        ValueError: If the problem is Docker-backed.
    """
    namespace = str(problem.namespace)
    if namespace == "docker":
        raise ValueError(
            "Docker-backed AIOpsLab problems are outside this Kubernetes agent's "
            "tool and safety model"
        )
    return namespace


def resolve_bridge_token(token: str, token_file: str | None) -> str:
    """Resolve the bearer token the bridge and the agent share.

    Failing here is the point: a token the agent cannot match disables every cluster
    tool it has, and a run that discovers that at the first ``kubectl`` call has already
    paid for a deployment and looks like a model failure in the results.

    Args:
        token: The value from ``--bridge-token`` or ``AIOPSLAB_BRIDGE_TOKEN``.
        token_file: A file to read it from instead.

    Returns:
        The token.

    Raises:
        ValueError: If neither was supplied, or the file is unreadable or too short to
            be a credential.
    """
    if token_file:
        path = Path(token_file)
        try:
            value = path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise ValueError(f"cannot read bridge token file {path}: {exc}") from exc
        if len(value) < _MINIMUM_TOKEN_LENGTH:
            raise ValueError(
                f"bridge token in {path} is {len(value)} characters; "
                f"at least {_MINIMUM_TOKEN_LENGTH} are required"
            )
        return value
    if not token:
        raise ValueError("a bridge token is required: set AIOPSLAB_BRIDGE_TOKEN or pass one")
    return token


def _get_json(url: str, *, timeout: float) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:  # loopback agent URL
        decoded: dict[str, Any] = json.loads(response.read())
        return decoded


def _post_json(url: str, payload: dict[str, Any], *, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # loopback agent URL
        decoded: dict[str, Any] = json.loads(response.read())
        return decoded


def check_agent_reachable(agent_url: str, *, timeout: float = 10) -> None:
    """Fail before AIOpsLab pays for a deployment the agent cannot answer for.

    Args:
        agent_url: The agent's base URL.
        timeout: Seconds to wait.

    Raises:
        RuntimeError: If the liveness probe does not answer.
    """
    url = f"{agent_url.rstrip('/')}/healthz"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:  # loopback agent URL
            response.read(256)
    except (HTTPError, URLError, OSError) as exc:
        raise RuntimeError(f"agent not reachable at {url}: {exc}") from exc


class AIOpsLabAgentAdapter:
    """AIOpsLab agent facade that delegates one whole task to the agent API."""

    def __init__(
        self,
        *,
        task_type: str,
        namespace: str,
        agent_url: str,
        timeout_seconds: float,
        observation_window_start: str | None = None,
        poll_seconds: float = 2.0,
    ) -> None:
        """Prepare the facade AIOpsLab will call once.

        Args:
            task_type: The graded task.
            namespace: The active problem namespace.
            agent_url: The agent's base URL.
            timeout_seconds: How long one incident may take.
            observation_window_start: When fault injection began, if observed.
            poll_seconds: Interval between incident reads.
        """
        self.task_type = task_type
        self.namespace = namespace
        self.agent_url = agent_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.observation_window_start = observation_window_start
        self.poll_seconds = poll_seconds
        self.system_message = "AIOpsLab adapter for kira"
        self.task_message = ""
        self.problem_description = ""
        self.instructions = ""
        self.actions: dict[str, str] = {}
        self.incident_id: str | None = None
        self.alert: dict[str, Any] | None = None
        self.incident_record: dict[str, Any] | None = None
        self.analysis_answer: dict[str, str | None] | None = None
        self.action: str | None = None

    def init_context(
        self,
        problem_description: str,
        instructions: str,
        actions: dict[str, str],
    ) -> None:
        """Receive the task framing AIOpsLab hands every agent.

        Args:
            problem_description: The task description, which becomes the alert's.
            instructions: AIOpsLab's own instructions, unused by the pipeline.
            actions: The actions AIOpsLab accepts, unused by the pipeline.
        """
        self.problem_description = problem_description
        self.instructions = instructions
        self.actions = actions
        self.task_message = problem_description

    async def get_action(self, _input: str) -> str:
        """Run the whole incident and return the one action AIOpsLab grades.

        Args:
            _input: AIOpsLab's turn input, unused: the pipeline is driven by the alert.

        Returns:
            The fenced ``submit(...)`` call.

        Raises:
            TimeoutError: If the incident does not reach a terminal record in time.
        """
        if self.action is not None:
            return self.action
        self.alert = build_alert(
            self.task_type,
            self.namespace,
            self.problem_description,
            observation_window_start=self.observation_window_start,
        )
        target_stage = "full" if self.task_type == "mitigation" else "diagnose"
        accepted = await asyncio.to_thread(
            _post_json,
            f"{self.agent_url}/api/v1/incidents",
            {"alert": self.alert, "target_stage": target_stage},
            timeout=30,
        )
        self.incident_id = str(accepted["session_id"])
        log.info(
            "incident %s started: task=%s stage=%s", self.incident_id, self.task_type, target_stage
        )

        awaiting = ""
        deadline = asyncio.get_running_loop().time() + self.timeout_seconds
        while True:
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError(
                    f"agent incident {self.incident_id} exceeded {self.timeout_seconds:.0f}s"
                )
            record = await asyncio.to_thread(
                _get_json, f"{self.agent_url}/api/v1/incidents/{self.incident_id}", timeout=30
            )
            if _is_terminal_record(record):
                self.incident_record = record
                if self.task_type == "analysis":
                    self.analysis_answer = _analysis_from_root_cause(_root_cause(record))
                self.action = submission_action(self.task_type, record)
                return self.action
            # A run parked on a person is neither finished nor progressing, and under
            # the prompt approval policy it can stay there for the rest of the timeout.
            # Said once per request rather than every poll, so the log stays readable.
            pending = str(record.get("awaiting_request_id") or "")
            if pending and pending != awaiting:
                log.info("incident %s is awaiting approval %s", self.incident_id, pending)
            awaiting = pending
            await asyncio.sleep(self.poll_seconds)


def write_submission_record(
    results_dir: Path,
    *,
    problem: str,
    task: str,
    namespace: str,
    adapter: "AIOpsLabAgentAdapter",
    started_utc: str,
) -> Path:
    """Record what was asked of the agent and what was submitted for it.

    Args:
        results_dir: The problem's results directory.
        problem: The AIOpsLab problem id, which stays on this side of the wire.
        task: The graded task type.
        namespace: The namespace the problem deployed into.
        adapter: The facade that ran the episode.
        started_utc: When the problem began.

    Returns:
        The path written.
    """
    record = {
        "problem": problem,
        "task_type": task,
        "namespace": namespace,
        "agent_incident_id": adapter.incident_id,
        "alert": adapter.alert,
        "incident_record": adapter.incident_record,
        "submitted_action": adapter.action,
        # Present for the analysis task alone, and carrying `source` and `rule`: the
        # answer AIOpsLab grades is not always the one the model wrote.
        "analysis_answer": adapter.analysis_answer,
        "started_utc": started_utc,
        "ended_utc": utc_now(),
    }
    path = results_dir / SUBMISSION_RECORD_NAME
    path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    return path


async def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run one AIOpsLab problem to its graded result.

    Args:
        args: The parsed command line.

    Returns:
        The AIOpsLab session, the incident id, the submitted action, and AIOpsLab's own
        outcome, for the sweep's per-problem record.

    Raises:
        ValueError: If the token is unusable or the problem is Docker-backed.
        RuntimeError: If the agent is not reachable.
    """
    # Both refusals come before AIOpsLab is even imported, because both are cheap and
    # what follows them is a cluster deployment that takes minutes to make and to undo.
    token = resolve_bridge_token(args.bridge_token, args.bridge_token_file)
    await asyncio.to_thread(check_agent_reachable, args.agent_url)

    # Imported only here so unit tests and the agent image do not need AIOpsLab.
    from aiopslab.orchestrator import Orchestrator  # type: ignore[import-not-found] # noqa: PLC0415

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    started_utc = utc_now()
    orchestrator = Orchestrator(results_dir=results_dir)
    placeholder = object()
    orchestrator.register_agent(placeholder, name="kira")
    observation_window: dict[str, str] = {}
    instrument_observation_window(orchestrator.probs, observation_window)
    problem_description, instructions, actions = orchestrator.init_problem(args.problem)
    problem = orchestrator.session.problem
    task = task_type(problem)
    namespace = kubernetes_namespace(problem)

    adapter = AIOpsLabAgentAdapter(
        task_type=task,
        namespace=namespace,
        agent_url=args.agent_url,
        timeout_seconds=args.agent_timeout,
        observation_window_start=observation_window.get("started_at"),
        poll_seconds=args.poll_seconds,
    )
    adapter.init_context(problem_description, instructions, actions)
    orchestrator.agent = adapter

    backend = BridgeBackend(
        namespace,
        allow_writes=task == "mitigation",
        prometheus_url=args.prometheus_url,
    )
    with serve_bridge(
        backend,
        token,
        host=args.bridge_host,
        port=args.bridge_port,
        access_log=results_dir / BRIDGE_LOG_NAME,
    ):
        outcome = await orchestrator.start_problem(max_steps=1)

    write_submission_record(
        results_dir,
        problem=args.problem,
        task=task,
        namespace=namespace,
        adapter=adapter,
        started_utc=started_utc,
    )
    return {
        "aiopslab_session": orchestrator.session.to_dict(),
        "agent_incident_id": adapter.incident_id,
        "submitted_action": adapter.action,
        "analysis_answer": adapter.analysis_answer,
        "orchestrator_outcome": outcome,
    }
