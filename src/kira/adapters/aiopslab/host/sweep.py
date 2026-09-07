"""Every selected problem, once, resumably: the bash sweep where it can be tested.

The prototype's sweep was 226 lines of shell around three ``poetry run python -c``
one-liners: one enumerated the registry, one inferred a task type, one graded a result
file. Those are the semantics of the experiment, and they were untestable where they sat.
They are functions here, and the shell keeps what a shell is good at.

Two rules carry over unchanged, because a resumed sweep has to behave like the archived
ones. **A result AIOpsLab graded is final**: it is never run again, not on a resume and
not after the runner exits nonzero cleaning up behind it, because a second run of a
graded problem replaces one cell with a second sample of the model and quietly improves
the arm. **Only a non-grade is retried** — a deployment failure, a host timeout, a crash —
at most ``--max-attempts`` times.

Problems are run one at a time and one bridge is served per problem. AIOpsLab mutates
cluster-wide state, so this is not a concurrency limitation to be optimised away; the
lock below exists to enforce it against a second sweep on the same host.
"""

import argparse
import asyncio
import fcntl
import json
import logging
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, NamedTuple

from . import lifecycle

__all__ = [
    "STATUS_COLUMNS",
    "TASK_TYPES",
    "Selected",
    "grade_of",
    "latest_result",
    "run_sweep",
    "select_problems",
    "task_type_for",
]

log = logging.getLogger("aiopslab-driver")

TASK_TYPES = ("detection", "localization", "analysis", "mitigation")
"""The four tasks AIOpsLab grades, in the order the registry lists them."""

STATUS_COLUMNS = (
    "index",
    "problem",
    "task_type",
    "attempts",
    "exit_code",
    "grade",
    "agent_incident_id",
    "started_utc",
    "ended_utc",
    "duration_seconds",
    "result_file",
)
"""One row per problem. A superset of the archived sweep's seven columns.

The four additions are the ones an archived row could not answer: which task was graded,
whether the grade came on the first attempt, which agent incident produced it, and how
long it took. The seven that were there keep their names and their meanings.
"""

_GRADED = frozenset({"pass", "fail"})
_HOST_TIMEOUT_EXIT = 124
"""``timeout(1)``'s own exit code, which is what the archived rows carry for this."""


class Selected(NamedTuple):
    """One problem this sweep will run, and the task it is graded on."""

    problem: str
    task_type: str


def task_type_for(problem_id: str) -> str:
    """Name the graded task from the problem id, before anything is deployed.

    The registry exposes no task type without instantiating a problem, and instantiating
    one to plan a sweep would deploy it. The naming rule is carried from the sweep script.

    Args:
        problem_id: An AIOpsLab problem id.

    Returns:
        One of detection, localization, analysis, mitigation.

    Raises:
        ValueError: If the id names no task. Loud rather than skipped: a problem dropped
            from the denominator for a naming change is a smaller benchmark reported as
            a completed one.
    """
    if problem_id.startswith("noop_detection"):
        return "detection"
    parts = problem_id.split("-")
    if len(parts) > 1 and parts[-1][:1].isdigit():
        parts = parts[:-1]
    task = parts[-1] if parts else ""
    if task in TASK_TYPES:
        return task
    raise ValueError(f"cannot infer the AIOpsLab task type from problem id {problem_id!r}")


def select_problems(
    task_types: Sequence[str], problems_file: Path | None = None
) -> tuple[list[Selected], list[str]]:
    """Choose what this sweep runs, and name what it will not.

    Args:
        task_types: The graded task types to include.
        problems_file: A file of problem ids, one per line, ``#`` starting a comment.
            When absent, AIOpsLab's own registry is read.

    Returns:
        The problems to run, in registry or file order, and the Docker-backed problem ids
        excluded from them. A file names what it names, so its excluded list is empty.

    Raises:
        ValueError: If a task type is not one AIOpsLab grades, if a problem id names no
            task, or if nothing is selected.
    """
    unsupported = [task for task in task_types if task not in TASK_TYPES]
    if unsupported:
        raise ValueError(f"unsupported AIOpsLab task types: {', '.join(unsupported)}")

    if problems_file is None:
        candidates, excluded = _registry_problems()
    else:
        candidates, excluded = _listed_problems(problems_file), []

    # Every id is resolved before any is filtered, so an unrecognised one fails the
    # sweep rather than silently dropping out of a task type nobody asked for.
    resolved = [Selected(problem, task_type_for(problem)) for problem in candidates]
    selected = [item for item in resolved if item.task_type in task_types]
    if not selected:
        raise ValueError("no AIOpsLab problem matches this selection")
    return selected, excluded


def _listed_problems(problems_file: Path) -> list[str]:
    lines = problems_file.read_text(encoding="utf-8").splitlines()
    return [
        stripped.split()[0]
        for line in lines
        if (stripped := line.strip()) and not stripped.startswith("#")
    ]


def _registry_problems() -> tuple[list[str], list[str]]:
    """Read the pinned registry, separating Kubernetes problems from Docker-backed ones.

    Docker-backed problems are outside this agent's tool and safety model, and reaching
    them would mean handing it the Docker socket or a host shell. They are named in the
    sweep's own record rather than dropped, so the denominator stays honest.
    """
    # Imported here for the reason lifecycle.run imports the orchestrator here: a laptop
    # running the unit tests, and the agent image, have no AIOpsLab.
    from aiopslab.orchestrator.problems.registry import (  # type: ignore[import-not-found]  # noqa: PLC0415
        ProblemRegistry,
    )

    registry = ProblemRegistry()
    kubernetes: list[str] = []
    docker: list[str] = []
    for problem_id in registry.get_problem_ids():
        target = kubernetes if registry.get_problem_deployment(problem_id) == "k8s" else docker
        target.append(problem_id)
    return kubernetes, docker


def latest_result(results_dir: Path) -> Path | None:
    """Find AIOpsLab's newest result file for one problem, if it wrote one.

    Args:
        results_dir: The problem's results directory.

    Returns:
        The newest result file, or None when the harness graded nothing.
    """
    candidates = [
        path
        for path in results_dir.glob("*.json")
        # Our own sidecar is written after the harness's file and would always be the
        # newest; grading it would read `unknown` for an already-graded problem.
        if path.is_file() and path.name != lifecycle.SUBMISSION_RECORD_NAME
    ]
    return max(candidates, key=lambda path: path.stat().st_mtime, default=None)


def grade_of(result_file: Path) -> str:
    """Read AIOpsLab's own verdict out of one result file.

    Args:
        result_file: A result file the harness wrote.

    Returns:
        ``pass``, ``fail``, or ``unknown`` for a file that records no verdict, which is
        what an interrupted or crashed problem leaves and is therefore retryable.
    """
    try:
        results = json.loads(result_file.read_text(encoding="utf-8")).get("results", {})
        value = results.get("success")
        if value is None and "Detection Accuracy" in results:
            value = results["Detection Accuracy"] == "Correct"
    except (OSError, ValueError, AttributeError):
        return "unknown"
    return "pass" if value is True else "fail" if value is False else "unknown"


class _Outcome(NamedTuple):
    attempts: int
    exit_code: int
    grade: str
    incident_id: str
    result_file: str


async def _attempt(args: argparse.Namespace, problem: str, results_dir: Path) -> tuple[int, str]:
    """Run one problem once, in this process.

    Returns:
        The exit code the archived sweep would have recorded, and the agent's incident id
        when the run reached one.
    """
    # Built from the sweep's own namespace, so a new per-problem knob on the command line
    # reaches the driver without a second edit here.
    problem_args = argparse.Namespace(
        **{**vars(args), "problem": problem, "results_dir": str(results_dir)}
    )
    try:
        async with asyncio.timeout(args.problem_timeout) as bound:
            result = await lifecycle.run(problem_args)
    except TimeoutError:
        # A cancelled problem leaves AIOpsLab's own cleanup unrun, exactly as the
        # archived sweep's SIGKILL did. The next problem's deployment is what recovers
        # the cluster; the bridge and its port-forwards close through their context
        # manager either way.
        if bound.expired():
            log.error("%s exceeded the %.0fs host bound", problem, args.problem_timeout)
            return _HOST_TIMEOUT_EXIT, ""
        log.error("%s: the agent did not finish within its own timeout", problem)
        return 1, ""
    except (RuntimeError, ValueError) as exc:
        log.error("%s: %s", problem, exc)
        return 1, ""
    return 0, str(result.get("agent_incident_id") or "")


async def _run_problem(args: argparse.Namespace, item: Selected, results_dir: Path) -> _Outcome:
    """Run one problem under the archived sweep's retry rule."""
    attempt, exit_code, grade, incident_id, result = 1, 1, "error", "", None
    for attempt in range(1, args.max_attempts + 1):
        exit_code, incident_id = await _attempt(args, item.problem, results_dir)
        result = latest_result(results_dir)
        grade = grade_of(result) if result is not None else "error"
        if grade in _GRADED:
            if exit_code:
                # The experimental outcome is the grade AIOpsLab persisted, and cleanup
                # failing after it does not make the cell rerunnable.
                log.info(
                    "keeping the externally graded %s result for %s after the runner exited %d",
                    grade,
                    item.problem,
                    exit_code,
                )
            break
        if attempt < args.max_attempts:
            log.info("retrying %s after exit=%d grade=%s", item.problem, exit_code, grade)
            await asyncio.sleep(args.retry_delay)
    return _Outcome(attempt, exit_code, grade, incident_id, str(result) if result else "")


async def run_sweep(args: argparse.Namespace) -> int:
    """Run every selected problem once, and record what happened to each.

    Args:
        args: The parsed ``sweep`` command line.

    Returns:
        Zero once every problem has been attempted. A problem that failed is a row in
        ``status.tsv`` rather than a nonzero exit: the sweep's job is to finish and to say.

    Raises:
        RuntimeError: If the agent is unreachable, or another sweep holds the lock.
        ValueError: If a preflight refuses, or nothing is selected.
    """
    if args.max_attempts < 1:
        raise ValueError("--max-attempts must be at least 1")
    task_types = [task.strip() for task in str(args.task_types).split(",") if task.strip()]
    sweep_dir = Path(args.sweep_dir)
    results_root = sweep_dir / "results"
    results_root.mkdir(parents=True, exist_ok=True)

    with _sweep_lock(sweep_dir.parent / ".sweep.lock"), _sweep_log(sweep_dir / "sweep.log"):
        # Both preflights come before the first deployment, for the reason lifecycle.run
        # runs them before importing AIOpsLab: a token the agent cannot match disables
        # every cluster tool it has, and neither failure is worth a cluster deployment.
        lifecycle.resolve_bridge_token(args.bridge_token, args.bridge_token_file)
        await asyncio.to_thread(lifecycle.check_agent_reachable, args.agent_url)

        selected, excluded = select_problems(task_types, args.problems)
        _write_manifest(sweep_dir, selected, excluded, task_types, args)
        log.info(
            "sweep %s: %d problems to run, %d Docker-backed problems excluded",
            sweep_dir.name,
            len(selected),
            len(excluded),
        )

        for index, item in enumerate(selected, start=1):
            results_dir = results_root / item.problem
            results_dir.mkdir(parents=True, exist_ok=True)
            existing = latest_result(results_dir)
            existing_grade = grade_of(existing) if existing is not None else "error"
            if existing_grade in _GRADED and not args.rerun:
                log.info(
                    "[%d/%d] skipping %s: already graded %s",
                    index,
                    len(selected),
                    item.problem,
                    existing_grade,
                )
                continue

            started, clock = lifecycle.utc_now(), time.monotonic()
            log.info("[%d/%d] starting %s (%s)", index, len(selected), item.problem, item.task_type)
            outcome = await _run_problem(args, item, results_dir)
            duration = round(time.monotonic() - clock, 1)
            _append_status_row(
                sweep_dir / "status.tsv",
                (
                    index,
                    item.problem,
                    item.task_type,
                    outcome.attempts,
                    outcome.exit_code,
                    outcome.grade,
                    outcome.incident_id,
                    started,
                    lifecycle.utc_now(),
                    duration,
                    outcome.result_file,
                ),
            )
            log.info(
                "[%d/%d] finished %s exit=%d grade=%s in %.0fs",
                index,
                len(selected),
                item.problem,
                outcome.exit_code,
                outcome.grade,
                duration,
            )

        with (sweep_dir / "metadata.env").open("a", encoding="utf-8") as metadata:
            metadata.write(f"completed_utc={lifecycle.utc_now()}\n")
        log.info("sweep complete: %s", sweep_dir)
    return 0


@contextmanager
def _sweep_lock(path: Path) -> Iterator[None]:
    """Refuse to start while another sweep is running on this host.

    AIOpsLab deploys into shared cluster-wide state, so two sweeps do not produce two
    arms: they produce two corrupted ones, and the archived sweep script held the same
    lock for the same reason.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f"another AIOpsLab sweep holds {path}") from exc
        yield
    finally:
        handle.close()


@contextmanager
def _sweep_log(path: Path) -> Iterator[None]:
    """Send the driver's log to the sweep directory as well as to the console.

    The sweep's own record of what it did. What AIOpsLab itself prints goes to the
    console, which is where the wrapper's ``tee`` catches it.
    """
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        yield
    finally:
        root.removeHandler(handler)
        handler.close()


def _write_manifest(
    sweep_dir: Path,
    selected: Sequence[Selected],
    excluded: Sequence[str],
    task_types: Sequence[str],
    args: argparse.Namespace,
) -> None:
    """Record what this sweep set out to do, and what it deliberately will not run."""
    (sweep_dir / "problem-list.txt").write_text(
        "".join(f"{item.problem}\t{item.task_type}\n" for item in selected), encoding="utf-8"
    )
    (sweep_dir / "excluded-docker-problems.txt").write_text(
        "".join(f"{problem}\n" for problem in excluded), encoding="utf-8"
    )
    metadata = {
        "sweep_id": sweep_dir.name,
        "started_utc": lifecycle.utc_now(),
        "agent_url": args.agent_url,
        "requested_task_types": ",".join(task_types),
        "problem_count": len(selected),
        "excluded_docker_problem_count": len(excluded),
        "max_attempts": args.max_attempts,
        "problem_timeout_seconds": int(args.problem_timeout),
        "agent_timeout_seconds": int(args.agent_timeout),
    }
    # Rewritten on a resume, and `completed_utc` with it: a sweep that has been restarted
    # is not a completed one, and the watcher reads that line to decide.
    (sweep_dir / "metadata.env").write_text(
        "".join(f"{key}={value}\n" for key, value in metadata.items()), encoding="utf-8"
    )


def _append_status_row(status_file: Path, row: tuple[Any, ...]) -> None:
    """Append one problem's row, writing the header the first time."""
    if not status_file.exists():
        status_file.write_text("\t".join(STATUS_COLUMNS) + "\n", encoding="utf-8")
    with status_file.open("a", encoding="utf-8") as handle:
        handle.write("\t".join(str(field) for field in row) + "\n")
