"""The driver's command line: run one problem, sweep every problem, or check the host.

``doctor`` exists because every one of its four checks has failed on a host at some
point, and each failure looked like something else: an agent container that was not
running reads as a hung episode, a missing ``kubectl`` reads as a cluster outage, and a
token the agent cannot match reads as a model that never used its tools.
"""

import argparse
import asyncio
import importlib.util
import json
import logging
import os
import shutil
from pathlib import Path

from . import lifecycle, sweep

__all__ = ["main"]

log = logging.getLogger("aiopslab-driver")


def _parser() -> argparse.ArgumentParser:
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--agent-url", default="http://127.0.0.1:8080")
    shared.add_argument("--bridge-token", default=os.environ.get("AIOPSLAB_BRIDGE_TOKEN", ""))
    shared.add_argument(
        "--bridge-token-file",
        default=os.environ.get("AIOPSLAB_BRIDGE_TOKEN_FILE") or None,
        help="read the bearer token from this file instead of the environment",
    )

    # What one problem costs, whether it arrives alone or inside a sweep. Shared so a
    # sweep cannot drift from the single-problem command it runs.
    episode = argparse.ArgumentParser(add_help=False)
    episode.add_argument("--agent-timeout", type=float, default=7200)
    episode.add_argument("--poll-seconds", type=float, default=2)
    episode.add_argument("--bridge-host", default="127.0.0.1")
    episode.add_argument("--bridge-port", type=int, default=1819)
    episode.add_argument("--prometheus-url", default="http://127.0.0.1:32000")

    parser = argparse.ArgumentParser(prog="aiopslab-driver", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    runner = commands.add_parser("run", parents=[shared, episode], help="run one AIOpsLab problem")
    runner.add_argument("--problem", required=True, help="AIOpsLab problem ID")
    runner.add_argument("--results-dir", default="/var/lib/aiopslab/results/maver")

    sweeper = commands.add_parser(
        "sweep", parents=[shared, episode], help="run every selected problem once, resumably"
    )
    sweeper.add_argument(
        "--sweep-dir",
        required=True,
        help="this sweep's directory: results, status.tsv and sweep.log are written here, "
        "and reusing it resumes the sweep",
    )
    sweeper.add_argument(
        "--problems",
        type=Path,
        default=None,
        help="a file of problem IDs, one per line; AIOpsLab's registry is read without it",
    )
    sweeper.add_argument(
        "--task-types",
        default=",".join(sweep.TASK_TYPES),
        help="comma-separated graded task types to include",
    )
    sweeper.add_argument(
        "--max-attempts",
        type=int,
        default=2,
        help="attempts per problem; a result AIOpsLab graded is never retried",
    )
    sweeper.add_argument(
        "--problem-timeout",
        type=float,
        default=7800,
        help="seconds one problem may take, its deployment and grading included",
    )
    sweeper.add_argument("--retry-delay", type=float, default=15)
    sweeper.add_argument(
        "--rerun", action="store_true", help="run problems that already have a graded result"
    )

    commands.add_parser("doctor", parents=[shared], help="check this host can run a problem")
    return parser


def _doctor(args: argparse.Namespace) -> int:
    """Report on each precondition separately, rather than on the first failure."""
    results: list[tuple[bool, str, str]] = []

    try:
        lifecycle.check_agent_reachable(args.agent_url)
        results.append((True, "agent", f"{args.agent_url} answers /healthz"))
    except RuntimeError as exc:
        results.append((False, "agent", str(exc)))

    found = importlib.util.find_spec("aiopslab") is not None
    results.append(
        (found, "aiopslab", "importable" if found else "not importable in this interpreter")
    )

    kubectl = shutil.which("kubectl")
    results.append((bool(kubectl), "kubectl", kubectl or "not on PATH"))

    try:
        lifecycle.resolve_bridge_token(args.bridge_token, args.bridge_token_file)
        results.append((True, "bridge token", "present"))
    except ValueError as exc:
        results.append((False, "bridge token", str(exc)))

    for ok, name, detail in results:
        print(f"[{'ok' if ok else 'FAIL'}] {name}: {detail}")
    return 0 if all(ok for ok, _, _ in results) else 1


def main(argv: list[str] | None = None) -> int:
    """Run one command.

    Args:
        argv: Arguments to parse, for tests. Defaults to the process's own.

    Returns:
        Zero on success. One for a failure the sweep should record and may retry,
        which is every startup refusal and an incident that ran out of time. A sweep
        returns zero once it has attempted every problem, whatever each of them did.
    """
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )
    if args.command == "doctor":
        return _doctor(args)
    try:
        if args.command == "sweep":
            return asyncio.run(sweep.run_sweep(args))
        result = asyncio.run(lifecycle.run(args))
    except (RuntimeError, ValueError, TimeoutError) as exc:
        log.error("%s", exc)
        return 1
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
