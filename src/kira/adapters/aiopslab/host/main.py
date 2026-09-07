"""The driver's command line: run one problem, or check the host can run one.

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

from . import lifecycle

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

    parser = argparse.ArgumentParser(prog="aiopslab-driver", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    runner = commands.add_parser("run", parents=[shared], help="run one AIOpsLab problem")
    runner.add_argument("--problem", required=True, help="AIOpsLab problem ID")
    runner.add_argument("--agent-timeout", type=float, default=7200)
    runner.add_argument("--poll-seconds", type=float, default=2)
    runner.add_argument("--bridge-host", default="127.0.0.1")
    runner.add_argument("--bridge-port", type=int, default=1819)
    runner.add_argument("--prometheus-url", default="http://127.0.0.1:32000")
    runner.add_argument("--results-dir", default="/var/lib/aiopslab/results/kira")

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
        which is every startup refusal and an incident that ran out of time.
    """
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )
    if args.command == "doctor":
        return _doctor(args)
    try:
        result = asyncio.run(lifecycle.run(args))
    except (RuntimeError, ValueError, TimeoutError) as exc:
        log.error("%s", exc)
        return 1
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
