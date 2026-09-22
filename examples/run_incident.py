#!/usr/bin/env python3
"""Drive one incident against a running MAVER and print what it concluded.

The shortest complete example of the service's contract: post an alert, poll until the
run reaches a terminal status, print the artifacts. Nothing here is MAVER-specific beyond
the two URLs, which is the point — an operator's own alerting can post the same body.

    maver serve &
    python examples/run_incident.py examples/alerts/high-error-rate.json

Pass ``--target-stage diagnose`` to stop after the diagnosis without touching the
cluster, which is what an environment that grades detection separately asks for.
"""

import argparse
import json
import pathlib
import sys
import time
from typing import Any

import httpx

TERMINAL = {"completed", "failed", "cancelled"}


def main() -> int:
    """Run one incident and report it.

    Returns:
        A process exit status: zero when the run reached a terminal state.
    """
    args = _arguments()
    alert = json.loads(pathlib.Path(args.alert).read_text(encoding="utf-8"))

    with httpx.Client(base_url=args.url, timeout=30.0) as client:
        accepted = client.post(
            "/api/v1/incidents",
            json={"alert": alert, "target_stage": args.target_stage},
        )
        accepted.raise_for_status()
        session_id = accepted.json()["session_id"]
        print(f"accepted as {session_id}", file=sys.stderr)

        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            response = client.get(f"/api/v1/incidents/{session_id}")
            response.raise_for_status()
            incident = response.json()
            status = incident.get("run_status")
            if status in TERMINAL:
                print(json.dumps(incident, indent=2))
                return 0
            if status == "awaiting_human":
                # The run is parked on an approval. It is not stuck: a decision through
                # `maver approvals` or the HTTP surface continues this same incident.
                print(json.dumps(incident, indent=2))
                print("parked on a human decision", file=sys.stderr)
                return 0
            if status == "needs_reconciliation":
                print(json.dumps(incident, indent=2))
                print(
                    "operator reconciliation required before this incident can continue",
                    file=sys.stderr,
                )
                return 0
            print(f"  {status} at {incident.get('current_stage') or 'startup'}", file=sys.stderr)
            time.sleep(args.poll)

    print(f"still running after {args.timeout}s", file=sys.stderr)
    return 1


def _arguments() -> Any:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("alert", help="Path to a JSON alert payload.")
    parser.add_argument("--url", default="http://127.0.0.1:8080", help="Where MAVER is serving.")
    parser.add_argument(
        "--target-stage",
        default="full",
        choices=("full", "diagnose"),
        help="How far to run. `diagnose` stops before anything is changed.",
    )
    parser.add_argument("--poll", type=float, default=5.0, help="Seconds between polls.")
    parser.add_argument("--timeout", type=float, default=3600.0, help="Seconds to wait.")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
