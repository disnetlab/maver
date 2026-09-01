"""The kickoff driver the harness runs as a subprocess.

The harness spawns this and does not tear the environment down until it exits, so
**process lifetime is environment lifetime** and this must block for the whole episode.
That is the single most load-bearing fact of the integration: an early exit tears down a
cluster the agent is still working in.

Four calls, in order. Wait for the service to answer its liveness probe, optionally
check the cluster bridge is up, post one episode, then poll the incident until its run
status is terminal. It exits zero only on a completed run.

Standard library only, deliberately: this runs in the harness's interpreter, not ours,
and it is loaded from a path in the harness's own agent configuration.
"""

import json
import logging
import os
import sys
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

__all__ = ["main"]

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger("sregym-driver")

AGENT_URL = os.environ.get("AGENT_URL", "http://127.0.0.1:8080").rstrip("/")
POLL_INTERVAL = int(os.environ.get("POLL_INTERVAL", "5"))
HEALTH_TIMEOUT = int(os.environ.get("HEALTH_TIMEOUT", "30"))
REQUIRE_KUBECTL_PROXY = os.environ.get("REQUIRE_KUBECTL_PROXY", "").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
KUBECTL_PROXY_URL = os.environ.get("KUBECTL_PROXY_URL", "http://127.0.0.1:16444").rstrip("/")

TERMINAL_RUN_STATUSES = {"completed", "failed", "cancelled"}
"""The run status is the only thing this waits on, never the lifecycle state.

The lifecycle flips to invalidated the moment a remediation is judged failed, which is
*before* the environment is released. Exiting on it tears the environment down
mid-episode and races the grader. A patch that did exactly that was removed once and
must not come back."""


def http_json(url: str, *, method: str = "GET", timeout: float = 30) -> dict[str, Any]:
    """Make one request and decode its reply.

    Args:
        url: The absolute URL.
        method: The HTTP method.
        timeout: Seconds to wait.

    Returns:
        The decoded body.
    """
    request = Request(url, method=method)
    if method == "POST":
        request.add_header("Content-Length", "0")
    with urlopen(request, timeout=timeout) as response:
        decoded: dict[str, Any] = json.loads(response.read())
        return decoded


def wait_for_health() -> bool:
    """Wait for the service to answer its liveness probe.

    Returns:
        True once it answers, False if it never does.
    """
    deadline = time.monotonic() + HEALTH_TIMEOUT
    while time.monotonic() < deadline:
        try:
            http_json(f"{AGENT_URL}/healthz", timeout=5)
            return True
        except (URLError, HTTPError, OSError):
            time.sleep(1)
    return False


def wait_for_kubectl_proxy() -> bool:
    """Wait for the cluster bridge, when this deployment requires one.

    Returns:
        True when the bridge answers or is not required.
    """
    if not REQUIRE_KUBECTL_PROXY:
        return True
    deadline = time.monotonic() + HEALTH_TIMEOUT
    while time.monotonic() < deadline:
        try:
            with urlopen(f"{KUBECTL_PROXY_URL}/version", timeout=5) as response:
                response.read(128)
            return True
        except (URLError, HTTPError, OSError):
            time.sleep(1)
    return False


def start_episode() -> dict[str, Any]:
    """Ask the service to take the running problem.

    Returns:
        The acceptance body.
    """
    url = f"{AGENT_URL}/api/v1/sregym/episodes?submit=true"
    log.info("POST %s", url)
    return http_json(url, method="POST")


def wait_for_completion(session_id: str) -> dict[str, Any]:
    """Poll one incident until its run reaches a terminal status.

    Args:
        session_id: The session to poll.

    Returns:
        The last record read.
    """
    url = f"{AGENT_URL}/api/v1/incidents/{session_id}"
    while True:
        time.sleep(POLL_INTERVAL)
        try:
            data = http_json(url)
        except (URLError, HTTPError, OSError) as exc:
            log.warning("poll failed (will retry): %s", exc)
            continue
        log.info(
            "session=%s run_status=%s stage=%s state=%s",
            session_id,
            data.get("run_status"),
            data.get("current_stage"),
            data.get("status"),
        )
        if data.get("run_status") in TERMINAL_RUN_STATUSES:
            return data


def main() -> int:
    """Run one episode to its end.

    Returns:
        Zero when the run completed, one otherwise.
    """
    log.info("agent_url=%s poll_interval=%ds", AGENT_URL, POLL_INTERVAL)

    if not wait_for_health():
        log.error(
            "agent not reachable at %s after %ds — is the container running on the host network?",
            AGENT_URL,
            HEALTH_TIMEOUT,
        )
        return 1
    log.info("agent is healthy")

    if not wait_for_kubectl_proxy():
        log.error(
            "kubectl proxy not reachable at %s after %ds — is the port-forward running?",
            KUBECTL_PROXY_URL,
            HEALTH_TIMEOUT,
        )
        return 1
    if REQUIRE_KUBECTL_PROXY:
        log.info("kubectl proxy is reachable at %s", KUBECTL_PROXY_URL)

    try:
        accepted = start_episode()
    except (URLError, HTTPError, OSError) as exc:
        log.error("failed to start episode: %s", exc)
        return 1

    session_id = accepted.get("session_id", "")
    problem_id = accepted.get("problem_id")
    log.info("episode started: session=%s problem=%s", session_id, problem_id)

    result = wait_for_completion(session_id)
    run_status = result.get("run_status")
    log.info(
        "episode finished: session=%s run_status=%s state=%s problem=%s",
        session_id,
        run_status,
        result.get("status"),
        problem_id,
    )
    return 0 if run_status == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
