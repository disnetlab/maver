"""The episode routes the external kickoff driver posts to.

Held byte-compatible with the prototype's, because the harness that drives them lives
outside this repository, on the machines that produced every published number, and every
comparison against that archive depends on nothing here moving. The acceptance body's
four keys and the 202 status are read by the driver; the two debug passthroughs are not
part of that contract but they are the fastest way to ask whether the conductor is
reachable at all, which is the first thing a deploy checks.

A path naming a benchmark lives here rather than under the core's API for the same
reason the transport does: the core has no route it would not serve for any deployment.
"""

import asyncio
import logging
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException

from kira.adapters.registry import get_adapter
from kira.adapters.sregym.conductor import Conductor
from kira.api.deps import Service, ServiceDep
from kira.orchestration.episode import open_episode, run_episode

__all__ = ["router"]

_log = logging.getLogger(__name__)

router = APIRouter()

_ADAPTER = "sregym"


@router.post("/sregym/episodes", status_code=202)
async def create_episode(service: ServiceDep, submit: bool = True) -> dict[str, Any]:
    """Take the running problem and start working on it.

    Answers as soon as the episode has a session, because the work behind it is minutes
    to hours of model calls and the caller polls the incident route for the rest.

    Args:
        service: The process's resources.
        submit: Whether answers are handed over for grading. False rehearses an episode
            without touching the grader.

    Returns:
        The four keys the kickoff driver reads: the session to poll, that it was
        accepted, which problem this was, and when it opened.

    Raises:
        HTTPException: 502 when the conductor could not be reached or could not
            describe the application, which is a refusal to start rather than a failed
            episode.
    """
    adapter = get_adapter(_ADAPTER)(service.deployment.settings)
    try:
        start = await open_episode(adapter, service.deployment)
    except httpx.HTTPError as exc:
        raise _unreachable(exc) from exc

    service.runs.register(start.session_id)
    task = asyncio.create_task(_drive(adapter, service, start, submit=submit))
    service.tasks.add(task)
    task.add_done_callback(service.tasks.discard)
    return {
        "session_id": start.session_id,
        "status": "accepted",
        # Read into the driver's log line and nowhere else. It is the ground-truth fault
        # label, so it must not reach a prompt, and the pipeline never sees this reply.
        "problem_id": start.context.external_key,
        "started_at": start.started_at,
    }


@router.get("/sregym/problem")
async def read_problem(service: ServiceDep) -> dict[str, Any]:
    """Ask the conductor what problem is running.

    Args:
        service: The process's resources.

    Returns:
        The conductor's answer, unmodified.

    Raises:
        HTTPException: 502 when the conductor could not be reached.
    """
    try:
        return {"problem": await _conductor(service).get_problem()}
    except httpx.HTTPError as exc:
        raise _unreachable(exc) from exc


@router.get("/sregym/status")
async def read_status(service: ServiceDep) -> dict[str, Any]:
    """Ask the conductor where the episode is.

    Args:
        service: The process's resources.

    Returns:
        The conductor's answer, unmodified.

    Raises:
        HTTPException: 502 when the conductor could not be reached.
    """
    try:
        return {"status": await _conductor(service).status()}
    except httpx.HTTPError as exc:
        raise _unreachable(exc) from exc


async def _drive(adapter: Any, service: Service, start: Any, *, submit: bool) -> None:
    """Run one episode and always let go of its place in the live-run registry."""
    try:
        await run_episode(adapter, service.deployment, start, submit=submit)
    except asyncio.CancelledError:
        _log.info("episode %s was cancelled", start.session_id)
        raise
    finally:
        service.runs.unregister(start.session_id)


def _conductor(service: Service) -> Conductor:
    """Build a throwaway client for a debug read, configured like the episode's own."""
    from kira.adapters.sregym import SregymConfig  # noqa: PLC0415 - avoids a cycle

    config = SregymConfig.resolve(service.deployment.settings)
    return Conductor(str(config.conductor_base_url), timeout_s=config.conductor_timeout_s)


def _unreachable(exc: Exception) -> HTTPException:
    """Render a conductor failure as a gateway error naming what could not answer."""
    return HTTPException(status_code=502, detail=f"conductor error: {type(exc).__name__}: {exc}")
