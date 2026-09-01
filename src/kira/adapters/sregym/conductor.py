"""The conductor's HTTP API, as a client.

Six calls drive one episode: what problem is running, what application it deploys, what
stage the grader is on, what it graded, and the submissions that advance it.

Every response is decoded leniently. Response shapes vary across versions of the
harness, so a body is parsed as JSON when it says it is, then when it parses anyway, and
otherwise handed back as text; each caller picks the fields it needs and tolerates their
absence. That is not defensive programming for its own sake — a conductor that dropped a
route entirely is a version we still have to run against.
"""

import json
import logging
from typing import Any

import httpx

__all__ = ["Conductor"]

_log = logging.getLogger(__name__)

SSID_HEADER = "sregym_ssid"
"""How the harness correlates one agent run's cluster mutations with its own state.

Sent on every conductor request and every tool call, set once per episode and stable for
its whole lifetime, approval resumes included: the rollback journal it keys is per
session, so a changed value loses the writes it was recording.
"""


class Conductor:
    """One episode's conversation with the grading harness."""

    def __init__(self, base_url: str, *, timeout_s: float = 30.0) -> None:
        """Bind the client to one conductor.

        Args:
            base_url: Where the conductor listens.
            timeout_s: Per-request timeout.
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout_s
        self.session_id = ""

    def _headers(self) -> dict[str, str]:
        return {SSID_HEADER: self.session_id} if self.session_id else {}

    async def _get(self, path: str) -> Any:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(f"{self.base_url}{path}", headers=self._headers())
            response.raise_for_status()
            return _decode(response)

    async def get_problem(self) -> Any:
        """Read what problem is running.

        Returns:
            The problem blob, or None. Newer conductors dropped the route and answer
            404; the value only reaches logging and the storage-only episode key, so
            losing it degrades the record rather than the run.
        """
        try:
            return await self._get("/get_problem")
        except httpx.HTTPStatusError:
            return None

    async def get_app(self) -> Any:
        """Read the application metadata the incident signal is built from.

        Returns:
            The app blob.

        Raises:
            httpx.HTTPError: This is the only source for the agent-visible signal, so a
                failure here is a refusal to start rather than a degraded episode.
        """
        return await self._get("/get_app")

    async def status(self) -> Any:
        """Read the conductor's own account of where the episode is.

        Returns:
            The status body.

        Raises:
            httpx.HTTPError: When the conductor could not answer.
        """
        return await self._get("/status")

    async def stage(self) -> str | None:
        """Read the stage the grader will currently accept a submission for.

        Returns:
            One of ``setup``, ``diagnosis``, ``mitigation``, ``accept``,
            ``tearing_down`` or ``done``, or None before a problem has started, which
            the conductor answers with a 400 rather than a stage.
        """
        try:
            data = await self.status()
        except httpx.HTTPStatusError:
            return None
        return data.get("stage") if isinstance(data, dict) else None

    async def results(self, *, include_ground_truth: bool = False) -> Any | None:
        """Read the grades, when this conductor serves them.

        Args:
            include_ground_truth: Also ask for the reference root cause and the judge's
                reasoning, which only an upper-bound arm consumes.

        Returns:
            The results body, or None. The endpoint is an additive patch shipped to the
            harness, so an unpatched conductor answers 404 and the episode runs ungraded
            rather than failing.
        """
        path = "/results?ground_truth=true" if include_ground_truth else "/results"
        try:
            return await self._get(path)
        except httpx.HTTPStatusError:
            return None

    async def submit(self, solution: str, stage: str) -> Any:
        """Hand one answer over.

        Args:
            solution: The text to grade. Empty for the stages whose grade comes from
                the cluster rather than from what was written.
            stage: ``diagnosis``, ``mitigation`` or ``accept``. Any other value is
                rejected with a 400.

        Returns:
            The conductor's reply.

        Raises:
            httpx.HTTPError: When the conductor rejected or could not take it. The
                caller records the failure rather than letting it end the episode.
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/submit",
                json={"solution": solution, "stage": stage},
                headers=self._headers(),
            )
            response.raise_for_status()
            return _decode(response)


def _decode(response: httpx.Response) -> Any:
    """Read a body in whichever of the three shapes this conductor version sends."""
    if "json" in response.headers.get("content-type", "").lower():
        try:
            return response.json()
        except ValueError:
            pass
    text = response.text
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return text
