"""Pausing and cancelling a run that is already going.

An in-process registry of the incidents running right now, checked by the loop before
each model call. State is lost on a restart, by design: only in-progress runs exist
here, and a run that did not survive the restart has nothing left to pause.

Distinct from the approval gate, which parks a run durably on a human decision. This is
the operator's hand on a run that is behaving badly — the one who wants it to stop now,
not the one being asked to approve a write.
"""

import asyncio
import logging
from dataclasses import dataclass, field

__all__ = ["RunCancelled", "RunControl", "RunRegistry"]

_log = logging.getLogger(__name__)


class RunCancelled(Exception):
    """A run was cancelled by an operator while it was waiting."""


@dataclass
class RunControl:
    """One live run's pause and cancel state."""

    session_id: str
    _resumed: asyncio.Event = field(default_factory=asyncio.Event)
    _cancelled: bool = False

    def __post_init__(self) -> None:
        """Start released. A run is only held once someone asks for it."""
        self._resumed.set()

    @property
    def paused(self) -> bool:
        """Whether the run is currently held.

        Returns:
            True while paused.
        """
        return not self._resumed.is_set()

    @property
    def cancelled(self) -> bool:
        """Whether the run has been asked to stop.

        Returns:
            True once cancelled.
        """
        return self._cancelled

    def pause(self) -> None:
        """Hold the run before its next model call."""
        self._resumed.clear()

    def resume(self) -> None:
        """Release a held run."""
        self._resumed.set()

    def cancel(self) -> None:
        """Ask the run to stop, and release it if it is waiting.

        Releasing matters: a paused run that is then cancelled would otherwise wait
        forever for a resume that is never coming.
        """
        self._cancelled = True
        self._resumed.set()

    async def wait_if_paused(self) -> None:
        """Block while paused, then continue or raise.

        Raises:
            RunCancelled: When the run was cancelled, whether before or during the wait.
        """
        if self._cancelled:
            raise RunCancelled(self.session_id)
        if not self._resumed.is_set():
            _log.info("run %s is paused", self.session_id)
            await self._resumed.wait()
        if self._cancelled:
            raise RunCancelled(self.session_id)


class RunRegistry:
    """The runs this process has going."""

    def __init__(self) -> None:
        """Start empty."""
        self._runs: dict[str, RunControl] = {}

    def register(self, session_id: str) -> RunControl:
        """Start tracking a run.

        Args:
            session_id: The incident.

        Returns:
            Its control handle.
        """
        control = RunControl(session_id)
        self._runs[session_id] = control
        return control

    def unregister(self, session_id: str) -> None:
        """Stop tracking a run that has finished.

        Args:
            session_id: The incident.
        """
        self._runs.pop(session_id, None)

    def get(self, session_id: str) -> RunControl | None:
        """Look up a live run.

        Args:
            session_id: The incident.

        Returns:
            Its control handle, or None when the run is not in this process.
        """
        return self._runs.get(session_id)

    def live(self) -> list[str]:
        """Which incidents are running here.

        Returns:
            Their session identifiers.
        """
        return sorted(self._runs)
