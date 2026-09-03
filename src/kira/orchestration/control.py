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
        self._by_alert: dict[str, str] = {}
        self._reserved = 0

    def register(self, session_id: str, *, fingerprint: str | None = None) -> RunControl:
        """Start tracking a run.

        Args:
            session_id: The incident.
            fingerprint: What identifies the alert that raised it, when the caller
                computed one. Used to recognise a repost of the same alert.

        Returns:
            Its control handle.
        """
        control = RunControl(session_id)
        self._runs[session_id] = control
        if fingerprint is not None:
            self._by_alert[fingerprint] = session_id
        return control

    def unregister(self, session_id: str) -> None:
        """Stop tracking a run that has finished.

        Args:
            session_id: The incident.
        """
        self._runs.pop(session_id, None)
        for fingerprint, tracked in list(self._by_alert.items()):
            if tracked == session_id:
                del self._by_alert[fingerprint]

    def find(self, fingerprint: str) -> str | None:
        """Find a live run already working on this alert.

        Deliberately only the live ones, and deliberately only in this process. An
        alerting source retries on a timescale of seconds, which is what this covers.
        A restart clears it, and a restart also loses the run it was tracking — so a
        repost after one is a new incident, correctly.

        Args:
            fingerprint: What identifies the alert.

        Returns:
            The session already handling it, or None.
        """
        return self._by_alert.get(fingerprint)

    def at_capacity(self, limit: int) -> bool:
        """Whether this process is already carrying as many incidents as it should.

        A private model endpoint serves a small number of concurrent sequences, and
        oversubscribing one degrades every run in flight rather than queueing — so the
        bound is on accepting work, not on dispatching it.

        Args:
            limit: The configured ceiling.

        Returns:
            True when a further incident should not be accepted.
        """
        return len(self._runs) + self._reserved >= limit

    def reserve(self, limit: int) -> bool:
        """Take one of this process's slots, without awaiting anything.

        Asking whether there is room and taking the room have to be one step. Opening a
        session and marking it running are two database writes, and every request in a
        burst yields at both — so each of them passed a capacity check that no other had
        yet invalidated, and a process configured for two carried seven. A bound that
        holds only when requests arrive one at a time is not a bound, and this one exists
        because oversubscribing a private endpoint wedges it rather than queueing.

        Args:
            limit: The configured ceiling.

        Returns:
            True when a slot was taken. The caller releases it once the run it started
            is registered, or once it is certain none will be.
        """
        if self.at_capacity(limit):
            return False
        self._reserved += 1
        return True

    def release(self) -> None:
        """Give back a reserved slot.

        Paired with :meth:`reserve` by the caller rather than consumed by
        :meth:`register`, because runs are registered on paths that never reserved —
        a resumed incident among them — and a counter two callers decrement is one that
        drifts.
        """
        self._reserved = max(0, self._reserved - 1)

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
