"""The incidents this process is carrying right now.

An in-process registry: the live set, the fingerprint index that recognises a repost of
one alert, the reservation that makes admission a single step, and the task handle an
operator's cancel reaches for. State is lost on a restart, by design: only runs in
progress exist here, and a run that did not survive the restart has nothing left to stop.

Distinct from the approval gate, which parks a run durably on a human decision. This is
the operator's hand on a run that is behaving badly — the one who wants it to stop now,
not the one being asked to approve a write.
"""

import asyncio

__all__ = ["RunRegistry"]


class RunRegistry:
    """The runs this process has going."""

    def __init__(self) -> None:
        """Start empty."""
        self._runs: dict[str, asyncio.Task[None] | None] = {}
        self._tasks: set[asyncio.Task[None]] = set()
        self._by_alert: dict[str, str] = {}
        self._reserved = 0
        self._draining = False

    def register(self, session_id: str, *, fingerprint: str | None = None) -> None:
        """Start tracking a run.

        Args:
            session_id: The incident.
            fingerprint: What identifies the alert that raised it, when the caller
                computed one. Used to recognise a repost of the same alert.
        """
        self._runs[session_id] = None
        if fingerprint is not None:
            self._by_alert[fingerprint] = session_id

    def attach(self, session_id: str, task: asyncio.Task[None]) -> None:
        """Hold the task carrying a registered run.

        Held strongly, because ``asyncio`` keeps only a weak reference to a task and a
        background incident nobody is awaiting can be collected mid-run. The reference
        is given up when the task ends rather than when its run unregisters, which are
        not the same moment: a run gives up its place in the live set from inside the
        task that is still unwinding.

        Args:
            session_id: The incident, already registered.
            task: The task running it.
        """
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        if session_id in self._runs:
            self._runs[session_id] = task

    def unregister(self, session_id: str) -> None:
        """Stop tracking a run that has finished.

        Args:
            session_id: The incident.
        """
        self._runs.pop(session_id, None)
        for fingerprint, tracked in list(self._by_alert.items()):
            if tracked == session_id:
                del self._by_alert[fingerprint]

    def cancel(self, session_id: str) -> bool:
        """Ask a run this process is carrying to stop.

        Cancelling the task rather than raising a flag the loop polls: every await in a
        run is already a cancellation point, so the stop arrives at the next one without
        the loop having to check for it, and the task's own handler is what records that
        the incident is over and lets an adapter hand its environment back.

        Args:
            session_id: The incident.

        Returns:
            True when a running task was asked to stop. False when this process is not
            carrying the run — it finished, it is parked on a person, or it never ran
            here.
        """
        task = self._runs.get(session_id)
        if task is None or task.done():
            return False
        task.cancel()
        return True

    def tasks(self) -> set[asyncio.Task[None]]:
        """The tasks carrying the runs in this process.

        Returns:
            A snapshot, so a caller may cancel or await them while they finish and
            remove themselves.
        """
        return set(self._tasks)

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

    @property
    def draining(self) -> bool:
        """Whether this process has begun shutting down.

        Returns:
            True once the drain has started.
        """
        return self._draining

    def begin_draining(self) -> None:
        """Stop accepting work, before waiting for what is already in flight.

        Draining is two steps in that order, and only the second was here. A process
        that keeps accepting while it drains opens an incident it cannot finish: the
        session says running, the process goes, and nothing resumes it — which is the
        outcome the drain exists to prevent, arriving through the front door. It is also
        what takes the pod out of a Service's endpoint list, so the requests stop
        arriving in the first place.
        """
        self._draining = True

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
        if self._draining or self.at_capacity(limit):
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

    def live(self) -> list[str]:
        """Which incidents are running here.

        Returns:
            Their session identifiers.
        """
        return sorted(self._runs)
