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
from dataclasses import dataclass, field

__all__ = ["Claim", "RunRegistry"]


@dataclass(slots=True)
class _Held:
    """What this process is doing about one alert, from the claim to the session."""

    opened: asyncio.Event = field(default_factory=asyncio.Event)
    session_id: str | None = None


@dataclass(frozen=True, slots=True)
class Claim:
    """One intake's hold on an alert and on a slot, taken in a single step."""

    fingerprint: str | None
    """What identifies the alert, when the caller deduplicates on one."""
    ours: bool
    """Whether this caller is the one opening the incident. False when another request
    got to the alert first, and this one is waiting for the session that request opens."""
    _held: _Held

    async def session(self) -> str | None:
        """Wait for the incident the request ahead of this one is opening.

        Returns:
            The session already handling this alert, or None when the request that
            claimed it gave up without opening one.
        """
        await self._held.opened.wait()
        return self._held.session_id


class RunRegistry:
    """The runs this process has going."""

    def __init__(self) -> None:
        """Start empty."""
        self._runs: dict[str, asyncio.Task[None] | None] = {}
        self._tasks: set[asyncio.Task[None]] = set()
        self._alerts: dict[str, _Held] = {}
        self._reserved = 0
        self._draining = False

    def register(self, session_id: str, *, fingerprint: str | None = None) -> None:
        """Start tracking a run, filling in the claim its intake took.

        Args:
            session_id: The incident.
            fingerprint: What identifies the alert that raised it, when the caller
                computed one. Used to recognise a repost of the same alert.
        """
        self._runs[session_id] = None
        if fingerprint is None:
            return
        held = self._alerts.setdefault(fingerprint, _Held())
        held.session_id = session_id
        held.opened.set()

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
        for fingerprint, held in list(self._alerts.items()):
            if held.session_id == session_id:
                del self._alerts[fingerprint]

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

    def claim(self, fingerprint: str | None, limit: int) -> Claim | None:
        """Take one of this process's slots, and the alert with it, in a single step.

        Asking whether an alert is already being handled and recording that it now is
        have to be one step, for the same reason asking whether there is room and taking
        the room do. Opening a session and marking it running are two database writes,
        and every request in a burst yields at both — so each of them passed checks no
        other had yet answered: a process configured for two carried seven, and four
        posts of one alert started four agents remediating the same fault against each
        other. A bound that holds only when requests arrive one at a time is not a
        bound, and neither is deduplication.

        Deduplication is deliberately over live runs only, and deliberately only in this
        process. An alerting source retries on a timescale of seconds, which is what this
        covers. A restart clears it, and a restart also loses the run it was tracking —
        so a repost after one is a new incident, correctly.

        Args:
            fingerprint: What identifies the alert, or None for an intake with nothing
                to deduplicate on.
            limit: The configured ceiling.

        Returns:
            A claim for this caller to fill by registering a session, a claim another
            request already holds — whose session id this one waits for — or None when
            the process is draining or already full.
        """
        held = self._alerts.get(fingerprint) if fingerprint is not None else None
        if held is not None:
            return Claim(fingerprint=fingerprint, ours=False, _held=held)
        if self._draining or self.at_capacity(limit):
            return None
        self._reserved += 1
        held = _Held()
        if fingerprint is not None:
            self._alerts[fingerprint] = held
        return Claim(fingerprint=fingerprint, ours=True, _held=held)

    def release(self, claim: Claim) -> None:
        """Give back what a claim took.

        Released by the caller rather than consumed by :meth:`register`, because runs
        are registered on paths that never claimed — a resumed incident among them — and
        a counter two callers decrement is one that drifts. The slot is held across the
        two writes that open the session, because that is the window a burst passes
        through, and given back once the run it started is registered or once it is
        certain none will be. A claim that opened nothing releases the alert too, so a
        request waiting behind it is answered rather than left waiting for an incident
        that is not coming.

        Args:
            claim: What :meth:`claim` returned. Releasing a claim this caller does not
                hold is a no-op.
        """
        if not claim.ours:
            return
        self._reserved = max(0, self._reserved - 1)
        if claim.fingerprint is None:
            return
        held = self._alerts.get(claim.fingerprint)
        if held is not None and held.session_id is None:
            del self._alerts[claim.fingerprint]
            held.opened.set()

    def live(self) -> list[str]:
        """Which incidents are running here.

        Returns:
            Their session identifiers.
        """
        return sorted(self._runs)
