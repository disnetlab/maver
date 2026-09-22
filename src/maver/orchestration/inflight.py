"""The episodes this process is still holding open.

An adapter instance *is* one episode's binding. It holds the transport that episode's
tools speak, the correlation id the harness keys its own rollback journal on, and
whatever taking the episode gave it. None of that is serialisable and none of it is
rebuildable from a name: an adapter built fresh from the registry has acquired nothing,
and its ``tool_transport`` is empty until it has. So an incident that parks on a person
has to be continued by the instance that started it, and this is where that instance
waits for the answer.

Process-local, and that is the limitation rather than an oversight. A restart empties
this, and the resume path is required to say so and end the episode rather than continue
it under whatever composition happens to be configured; see
``docs/operations.md`` for restart recovery and reconciliation boundaries.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from maver.adapters.base import EnvironmentAdapter

__all__ = [
    "HeldEpisode",
    "clear_held_episodes",
    "drop_episode",
    "held_episode",
    "hold_episode",
]


@dataclass(frozen=True, slots=True)
class HeldEpisode:
    """One episode still open in this process, and how to bring it to an end."""

    adapter: EnvironmentAdapter
    finish: Callable[[str], Awaitable[str]]
    """Continue the parked incident and close the episode out, given the decided request.

    Supplied by the driver that opened the episode, because submitting the answers,
    collecting the grades and releasing the environment are its business and not this
    registry's."""


_HELD: dict[str, HeldEpisode] = {}


def hold_episode(
    session_id: str, adapter: EnvironmentAdapter, finish: Callable[[str], Awaitable[str]]
) -> None:
    """Record that an environment is serving this incident.

    Args:
        session_id: The incident the episode runs as.
        adapter: The instance serving it.
        finish: Continues the incident after a decision and closes the episode out.
    """
    _HELD[session_id] = HeldEpisode(adapter=adapter, finish=finish)


def held_episode(session_id: str) -> HeldEpisode | None:
    """Look up the episode serving one incident.

    Args:
        session_id: The incident.

    Returns:
        The held episode, or None when no environment in this process is serving it.
    """
    return _HELD.get(session_id)


def drop_episode(session_id: str) -> HeldEpisode | None:
    """Take one episode out of the registry, if it is still in it.

    Args:
        session_id: The incident.

    Returns:
        The entry that was removed, or None when it had already gone. Taken rather than
        read, so that whoever removed it is the one that ends the episode: an episode has
        one ending, and two continuations racing for it would submit and release twice.
    """
    return _HELD.pop(session_id, None)


def clear_held_episodes() -> None:
    """Forget every held episode. For a clean shutdown, and for tests."""
    _HELD.clear()
