"""Persist private run state independently from model-visible incident artifacts."""

from maver.contracts.runtime import RunState
from maver.memory.ports import SessionStore

__all__ = ["load_runtime", "save_runtime"]


def save_runtime(sessions: SessionStore, session_id: str, runtime: RunState) -> None:
    """Save a detached JSON snapshot of the shared state.

    Args:
        sessions: The configured durable session store.
        session_id: The incident whose state is saved.
        runtime: Its selected composition, metadata and isolated agent states.
    """
    sessions.set_runtime(session_id, runtime.model_dump(mode="json"))


def load_runtime(sessions: SessionStore, session_id: str) -> RunState | None:
    """Restore private run state, without inferring it from the alert or checkpoint.

    Args:
        sessions: The configured durable session store.
        session_id: The incident to restore.

    Returns:
        Validated state, or None for an absent session or a run predating this state.
    """
    session = sessions.get_session(session_id)
    if session is None or not session.runtime:
        return None
    return RunState.model_validate(session.runtime)
