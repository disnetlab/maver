"""Building the storage stack, once, where it can be seen.

The only module that imports a backend. Everything above takes its stores as
arguments, which is the whole point: the prototype reached them through import-time
singletons, and a tool that can call ``get_session_store()`` is a tool that can also
read configuration, which is how experiment plumbing ended up inside the retrieval
path. Injection makes that impossible rather than discouraged.

One stack per process, built by the composition root and handed down. The operational
stores and the corpus index share a database and therefore a transaction boundary,
which is why a backend supplies all of them together rather than one at a time.
"""

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from kira.memory.backends.sqlite import SqliteStore
from kira.memory.lifecycle import resolve_weights
from kira.memory.ports import ApprovalStore, RunStore, SessionStore, TextSearch
from kira.memory.retrieval import Retriever
from kira.settings.root import Settings

__all__ = ["StorageStack", "build_storage"]

_log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class StorageStack:
    """Every persistent store this deployment has, and the way to close them."""

    sessions: SessionStore
    approvals: ApprovalStore
    runs: RunStore
    index: TextSearch
    retriever: Retriever
    close: Callable[[], None]


def build_storage(settings: Settings) -> StorageStack:
    """Open the configured stores.

    Args:
        settings: The run's settings.

    Returns:
        The stack, already wired to a retriever configured from the same settings.

    Raises:
        ValueError: When the configured backend is not available in this installation,
            naming what to install. Raised here at startup rather than at the first
            incident, because a deployment that cannot persist has nothing useful to do.
    """
    if settings.memory.backend == "postgres":
        # The settings group already refuses this backend without a connection string,
        # so reaching here means the driver, not the configuration, is what is absent.
        raise ValueError(
            "memory.backend='postgres' needs the postgres extra: install kira[postgres]"
        )

    path = _database_path(settings)
    _log.info("opening the embedded experience store at %s", path)
    store = SqliteStore(path)
    return StorageStack(
        sessions=store,
        approvals=store,
        runs=store,
        index=store.records,
        retriever=Retriever(
            text=store.records,
            k=settings.memory.top_k,
            # The session row, not the record's write-time snapshot, is the authority on
            # a record's standing, so the retriever re-reads it as it ranks.
            states=store.get_states,
            weights=resolve_weights(settings.memory.lifecycle_weights),
        ),
        close=store.close,
    )


def _database_path(settings: Settings) -> Path:
    """Place the embedded database beside the run's other durable state.

    Named for the corpus rather than fixed, which is how one campaign's records are kept
    out of another's: the partition is the database, so a run configured for a different
    corpus cannot read the previous one even by mistake.
    """
    directory = settings.server.workdir / "memory"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{_slug(settings.memory.corpus_id)}.sqlite"


def _slug(corpus_id: str) -> str:
    """Render a corpus name as a filename, refusing anything that could escape.

    Args:
        corpus_id: The configured corpus.

    Returns:
        The stem of its database file.

    Raises:
        ValueError: When the name is empty or is not a plain identifier. A corpus name
            reaches the filesystem, so it is validated rather than sanitised: silently
            rewriting it would put a campaign's records somewhere nobody looked.
    """
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", corpus_id):
        raise ValueError(
            f"memory.corpus_id {corpus_id!r} is not a usable name: it must start "
            "alphanumeric and contain only letters, digits, dot, dash or underscore"
        )
    return corpus_id
