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
from importlib import import_module
from pathlib import Path
from typing import Protocol

from maver.memory.backends.sqlite import SqliteStore
from maver.memory.lifecycle import resolve_weights
from maver.memory.ports import ApprovalStore, RunStore, SessionStore, TextSearch
from maver.memory.ranking import build_ranker, load_entry_point_rankers
from maver.memory.retrieval import Retriever
from maver.settings.root import Settings

__all__ = ["StorageStack", "build_storage"]

_log = logging.getLogger(__name__)

_MISSING_EXTRA = "memory.backend='postgres' needs the postgres extra: install maver[postgres]"


class _Backend(SessionStore, ApprovalStore, RunStore, Protocol):
    """What a backend module supplies, as one object.

    The three operational ports, the corpus index beside them, and the way to close the
    connection they share. One object rather than four, because the stores and the index share a database and
    therefore a transaction boundary. This is the shape :func:`build_storage` needs and
    the only thing it knows about either backend.
    """

    @property
    def records(self) -> TextSearch: ...

    def close(self) -> None: ...


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
        ValueError: When the configured backend or ranker is not available in this
            installation, naming what to install. Raised here at startup rather than at
            the first incident, because a deployment that cannot persist, or cannot
            retrieve with the ranker it reports, has nothing useful to do.
    """
    load_entry_point_rankers()
    store = (
        _open_postgres(settings)
        if settings.memory.backend == "postgres"
        else _open_sqlite(settings)
    )
    return StorageStack(
        sessions=store,
        approvals=store,
        runs=store,
        index=store.records,
        retriever=Retriever(
            text=store.records,
            k=settings.memory.top_k,
            pool=settings.memory.candidate_pool,
            # Built here rather than defaulted, because the ranker decides how many
            # precedents a run is served and a campaign that ran under a different
            # relevance floor has no other way to say so.
            ranker=build_ranker(settings.memory),
            # The session row, not the record's write-time snapshot, is the authority on
            # a record's standing, so the retriever re-reads it as it ranks.
            states=store.get_states,
            weights=resolve_weights(settings.memory.lifecycle_weights),
        ),
        close=store.close,
    )


def _open_sqlite(settings: Settings) -> _Backend:
    """Open the embedded database under this deployment's workspace."""
    path = _database_path(settings)
    _log.info("opening the embedded experience store at %s", path)
    return SqliteStore(path)


def _open_postgres(settings: Settings) -> _Backend:
    """Open the server database, loading the driver only where it is needed.

    Imported by name rather than with a statement so that the module needing the extra
    never sits on the import path of an install without it, and because a function-local
    import statement is a lint error in this repository.

    Args:
        settings: The run's settings.

    Returns:
        The server backend, its schema already created.

    Raises:
        ValueError: When the extra is not installed, naming what to install. The
            settings group already refuses this backend without a connection string, so
            reaching here means the driver rather than the configuration is what is
            absent.
    """
    memory = settings.memory
    if memory.postgres_dsn is None:  # pragma: no cover - the settings group refuses this
        raise ValueError("memory.backend='postgres' requires memory.postgres_dsn")
    try:
        module = import_module("maver.memory.backends.postgres")
    except ImportError as exc:
        raise ValueError(f"{_MISSING_EXTRA} ({exc})") from exc
    _log.info(
        "opening the server experience store in schema %s for corpus %s",
        memory.postgres_schema,
        memory.corpus_id,
    )
    store: _Backend = module.PostgresStore(
        memory.postgres_dsn.get_secret_value(),
        schema=memory.postgres_schema,
        corpus_id=memory.corpus_id,
    )
    return store


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
