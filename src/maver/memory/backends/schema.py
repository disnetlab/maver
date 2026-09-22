"""The store layout's version, and the open sequence that keeps a database at it.

Every backend meets the same four cases when it opens a database — one that is empty, one
written before this mechanism existed, one a step or more behind, and one written by a
build that knows a later layout — and only the SQL differs between them. So the sequence
lives here and each backend supplies its statements as a :class:`StoreSchema`.

A layout change belongs in a step function and nowhere else. The step is the single place
that says how version *n* becomes *n+1*, which is what keeps the open path from having to
work out what shape it is looking at: it reads a number. The one exception is a database
written before the number existed, whose shape has to be read off its own tables, once,
by :meth:`StoreSchema.detect_version`.
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping

__all__ = ["STORE_SCHEMA_VERSION", "StoreSchema", "StoreSchemaTooNew"]

_log = logging.getLogger(__name__)

STORE_SCHEMA_VERSION = 9
"""Version of the store *layout*: which tables, columns and indexes a database holds.

Not :data:`maver.contracts.SCHEMA_VERSION`, which versions the submission schemas and the
event contract — what a record says, and what every run writes into its configuration
snapshot so a stored result names the schemas that produced it. This one versions where a
record is kept. It is never written into a result, it moves whenever the DDL in a backend
module moves, and the two are free to differ.

Version 1 is the layout up to the shared normaliser, whose record index was over raw
text. Version 2 indexes the normalised terms of :mod:`maver.memory.lexemes` instead.
Version 3 carries the environment's own labels beside each record, unindexed.
Version 4 persists private pipeline and agent runtime state beside each session.
Version 5 records durable intake keys beside their initialized sessions.
Version 6 quarantines permanently rejected Kafka records for operator retry.
Version 7 records append-only operator outcome feedback.
Version 8 orders feedback by a session-scoped sequence allocated at commit.
Version 9 records eligible delayed Learning intents for recovery and retries.
"""


class StoreSchemaTooNew(RuntimeError):
    """A database written by a build that knows a later layout than this one."""

    def __init__(self, found: int, known: int) -> None:
        """Name both versions, because which build to run is the reader's next question.

        Args:
            found: The layout version stamped in the database.
            known: The layout version this build writes.
        """
        super().__init__(
            f"the store is at layout version {found} and this build knows version {known}: "
            "it was written by a later build, which may have moved columns this one would "
            "write past. Run that build, or point this one at another database."
        )


class StoreSchema(ABC):
    """What one backend supplies so the shared open sequence can run against it.

    Subclassed once per backend and constructed with the half of the database the caller
    owns — the operational tables, or the record index — because a deployment can open
    either without the other. The version, and the steps that move it, belong to the
    database as a whole and are therefore the same for both halves.
    """

    name: str
    """How this half of this backend's database is named in a log line."""

    def open(self) -> int:
        """Bring the database to :data:`STORE_SCHEMA_VERSION` and report the version.

        Returns:
            The stamped layout version, always the current one.

        Raises:
            StoreSchemaTooNew: When the database is stamped above what this build knows.
                Refused rather than opened read-only or migrated backwards: an older
                binary writing into a newer layout is how a corpus is quietly corrupted.
        """
        self.create_version_table()
        version = self.stamped_version()
        if version is None:
            # An unstamped database that already holds tables predates this mechanism, so
            # its layout has to be read off the tables themselves. The only such read.
            version = STORE_SCHEMA_VERSION if self.is_unused() else self.detect_version()
            self.stamp(version)
        elif version > STORE_SCHEMA_VERSION:
            raise StoreSchemaTooNew(version, STORE_SCHEMA_VERSION)
        self.create_layout()
        while version < STORE_SCHEMA_VERSION:
            _log.info(
                "migrating %s from store layout version %d to %d", self.name, version, version + 1
            )
            self.steps[version]()
            version += 1
        _log.info("%s is at store layout version %d", self.name, version)
        return version

    @abstractmethod
    def create_version_table(self) -> None:
        """Create the one-row version table, if it is not there."""

    @abstractmethod
    def create_layout(self) -> None:
        """Create this half's tables and indexes at the current version, if not there."""

    @abstractmethod
    def stamped_version(self) -> int | None:
        """Read the stored layout version, or None when nothing has stamped one."""

    @abstractmethod
    def is_unused(self) -> bool:
        """Report whether the database holds nothing but the version table."""

    @abstractmethod
    def detect_version(self) -> int:
        """Read the layout version off the tables of a database that carries no stamp."""

    @abstractmethod
    def stamp(self, version: int) -> None:
        """Record a layout version, replacing whatever is stored."""

    @property
    @abstractmethod
    def steps(self) -> Mapping[int, Callable[[], None]]:
        """The step that takes the database from each version to the one after it.

        A step applies its change and stamps the version it produces in one transaction,
        so a migration interrupted part-way leaves the database at the version it began
        at rather than between two. It is also written against the layout it produces:
        when the current DDL moves on past a step, that step keeps the statements it
        needs rather than following, or it will hand the next step a shape from the
        future.
        """
