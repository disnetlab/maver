"""How candidate records are scored, and how another way of scoring them is added.

Retrieval is three steps: a backend narrows the corpus to a candidate pool, something
scores that pool against the query, and the caller decides what to do with the scores.
Only the middle step is a research question, and it is the one most likely to be
replaced — by a dense embedder, a hybrid of the two, a cross-encoder, a reranking model.
So it is the one behind a seam.

A ranker decides relevance as well as order, because only the thing that produced a
score knows what that score means. BM25 scores are unbounded and scale with query
length; a cosine similarity is in [-1, 1]; a cross-encoder emits a logit. A single
threshold in the caller cannot serve all three, and one that tried would silently mean
something different for each. So a ranker returns what it considers worth returning, and
returning fewer results than it was given documents is how it says the rest are not
comparable.

An in-tree ranker registers with the decorator below. One shipped elsewhere declares an
entry point in the ``kira.rankers`` group and needs no fork.
"""

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import entry_points
from typing import Protocol, runtime_checkable

from kira.memory import bm25
from kira.settings.groups import MemorySettings

__all__ = [
    "Bm25Ranker",
    "Ranker",
    "RankerFactory",
    "available_rankers",
    "build_ranker",
    "clear_rankers",
    "get_ranker",
    "load_entry_point_rankers",
    "ranker",
    "registered_rankers",
]

_log = logging.getLogger(__name__)

_READY = "ready"


@runtime_checkable
class Ranker(Protocol):
    """Scores a pool of candidate documents against one query."""

    def rank(self, query: str, documents: Sequence[str]) -> Sequence[tuple[int, float]]:
        """Score and order the pool.

        Args:
            query: The query text.
            documents: The candidate documents, in pool order.

        Returns:
            ``(index, score)`` pairs into ``documents``, best first, containing only the
            candidates this ranker considers relevant. An empty result is a real answer:
            it says nothing in the pool is comparable, which is more useful to a role
            than five unrelated incidents.
        """
        ...


@dataclass(frozen=True, slots=True)
class Bm25Ranker:
    """Okapi BM25 with a relevance floor. The shipped ranker.

    Lexical ranking is the ceiling on this corpus: pure BM25 beat six dense embedders,
    two cross-encoders and a model reranker. That is a claim about short records dense
    with service names, error strings and resource identifiers, not about retrieval in
    general — which is why the seam this sits behind exists.
    """

    min_score_per_token: float = 0.5
    """Floor on a hit's score divided by the query's token count.

    Divided by token count so the threshold does not move with query length. Zero
    disables the floor, which restores always returning the whole pool — what the
    prototype shipped, and what an arm reproducing one of its campaigns pins through
    ``memory.min_score_per_token``.
    """
    k1: float = bm25.DEFAULT_K1
    b: float = bm25.DEFAULT_B
    epsilon: float = bm25.DEFAULT_EPSILON

    def rank(self, query: str, documents: Sequence[str]) -> Sequence[tuple[int, float]]:
        """Rank the pool and cut it at the floor.

        Args:
            query: The query text.
            documents: The candidate documents.

        Returns:
            ``(index, score)`` pairs above the floor, best first.
        """
        tokens = bm25.tokenize(query)
        if not tokens or not documents:
            # Not the same as "everything scores zero". With an empty query every score
            # is zero and a zero floor admits all of them, which is the whole pool
            # returned in arbitrary order under the name of a search.
            return []
        floor = self.min_score_per_token * len(tokens)
        ranked = bm25.rank(list(documents), query, k1=self.k1, b=self.b, epsilon=self.epsilon)
        kept: list[tuple[int, float]] = []
        for index, score in ranked:
            if score < floor:
                break  # ranked best first, so the first one under the floor ends it
            kept.append((index, score))
        return kept


RankerFactory = Callable[[MemorySettings], Ranker]

_REGISTRY: dict[str, RankerFactory] = {}
_UNAVAILABLE: dict[str, str] = {}


def ranker(name: str) -> Callable[[RankerFactory], RankerFactory]:
    """Register a ranker factory under a name.

    Args:
        name: The name a deployment selects it by, through ``memory.ranker``.

    Returns:
        A decorator that registers and returns the factory.

    Raises:
        ValueError: When the name is taken. A silent overwrite would make a run report
            one ranker and retrieve with another.
    """

    def register(factory: RankerFactory) -> RankerFactory:
        if name in _REGISTRY:
            raise ValueError(f"ranker {name!r} is already registered")
        _REGISTRY[name] = factory
        return factory

    return register


@ranker("bm25")
def _build_bm25(settings: MemorySettings) -> Ranker:
    """Build the shipped ranker from the memory group."""
    return Bm25Ranker(min_score_per_token=settings.min_score_per_token)


def load_entry_point_rankers(group: str = "kira.rankers") -> None:
    """Import every distribution registering in the ranker entry-point group.

    A module that raises is recorded and skipped, so one broken third-party package
    cannot stop an incident service from starting. Selecting a ranker that could not be
    loaded is fatal, and says why.

    Args:
        group: The entry-point group to load.
    """
    for point in entry_points(group=group):
        try:
            point.load()
        except Exception as exc:
            _UNAVAILABLE[point.name] = f"{type(exc).__name__}: {exc}"
            _log.warning("ranker %s could not be loaded: %s", point.name, exc)


def get_ranker(name: str) -> RankerFactory:
    """Look up one ranker's factory.

    Args:
        name: The registered name.

    Returns:
        The factory.

    Raises:
        ValueError: When no such ranker is available, naming the import failure when
            there was one. Falling back to the default would make a deployment report
            the ranker it asked for and retrieve with a different one.
    """
    factory = _REGISTRY.get(name)
    if factory is not None:
        return factory
    reason = _UNAVAILABLE.get(name)
    if reason is not None:
        raise ValueError(f"ranker {name!r} is installed but failed to load: {reason}")
    known = ", ".join(sorted(_REGISTRY)) or "none"
    raise ValueError(f"unknown ranker {name!r}; registered: {known}")


def build_ranker(settings: MemorySettings) -> Ranker:
    """Build the ranker this deployment configured.

    Args:
        settings: The memory group.

    Returns:
        The ranker.
    """
    return get_ranker(settings.ranker)(settings)


def registered_rankers() -> Mapping[str, RankerFactory]:
    """Every ranker that loaded.

    Returns:
        Name to factory.
    """
    return dict(_REGISTRY)


def available_rankers() -> Mapping[str, str]:
    """Report what a deployment can select, and why anything else cannot be.

    Returns:
        Name to ``ready`` or the reason it could not be loaded.
    """
    return {**dict.fromkeys(_REGISTRY, _READY), **_UNAVAILABLE}


def clear_rankers() -> None:
    """Empty the registry, keeping the shipped one. For tests registering throwaways."""
    _REGISTRY.clear()
    _UNAVAILABLE.clear()
    _REGISTRY["bm25"] = _build_bm25
