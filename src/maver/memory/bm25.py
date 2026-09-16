"""Okapi BM25, the ranker an offline campaign found to be the ceiling on this corpus.

Pure lexical BM25 at ``k1=1.2, b=1.0`` beat six dense embedders, two cross-encoders and
a model reranker. That is an unusual result and worth stating plainly rather than
apologising for: incident records are short, dense with identifiers, and matched far more
often by a service name or an error string than by anything a sentence embedding captures.

Reimplemented rather than taken as a dependency. The formula, including the negative-IDF
floor, follows ``rank_bm25.BM25Okapi``, and a test pins it against that library so the two
cannot silently drift.

Ranking happens in process. Corpora here are hundreds of records, so scoring them
directly is cheap and — more importantly — reuses the exact scoring the offline work
measured. A database's own text ranking cannot: PostgreSQL's ``ts_rank`` weights term
frequency and position with no inverse-document-frequency term at all, and SQLite's FTS5
BM25 uses a different length normalisation. A backend that ranks with its own function
is a different retriever, whatever it is called.
"""

import math
import re
from collections import Counter
from collections.abc import Sequence

__all__ = ["DEFAULT_B", "DEFAULT_EPSILON", "DEFAULT_K1", "rank", "scores", "tokenize"]

# Lowercased alphanumeric runs, the tokenisation the offline harness scored with, over
# the raw record text. Deliberately no stemming and no stopword list: maver.memory.lexemes
# applies both once, for both backends, when the candidate pool is built, and applying
# them again here would score different text than was measured.
_TOKEN = re.compile(r"[a-z0-9]+")

DEFAULT_K1 = 1.2
DEFAULT_B = 1.0
"""Full length normalisation, rather than the usual 0.75. Measured, not inherited."""
DEFAULT_EPSILON = 0.25


def tokenize(text: str) -> list[str]:
    """Split text into scoring tokens.

    Args:
        text: Any text.

    Returns:
        Lowercased alphanumeric runs.
    """
    return _TOKEN.findall((text or "").lower())


def scores(
    corpus: Sequence[Sequence[str]],
    query: Sequence[str],
    *,
    k1: float = DEFAULT_K1,
    b: float = DEFAULT_B,
    epsilon: float = DEFAULT_EPSILON,
) -> list[float]:
    """Score every document against a query.

    A term appearing in more than half the corpus gets a negative inverse document
    frequency under the Okapi formula, which would let a ubiquitous word *subtract* from
    relevance. Those are floored instead. That matters here specifically: incident
    records share a great deal of boilerplate, so without the floor the words every
    record contains would actively push the right record down.

    Args:
        corpus: Tokenised documents.
        query: Tokenised query.
        k1: Term-frequency saturation.
        b: Length normalisation.
        epsilon: Floor for negative inverse document frequency, as a fraction of the mean.

    Returns:
        One score per document, in corpus order.
    """
    n = len(corpus)
    if n == 0 or not query:
        return [0.0] * n

    lengths = [len(doc) for doc in corpus]
    average = sum(lengths) / n
    frequencies = [Counter(doc) for doc in corpus]

    document_frequency: Counter[str] = Counter()
    for counts in frequencies:
        document_frequency.update(counts.keys())

    idf: dict[str, float] = {}
    negatives: list[str] = []
    total = 0.0
    for word, freq in document_frequency.items():
        value = math.log(n - freq + 0.5) - math.log(freq + 0.5)
        idf[word] = value
        total += value
        if value < 0:
            negatives.append(word)
    if idf:
        floor = epsilon * (total / len(idf))
        for word in negatives:
            idf[word] = floor

    out = [0.0] * n
    for i, counts in enumerate(frequencies):
        norm = k1 * (1 - b + b * (lengths[i] / average if average else 0.0))
        score = 0.0
        for term in query:
            tf = counts.get(term, 0)
            if not tf:
                continue
            score += idf.get(term, 0.0) * (tf * (k1 + 1) / (tf + norm))
        out[i] = score
    return out


def rank(
    documents: Sequence[str],
    query: str,
    *,
    k1: float = DEFAULT_K1,
    b: float = DEFAULT_B,
    epsilon: float = DEFAULT_EPSILON,
) -> list[tuple[int, float]]:
    """Rank documents against a query.

    Args:
        documents: The documents.
        query: The query text.
        k1: Term-frequency saturation.
        b: Length normalisation.
        epsilon: Floor for negative inverse document frequency.

    Returns:
        ``(index, score)`` pairs, best first. Ties keep input order.
    """
    corpus = [tokenize(document) for document in documents]
    values = scores(corpus, tokenize(query), k1=k1, b=b, epsilon=epsilon)
    order = sorted(range(len(documents)), key=lambda i: (-values[i], i))
    return [(i, values[i]) for i in order]
