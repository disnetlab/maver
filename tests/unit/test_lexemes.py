"""The shared normaliser, pinned against what PostgreSQL's english configuration does.

Every expectation here was read off a live PostgreSQL 17 server with

    SELECT array_agg(lexeme) FROM unnest(to_tsvector('english', <text>));

and is recorded rather than recomputed, so this file is the check that runs without one.
The classes PostgreSQL keeps whole and this does not are pinned too, in
``test_the_documented_residual_classes_are_split``: they are a deliberate simplification
and a test is the only thing that keeps a simplification from drifting into a bug.
"""

import pytest

from kira.memory.lexemes import STOPWORDS, lexemes


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Morphology and stopwords, the whole reason this module exists.
        ("the pods were crashing", ["pod", "crash"]),
        ("pod crash", ["pod", "crash"]),
        ("the and of", []),
        ("raising the memory limits", ["rais", "memori", "limit"]),
        ("misconfigured resolution failing", ["misconfigur", "resolut", "fail"]),
        # Case is folded before anything else.
        ("CrashLoopBackOff OOMKilled", ["crashloopbackoff", "oomkil"]),
        # A hyphenated compound: every part, then the whole, stemmed because every part
        # is alphabetic.
        ("hotel-reservation", ["hotel", "reserv", "hotel-reserv"]),
        ("otel-collector-agent", ["otel", "collector", "agent", "otel-collector-ag"]),
        # A compound with a digit in it reaches `simple`, so the whole is not stemmed.
        (
            "checkout-7d9f8b6c5-x2klm",
            ["checkout", "7d9f8b6c5", "x2klm", "checkout-7d9f8b6c5-x2klm"],
        ),
        # A hyphen in front of a bare number is that number's sign, and it ends the
        # compound rather than continuing it.
        ("pod-1 pod-2", ["pod", "-1", "pod", "-2"]),
        ("node-1-worker", ["node", "-1", "worker"]),
        ("123-abc", ["123", "abc"]),
        ("2026-07-15T02:52:00Z", ["2026", "-07", "-15", "t02", "52", "00z"]),
        # Numbers, versions and units pass through unstemmed.
        ("10.96.0.1", ["10.96.0.1"]),
        ("memory limit 512Mi cpu 500m", ["memori", "limit", "512mi", "cpu", "500m"]),
        ("p99 latency 250ms", ["p99", "latenc", "250ms"]),
        ("replicas=1 readyReplicas=0", ["replica", "1", "readyreplica", "0"]),
        # An underscore is a separator there, so it is one here.
        ("a_b", ["b"]),
        ("", []),
    ],
)
def test_the_lexemes_are_the_ones_postgresql_produces(text: str, expected: list[str]) -> None:
    assert lexemes(text) == expected


def test_the_documented_residual_classes_are_split() -> None:
    """The three classes the module docstring gives up on, held where they are.

    PostgreSQL keeps each of these as one lexeme. Splitting them can only widen a pool,
    and it un-hides an identifier that the measured pool could only match in full.
    """
    assert lexemes("deployment/checkout") == ["deploy", "checkout"]
    assert lexemes("RunContainerError/CrashLoopBackOff") == [
        "runcontainererror",
        "crashloopbackoff",
    ]
    assert lexemes("foo@bar.com") == ["foo", "bar", "com"]
    assert lexemes("v1.2.3") == ["v1", "2.3"]


def test_the_stopword_list_is_postgresql_s() -> None:
    """Vendored, so its size and its oddities are worth asserting rather than trusting."""
    assert len(STOPWORDS) == 127
    assert {"the", "and", "of", "was", "were", "not", "no"} <= STOPWORDS
    # The contraction fragments, which are why "don't" contributes nothing at all.
    assert {"s", "t", "don"} <= STOPWORDS


def test_a_query_and_a_record_meet_on_the_same_terms() -> None:
    """The property the whole design rests on: normalising twice is normalising once."""
    record = lexemes("The checkout pods were crashing repeatedly with OOMKilled")
    assert set(lexemes("pod crash")) <= set(record)
    assert set(lexemes("crashes")) <= set(record)
    assert set(lexemes("the of and was")) == set()
