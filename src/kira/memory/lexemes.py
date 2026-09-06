"""The one normalisation both backends index and query with.

The candidate pool the offline campaign measured was PostgreSQL's: ``to_tsvector`` with
the ``english`` configuration over the raw record text, which is its default parser, its
127-word english stopword list and Snowball English (Porter2) stemming. BM25 then scored
the raw tokens of that pool in process. The embedded backend did none of it: FTS5's
``unicode61`` tokeniser splits on non-alphanumerics and stops there, so ``pod crash``
found "the pods were crashing" on one backend and nothing on the other, and a query of
nothing but stopwords matched the whole corpus rather than none of it.

Doing it here rather than in each backend is what makes the two identical *by
construction* rather than by inspection. Both store the terms this module produces beside
the raw text and index those, taking the terms as written: PostgreSQL's generated vectors
are built with ``array_to_tsvector`` rather than ``to_tsvector('simple', ...)``, whose
parser would split a hyphenated term again, and SQLite's FTS5 table keeps ``-`` and ``.``
as token characters for the same reason. An adopter who will not run PostgreSQL still
gets the retriever that was measured.

Scoring is unaffected and deliberately so: :mod:`kira.memory.bm25` keeps tokenising the
raw text, because raw tokens are what the published numbers scored.

**What is reproduced.** Alphabetic words, stemmed and stopword-filtered. Words mixing
letters and digits (``p99``, ``512mi``, ``x2klm``), integers, floats and dotted versions
(``10.96.0.1``) passed through unstemmed, as the ``simple`` dictionary does. Hyphenated
compounds yield their parts *and* the whole compound, so ``user-service`` is found by
``user`` and by ``user-service``; the whole is stemmed only when every part is
alphabetic, matching the parser's split between ``asciihword`` and ``numhword``. A
hyphen directly in front of a bare number ends the compound and the number keeps the
sign, because the parser reads ``pod-1`` as a word and a signed integer.

**What is not**, verified against a PostgreSQL 17 server over 400 strings of real
incident text from the campaign corpus. 52.5% of strings normalise to exactly the same
lexemes and 91.5% of distinct lexemes agree, and every one of the differences falls in
one of three classes:

* *Compound tokens the parser keeps whole* (151 strings). It recognises URLs, hosts,
  email addresses, file paths and URL paths as single tokens, so
  ``RunContainerError/CrashLoopBackOff`` is one lexeme there and two terms here,
  ``foo@bar.com`` one and three, ``v1.2.3`` one and two. Reproducing that grammar would
  be several hundred lines for classes a query either names in full or does not name at
  all, and splitting them can only widen a pool, never narrow it — under the old pool a
  query for ``crashloopbackoff`` did not reach a record that spelled it after a slash.
* *Host- and version-shaped tokens*, the same cause (56 strings).
* *Five words in 4,413* (5 strings): ``snowballstemmer`` 3.1.1 ships a later revision of
  the Snowball English algorithm than PostgreSQL 17 bundles, which stems ``internal`` to
  ``intern`` and ``added`` to ``ad`` where this one does not. Over the corpus's whole
  vocabulary the two stemmers agree on 99.82% of non-stopword words. Nothing here
  depends on which of the two is right, only that one of them is used on both sides.
"""

import re

import snowballstemmer

__all__ = ["STOPWORDS", "lexemes"]

# PostgreSQL's english.stop, verbatim, from share/tsearch_data/english.stop in the
# 17.11 distribution; it is the Snowball English stopword list, unchanged in every
# release this system has run against. Vendored rather than read off a server, because
# the embedded backend has no server to read it from, which is the whole point.
_ENGLISH_STOP = """
    i me my myself we our ours ourselves you your yours yourself yourselves he him his
    himself she her hers herself it its itself they them their theirs themselves what
    which who whom this that these those am is are was were be been being have has had
    having do does did doing a an the and but if or because as until while of at by for
    with about against between into through during before after above below to from up
    down in out on off over under again further then once here there when where why how
    all any both each few more most other some such no nor not only own same so than too
    very s t can will just don should now
    """

STOPWORDS = frozenset(_ENGLISH_STOP.split())

# A dotted number, or a run of letters and digits. Dotted first, so that an address or a
# version stays one token. Underscore is excluded because the parser treats it as blank:
# `a_b` is two words there, not one.
_TOKEN = re.compile(r"\d+(?:\.\d+)+|[^\W_]+", re.UNICODE)
_LEADING_NUMBER = re.compile(r"\d+(?:\.\d+)*")

_MAX_TERM_BYTES = 2046
"""PostgreSQL's lexeme ceiling, which it applies by ignoring anything longer.

Dropped here rather than in a backend so that both drop the same thing. A base64 blob or
a stack trace pasted into tool output produces runs this long, and a term neither backend
can be queried for is worth nothing to either.
"""

_stem = snowballstemmer.stemmer("english").stemWord


def lexemes(text: str) -> list[str]:
    """Normalise text into the terms both backends index and query.

    Args:
        text: Any text.

    Returns:
        The terms, in order of appearance, with repeats kept.
    """
    out: list[str] = []
    for compound in _compounds(text or ""):
        out.extend(term for term in _lexemes_of(compound) if not _too_long(term))
    return out


def _too_long(term: str) -> bool:
    # A character is at most four bytes, so a short term cannot exceed the byte ceiling
    # and does not need encoding to find out.
    return len(term) * 4 > _MAX_TERM_BYTES and len(term.encode("utf-8")) > _MAX_TERM_BYTES


def _compounds(text: str) -> list[list[str]]:
    """Group tokens into the hyphenated compounds the default parser would build."""
    lowered = text.lower()
    groups: list[list[str]] = []
    current: list[str] = []
    joinable = False
    end = 0
    for match in _TOKEN.finditer(lowered):
        token = match.group()
        hyphenated = lowered[end : match.start()] == "-"
        end = match.end()
        if hyphenated and joinable and not _is_number(token):
            current.append(token)
            continue
        if current:
            groups.append(current)
            current = []
        # `pod-1`, `2026-07-15`: a hyphen in front of a number is that number's sign,
        # and whatever follows the digits starts a token of its own.
        leading = _LEADING_NUMBER.match(token) if hyphenated else None
        if leading:
            groups.append(["-" + leading.group()])
            token = token[leading.end() :]
            if not token:
                joinable = False
                continue
        current = [token]
        # A number cannot open a compound: `123-abc` is two tokens, `abc-123` is not.
        joinable = not _is_number(token)
    if current:
        groups.append(current)
    return groups


def _lexemes_of(parts: list[str]) -> list[str]:
    """Turn one compound into its lexemes: every part, plus the whole if it is one."""
    out = [term for part in parts if (term := _part(part))]
    if len(parts) > 1:
        whole = "-".join(parts)
        # A compound with a digit anywhere in it is a `numhword`, which the english
        # configuration hands to `simple` rather than to the stemmer.
        out.append(whole if any(_has_digit(part) for part in parts) else _stem(whole))
    return out


def _part(token: str) -> str:
    if _has_digit(token):
        return token
    return "" if token in STOPWORDS else _stem(token)


def _is_number(token: str) -> bool:
    match = _LEADING_NUMBER.match(token)
    return match is not None and match.end() == len(token)


def _has_digit(token: str) -> bool:
    return any(character.isdigit() for character in token)
