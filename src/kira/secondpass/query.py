"""Composing the retrieval queries, from observations and never from a conclusion.

Two legs, not one. Algorithm 1 in the manuscript says ``q <- Query(a, obs(r))``; the
mechanism has always searched twice and merged, and the reason is a specific failure.
The evidence leg carries the first run's concrete error strings but inherits its anchor
— the kafka poison-pill shakeout, where a grafana-anchored run produced a grafana-shaped
query and the poison-pill precedents were never served. The pre-anchor leg is built from
the alert and the namespace snapshot alone, formed before any hypothesis exists, so it
cannot inherit one. Merging the two top-k lists reaches both neighbourhoods without the
measured dilution cost of a single oversized query.

Both legs are pure functions with an explicit length budget, and neither reads settings:
a single-leg ablation is a policy value rather than a code edit.

The pre-anchor leg has a live defect that this module bounds rather than hides. The
query it produced in the campaign ran to roughly 206 terms and 2,360 characters while
the model's own retrieval queries average 8 terms and 89 characters, and the direction
of that effect reverses by corpus: on one it retrieved below chance at 0.4x a random
baseline and a tail trim lifted it to 3.2x, on another it was already 7.3x and the trim
was neutral. A pooled claim about it is Simpson's paradox and was retracted. So the
budget is a constructor argument rather than a knob, and its defaults are what the
campaign ran on: a 3,600-character cap with the boilerplate filter off, and the
composition reproducing the published cells passes the twelve observation lines that
campaign passed. The screened mitigations are each one argument away, and turning one on
is a different experiment rather than a repair to this one.
"""

import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from kira.contracts.artifacts import RootCause
from kira.secondpass.types import NamedQuery, QueryBuilder, SecondPassContext

__all__ = ["MergedQuery", "ObservationQuery", "PreAnchorQuery"]

_log = logging.getLogger(__name__)

_OBSERVATION_SIGNAL = re.compile(
    r"(?i)(alert|error|fail|timeout|refused|unavailable|not.?ready|restart|"
    r"warning|pressure|oom|latency|5\d\d|degraded|crash|evict|endpoint)"
)
_BOILERPLATE = re.compile(
    r"\(unavailable\)|\(no Warning events|WORKLOADS \(kind/name|"
    r"SERVICES / BINDINGS \(selector|RECENT CHANGES \(Warning events|"
    r"Deterministic read-only starting facts|unable to enforce \w+ namespace scope"
)


def _identifier(alert: Mapping[str, Any], key: str) -> str:
    return str(alert.get(key) or "").strip()


@dataclass(frozen=True, slots=True)
class ObservationQuery:
    """Identifiers plus the first run's own observations, never its conclusion.

    Searching on the conclusion builds an echo chamber around a possibly-wrong answer,
    so ``fault_type``, ``affected_resource``, ``summary`` and ``confidence`` are
    excluded by construction rather than by filtering.
    """

    name: str = "evidence"
    line_chars: int = 160
    max_chars: int = 1200

    async def build(self, ctx: SecondPassContext, first: RootCause) -> list[NamedQuery]:
        """Build the evidence-keyed query.

        Args:
            ctx: The pass's context.
            first: The diagnosis whose evidence lines are used.

        Returns:
            One query, possibly empty when the alert and the evidence are both bare.
        """
        lines: list[str] = []
        for key in ("alertname", "namespace", "service"):
            if value := _identifier(ctx.alert, key):
                lines.append(f"{key}={value}")
        for item in first.evidence or []:
            text = " ".join(str(item).split())
            if text:
                lines.append(text[: self.line_chars])
        return [NamedQuery(self.name, "\n".join(lines)[: self.max_chars])]


@dataclass(frozen=True, slots=True)
class PreAnchorQuery:
    """Identifiers and a live namespace snapshot, formed before any hypothesis exists.

    A snapshot failure degrades the query to identifiers alone rather than failing the
    pass. That is more common than it should be: the auto-collected snapshot came back
    empty in roughly 93% of runs in every measured campaign, which starves the one leg
    that is supposed to be hypothesis-free. The defect belongs to the snapshot tool;
    the degradation is handled here so the pass survives it.
    """

    name: str = "pre_anchor"
    observation_lines: int = 8
    """Eight is the prototype's own default and the offline champion's budget. The
    measured MA+S campaign called the builder with twelve, so the composition that
    reproduces those cells passes twelve; the A/B that would settle which is better was
    specified and never run."""
    max_chars: int = 3600
    """The cap the campaign ran under, kept because a tighter budget is the obvious
    lever and the wrong one: filtering the query scored better than truncating it, and
    stacking truncation on the filter changed nothing."""
    drop_boilerplate: bool = False
    """Off, because that is how every published cell was retrieved.

    The filter is a screened mitigation, not a repair already banked. 84.6% of the
    campaign query's body was section headers and empty-state markers that the
    observation-signal regex admits, and dropping them scored 0.116 same-family
    precedents per retrieved memory against 0.011 as deployed. The live A/B was written
    and staged on two hosts and never ran, so the direction of the effect on a diagnosis
    is still unmeasured."""
    line_chars: int = 360

    async def build(self, ctx: SecondPassContext, first: RootCause) -> list[NamedQuery]:
        """Build the hypothesis-free query.

        Args:
            ctx: The pass's context.
            first: Unused. The whole point of this leg is that it reads nothing the
                first run concluded.

        Returns:
            One query.
        """
        del first
        try:
            snapshot = await ctx.snapshot()
        except Exception as exc:
            _log.warning("snapshot for the pre-anchor query failed: %s", exc, exc_info=exc)
            snapshot = ""

        lines = ["Current incident observations:"]
        identifiers: list[str] = []
        for key in ("alertname", "namespace", "service", "severity"):
            if value := _identifier(ctx.alert, key):
                identifiers.append(f"{key}={value}")
        labels = ctx.alert.get("labels")
        labels = labels if isinstance(labels, dict) else {}
        for key in ("alertname", "namespace", "service", "deployment", "pod", "job"):
            value = str(labels.get(key) or "").strip()
            item = f"{key}={value}" if value else ""
            if item and item not in identifiers:
                identifiers.append(item)
        if identifiers:
            lines.append("Identifiers: " + "; ".join(identifiers))
        lines += [f"Snapshot: {line}" for line in self._observations(snapshot)]
        return [NamedQuery(self.name, "\n".join(lines)[: self.max_chars].rstrip())]

    def _observations(self, text: str) -> list[str]:
        """Select bounded factual lines without asking a model to summarise them."""
        lines = [
            " ".join(line.split())[: self.line_chars] for line in text.splitlines() if line.strip()
        ]
        signalled = [line for line in lines if _OBSERVATION_SIGNAL.search(line)]
        if self.drop_boilerplate:
            filtered = [line for line in signalled if not _BOILERPLATE.search(line)]
            # An entirely boilerplate snapshot keeps its lines rather than being erased:
            # a filter that can empty the query is worse than one that lets noise past.
            signalled = filtered or signalled
        return signalled[: self.observation_lines] or lines[: min(3, self.observation_lines)]


class MergedQuery:
    """Runs several legs and concatenates their queries in declaration order.

    Deduplication is the retriever's job, not this one's, so dropping a leg is a change
    to a policy value and nothing else.
    """

    __slots__ = ("legs",)

    def __init__(self, *legs: QueryBuilder) -> None:
        """Compose the legs.

        Args:
            *legs: The query builders to run, in the order their results are merged.
        """
        self.legs: tuple[QueryBuilder, ...] = tuple(legs)

    async def build(self, ctx: SecondPassContext, first: RootCause) -> list[NamedQuery]:
        """Build every leg's queries.

        Args:
            ctx: The pass's context.
            first: The first opinion.

        Returns:
            The concatenated queries, in declaration order.
        """
        queries: list[NamedQuery] = []
        for leg in self.legs:
            queries.extend(await leg.build(ctx, first))
        return queries
