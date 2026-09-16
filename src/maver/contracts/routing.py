"""Where the pipeline can go, and why it went back.

Both the roles and the orchestration above them need these, so they live at the bottom
of the stack rather than in the layer that does the routing. A role has to know it is
being re-entered — the text it is given differs — without importing the graph that
re-entered it.
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

__all__ = ["HandBack", "HandBackKind", "Route"]


class Route(StrEnum):
    """Where the graph goes next. The names are the node names, plus an end."""

    RCA = "rca"
    PLANNING = "planning"
    EXECUTION = "execution"
    VERIFY = "verify"
    END = "end"


HandBackKind = Literal["reassessment", "replan_from_verify", "replan_from_execution"]


@dataclass(frozen=True, slots=True)
class HandBack:
    """A back-edge: a role being entered again because a later stage sent it back.

    Derived from the artifacts rather than from a flag set when the edge was taken, so
    the router and the node that receives it cannot disagree, and so a run resumed from
    persisted state reaches the same conclusion as one that never stopped.
    """

    target: Route
    reason: str
    """Why it was sent back. Never empty — a hand-back with no stated reason gives the
    receiving role nothing to do differently, so it repeats itself."""
    kind: HandBackKind
