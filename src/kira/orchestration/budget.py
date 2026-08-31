"""One retry budget, shared by every loop that can send the pipeline backwards.

Three back-edges exist — a failed apply returning to planning, a wrong cause returning to
diagnosis, an insufficient plan returning to planning — and they draw on one budget rather
than three. Separate budgets would let an incident cycle far longer than any single number
suggests, and the published description of the system states one bound.

A value over three keys of the state, never stored as an object. Reconstructing it from
the state on each use is what makes a run resumed from a database reach the same
conclusion as one that never stopped.
"""

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from kira.contracts.state import IncidentState
from kira.roles.base import StateDelta

if TYPE_CHECKING:
    from kira.orchestration.routing import RoutingPolicy

__all__ = ["RetryBudget"]


@dataclass(frozen=True, slots=True)
class RetryBudget:
    """What the incident has spent, and what it may still spend."""

    remediation_spent: int
    reassessments: int
    """A telemetry subset of ``remediation_spent``, not a separate allowance."""
    verify_rechecks: int
    max_remediation: int
    max_rechecks: int

    @classmethod
    def of(cls, state: IncidentState, policy: "RoutingPolicy") -> "RetryBudget":
        """Read the budget out of the incident state.

        Args:
            state: The incident state.
            policy: The configured limits.

        Returns:
            The budget as it currently stands.
        """
        return cls(
            remediation_spent=state.get("remediation_retries", 0),
            reassessments=state.get("reassessment_count", 0),
            verify_rechecks=state.get("verify_retries", 0),
            max_remediation=policy.remediation_retries,
            max_rechecks=policy.inconclusive_retries,
        )

    def to_state(self) -> StateDelta:
        """Render the budget back as a state delta.

        Returns:
            The three counter keys.
        """
        return {
            "remediation_retries": self.remediation_spent,
            "reassessment_count": self.reassessments,
            "verify_retries": self.verify_rechecks,
        }

    @property
    def remediation_left(self) -> bool:
        """Whether another cluster-mutating attempt is allowed.

        Returns:
            True while the shared budget is unspent.
        """
        return self.remediation_spent < self.max_remediation

    @property
    def recheck_left(self) -> bool:
        """Whether verification may look once more.

        Returns:
            True while the re-check allowance is unspent.
        """
        return self.verify_rechecks < self.max_rechecks

    def spend_replan(self) -> "RetryBudget":
        """Charge a return to planning.

        Returns:
            The budget after the charge.
        """
        return replace(self, remediation_spent=self.remediation_spent + 1)

    def spend_reassessment(self) -> "RetryBudget":
        """Charge a return to diagnosis.

        It costs the same shared budget as a replan and additionally counts itself, so
        how often the system re-diagnosed is answerable without changing what it costs.

        Returns:
            The budget after the charge.
        """
        return replace(
            self,
            remediation_spent=self.remediation_spent + 1,
            reassessments=self.reassessments + 1,
        )

    def spend_recheck(self) -> "RetryBudget":
        """Charge a second look by verification.

        Kept off the remediation budget deliberately: that budget exists to bound
        *mutations*, and looking again changes nothing.

        Returns:
            The budget after the charge.
        """
        return replace(self, verify_rechecks=self.verify_rechecks + 1)

    def reset_rechecks(self) -> "RetryBudget":
        """Clear the re-check count at the start of a new remediation cycle.

        A hand-back begins a fresh cycle whose verification has not re-checked anything
        yet. Without this the counter carries across cycles and the second cycle's
        verification cannot look twice even though it has never looked twice.

        Returns:
            The budget with re-checks zeroed.
        """
        return replace(self, verify_rechecks=0)
