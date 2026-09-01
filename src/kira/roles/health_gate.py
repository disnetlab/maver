"""The namespace health gate: a computed recovery floor that Verify cannot reason past.

Verify's recall on real mitigation failures was 25%. It declared `recovered` on 46% of the
cells the mitigation oracle says were still broken, with mean confidence 0.93 —
statistically identical to its correct calls. Reading 22 of those transcripts showed the
evidence was almost always already in Verify's own tool output, 16 of 22 catchable, and was
then written off, frequently citing the prompt's own incident-relevance lock: "not causally
linked by RCA", "outside RCA scope", "belongs to an old ReplicaSet not involved in the root
cause chain". One cell dismissed 64 Pending pods; another dismissed a CrashLoopBackOff with
24 restarts in 105s on a cell whose injected fault was an over-aggressive liveness probe.

The common factor is that recovery was judged against RCA's causal chain. When the diagnosis
is wrong — 18 of those 25 misses — the real fault is by construction outside that chain, so
the chain-relative test cannot see it.

So this gate is computed here rather than asked for in the prompt, and enforced in Verify's
submit guard rather than described to the model. It answers one hypothesis-free question: is
anything in the incident namespace unhealthy right now? A failure blocks the `recovered`
verdict outright; the model may still argue in its summary, it just cannot close the
incident.

Asymmetry is deliberate. A false block costs one extra loop pass — of the cells that took a
back-edge, none were damaged by it and three had a wrong diagnosis corrected. A false pass
closes an unresolved incident, which is the failure this exists to stop. So the gate is
biased toward blocking and its signals are chosen to be objective, not interpretive.

It lives with the roles rather than under ``tools/`` because it is verification policy, not
something the model can reach: Verify never chooses to run it and never sees it in a schema.
"""

import ast
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from kira.prompts.loader import read_runtime, render
from kira.tools.context import ToolContext
from kira.tools.errors import call_failed
from kira.tools.kubernetes import read_kubectl

__all__ = ["HealthGate", "sweep_namespace_health"]

_log = logging.getLogger(__name__)

# Restart count at which a pod that is otherwise Ready is treated as still failing.
#
# Calibrated against live runs rather than guessed. At 3 this false-blocked a genuinely
# recovered cell: astronomy-shop's `accounting` pod reaches 4 restarts as ordinary startup
# churn, and the gate held `recovered` hostage across all three verify passes until the
# retry budget was spent — on a cell whose mitigation the oracle passed. The crash loop this
# rule exists to catch ran to 24 restarts, so 10 keeps that signal while clearing the noise.
#
# Only Ready pods are judged by this number; anything not Running or not Ready already fails
# above, so the threshold governs the narrow "Ready in the instant we sampled it, but still
# cycling" case and should stay well clear of normal churn.
_RESTART_CEILING = 10

# Waiting reasons that a healthy pod passes through on its way up. They are only evidence of
# a fault once they PERSIST: a pod stuck initialising for minutes is a hung dependency, the
# same pod at twenty seconds is an ordinary rollout. Blocking on the transient case cost a
# false block on duplicate_pvc_mounts_hotel_reservation, where the gate caught a frontend pod
# mid-restart and held a correctly-mitigated cell for an extra loop.
_TRANSIENT_WAITING = frozenset({"ContainerCreating", "PodInitializing"})
_TRANSIENT_GRACE_SECONDS = 90

# How many failures the persisted record keeps. The count is exact either way; the list is
# there to say what kind of thing blocked, not to be a second copy of the namespace.
_AUDIT_FAILURES = 10


def _indent(text: str) -> list[str]:
    """Indent a rendered block by the two spaces the sweep's body carries."""
    return [f"  {line}" for line in text.splitlines()]


def _bullets(failures: tuple[str, ...], indent: str) -> str:
    return "\n".join(f"{indent}- {failure}" for failure in failures)


@dataclass(frozen=True, slots=True)
class HealthGate:
    """The result of one namespace sweep.

    Attributes:
        checked: Whether the sweep actually ran. False means the gate could not look.
        failures: One line per unhealthy pod, in the words the model reads.
        error: Why the sweep could not run, when it could not.
        namespace: The namespace swept.
        observations: Findings that are explicitly *not* failures, such as a pod inside
            the transient grace, kept so a non-block is legible rather than silent.
    """

    checked: bool = False
    failures: tuple[str, ...] = ()
    error: str | None = None
    namespace: str = ""
    observations: tuple[str, ...] = ()

    @property
    def blocks_recovered(self) -> bool:
        """Whether this gate withholds the `recovered` verdict.

        Only a gate that actually ran and found a failure may block a verdict. An
        unavailable gate must never manufacture a `not_recovered`: that would convert an
        infrastructure fault into a false incident and burn the retry budget on nothing.

        Returns:
            True when the sweep ran and found at least one failure.
        """
        return self.checked and bool(self.failures)

    def context_text(self) -> str:
        """Render the sweep as the block prepended to Verify's opening turn.

        Empty when the gate did not run, so a sweep that could not look adds nothing to
        the conversation rather than telling Verify to go and establish health itself.
        Verify is already told to survey the namespace, and a block that appears only on
        the gate's bad days is a variable in the prompt nobody measured.

        Returns:
            The block, or the empty string when the gate did not run.
        """
        if not self.checked:
            return ""
        header = render(
            read_runtime("verify", "health_gate_header.txt"),
            {"NAMESPACE": f"{self.namespace!r}"},
        )
        if self.failures:
            body = render(
                read_runtime("verify", "health_gate_unhealthy.txt"),
                {
                    "COUNT": str(len(self.failures)),
                    "FAILURES": _bullets(self.failures, "  "),
                },
            )
        else:
            body = read_runtime("verify", "health_gate_healthy.txt")
        notes = [f"  note: {observation}" for observation in self.observations]
        return "\n".join([header, *_indent(body), *notes])

    def rejection_text(self) -> str:
        """Render the guard's refusal of a `recovered` submission.

        Returns:
            The rejection the model reads as a turn, so it can revise rather than fail.
        """
        return render(
            read_runtime("verify", "submit_while_unhealthy.txt"),
            {"FAILURES": _bullets(self.failures, "  ")},
        )

    def audit(self) -> dict[str, Any]:
        """Build the persistable record of what the gate saw.

        The eval container is recreated per cell, so its logs are gone by the time a run is
        analysed — leaving "gate ran and the namespace was healthy" indistinguishable from
        "gate failed and fell open", which are opposite conclusions about the same absence
        of blocks. This goes on the run row so the distinction survives.

        Returns:
            A JSON-serialisable summary of the sweep.
        """
        return {
            "checked": self.checked,
            "namespace": self.namespace,
            "failure_count": len(self.failures),
            "failures": list(self.failures[:_AUDIT_FAILURES]),
            "error": self.error,
            "blocked_recovered": self.blocks_recovered,
        }


def _pod_age_seconds(item: dict[str, Any], now: datetime | None = None) -> float | None:
    """Seconds since the pod was created, or None when the timestamp is unusable."""
    stamp = (item.get("status") or {}).get("startTime") or (item.get("metadata") or {}).get(
        "creationTimestamp"
    )
    if not isinstance(stamp, str):
        return None
    try:
        created = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except ValueError:
        return None
    return ((now or datetime.now(UTC)) - created).total_seconds()


def _parse_items(raw: str) -> tuple[list[Any] | None, str]:
    """Read the pod list out of one ``kubectl get pods -o json``.

    The JSON parse falls back to :func:`ast.literal_eval` because a transport can hand back
    the Python repr of the same object rather than JSON, and a gate that cannot parse its
    own read falls open on a formatting difference.

    Returns:
        The items and an empty note, or None and the note explaining what was unusable.
    """
    payload: Any
    try:
        payload = json.loads(raw)
    except (ValueError, TypeError):
        try:
            payload = ast.literal_eval(raw)
        except (SyntaxError, ValueError, TypeError, MemoryError, RecursionError):
            return None, f"pod inventory unparseable ({raw.strip()[:120]})"
    items = payload.get("items") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        return None, "pod inventory had no item list"
    return items, ""


def _waiting_reason(statuses: list[Any]) -> str:
    for container in statuses:
        waiting = (
            (container.get("state") or {}).get("waiting") if isinstance(container, dict) else None
        )
        if isinstance(waiting, dict) and waiting.get("reason"):
            return str(waiting["reason"])
    return ""


def _pod_failures(raw: str, now: datetime | None = None) -> tuple[list[str] | None, list[str]]:
    """Extract the unhealthy pods from ``kubectl get pods -o json``.

    ``now`` is injectable so the transient-state grace period is testable without sleeping.

    Returns:
        The failure lines and any notes, or None for the failures when the inventory
        could not be read at all — which is not the same as an inventory with nothing
        wrong in it.
    """
    items, note = _parse_items(raw)
    if items is None:
        return None, [note]

    failures: list[str] = []
    notes: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        meta = item.get("metadata") or {}
        status = item.get("status") or {}
        name = meta.get("name") or "<unnamed>"
        # A pod being deleted is mid-rollout, not a fault.
        if meta.get("deletionTimestamp"):
            continue
        phase = status.get("phase")
        if phase == "Succeeded":
            continue
        statuses = status.get("containerStatuses") or []
        restarts = max(
            (c.get("restartCount", 0) for c in statuses if isinstance(c, dict)), default=0
        )
        if phase != "Running":
            reason = _waiting_reason(statuses)
            age = _pod_age_seconds(item, now)
            # A pod still coming up is only a fault once it has been coming up too long.
            if reason in _TRANSIENT_WAITING and age is not None and age < _TRANSIENT_GRACE_SECONDS:
                notes.append(f"pod {name} is {reason} but only {int(age)}s old — still starting")
                continue
            suffix = f" ({reason})" if reason else ""
            if reason in _TRANSIENT_WAITING and age is not None:
                suffix += f" for {int(age)}s"
            failures.append(f"pod {name} is {phase}{suffix}, not Running")
            continue
        not_ready = [
            c.get("name") for c in statuses if isinstance(c, dict) and not c.get("ready", False)
        ]
        if not_ready:
            failures.append(
                f"pod {name} is Running but container(s) {', '.join(str(n) for n in not_ready)} "
                "are not Ready"
            )
            continue
        if restarts > _RESTART_CEILING:
            failures.append(
                f"pod {name} is Ready but has restarted {restarts} times — a container that "
                "keeps restarting is not a recovered workload"
            )
    return failures, notes


async def sweep_namespace_health(ctx: ToolContext, namespace: str) -> HealthGate:
    """Sweep one namespace for anything unhealthy, hypothesis-free.

    Swept before the model sees anything, so the namespace's real state is established
    independently of RCA's chain rather than through it. It never raises: a broken gate must
    degrade to unavailable, never break Verify.

    The namespace is a parameter rather than a scope lookup because the caller already holds
    it, and a policy that reads ambient state cannot be called from a test without arranging
    that state first.

    Args:
        ctx: The role run's context, supplying the read-only cluster path.
        namespace: The incident namespace.

    Returns:
        The gate. ``checked=False`` whenever the sweep could not run, for any reason.
    """
    ns = namespace.strip()
    if not ns:
        _log.warning("health gate: no incident namespace supplied — gate inert")
        return HealthGate(checked=False, error="no incident namespace in scope")
    try:
        raw = await read_kubectl(ctx, f"kubectl get pods -n {ns} -o json")
    except Exception as exc:  # a gate that raised would take the verdict down with it
        _log.warning("health gate: pod sweep failed for ns=%s: %s", ns, exc, exc_info=exc)
        return HealthGate(checked=False, error=f"{type(exc).__name__}: {exc}"[:200], namespace=ns)
    # Both failure shapes, not just the prefix. A read that ran and exited non-zero says so
    # with its exit code, and reading past that left the gate on the fail-open path with
    # `checked=True`: an assertion of health about a namespace it had never seen.
    if call_failed(raw.strip()):
        _log.warning("health gate: pod sweep could not read ns=%s: %s", ns, raw.strip()[:160])
        return HealthGate(checked=False, error=raw.strip()[:200], namespace=ns)

    failures, notes = _pod_failures(raw)
    if failures is None:
        # A body it could not read is a sweep that did not happen. Counting it as one
        # that did kept the gate off the blocking path, which was the intent, but it
        # also rendered the healthy block into Verify's opening turn: the model was told
        # every pod was Running and Ready on the strength of an unparsed response, and
        # the audit row recorded a gate that ran and found a clean namespace.
        _log.warning("health gate: pod inventory unreadable for ns=%s: %s", ns, notes[0][:160])
        return HealthGate(checked=False, error=notes[0][:200], namespace=ns)

    _log.info("health gate: ns=%s swept, %d failure(s)", ns, len(failures))
    return HealthGate(
        checked=True,
        failures=tuple(failures),
        namespace=ns,
        observations=tuple(notes),
    )
