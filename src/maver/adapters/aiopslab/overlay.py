"""What each role is told, and allowed, when the incident is an AIOpsLab task.

The prototype carried five branches for this, one in each role that needed it, written
as ``if alert["benchmark"] == "aiopslab"`` beside the code that composed the prompt
(``agent/agents/rca.py:346-374``, ``planning.py:31-35``, ``execution.py:86-91``,
``verify.py:172-184``). They are the same five decisions here, expressed once, as a
table keyed on the role and the graded task, so a second environment adds a table of
its own rather than a sixth branch in the same files.

What the model reads is carried, not rewritten. Three of the assets are the prototype's
own files byte for byte; the other four are the strings it built in Python, moved into
``.txt`` because that is where this repository keeps prompt text. Two compositional
differences are unavoidable and are named where they occur: MAVER wraps a replaced system
prompt in the same header and tool-call protocol the prototype's ``agent_system_prompt``
wrapped it in, and a context block is prepended to the role's opening turn rather than
substituted into the middle of it.
"""

import json
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from maver.adapters.aiopslab.config import AiopsLabConfig
from maver.adapters.base import RoleOverlay, SubmitGuard
from maver.audit.journal import Journal
from maver.contracts.state import IncidentState
from maver.prompts.loader import read_asset, render
from maver.tools.context import ToolContext
from maver.tools.scope import IncidentScope
from maver.tools.snapshot import namespace_snapshot
from maver.tools.transport.base import ToolTransport
from maver.tools.workspace import SessionWorkspace

__all__ = ["BENCHMARK", "role_overlay", "task_of"]

BENCHMARK = "aiopslab"
"""The value of the alert's ``benchmark`` key that this adapter answers for."""

PROMPT_PACKAGE = "maver.adapters.aiopslab"
"""Read through the same loader the core's prompts use, so the containment check and
the strict two-way rendering apply to an adapter's assets exactly as they do to ours."""

_DETECTION = "detection"
_MITIGATION = "mitigation"
_CONTRACT_TASKS = frozenset({"localization", "analysis", _MITIGATION})
"""The tasks whose diagnosis is the one the role was written for, and so is qualified by
a suffix rather than replaced. AIOpsLab grades exactly four tasks and the driver refuses
any other, so this and detection are the whole vocabulary."""

_BOUNDARY_MARKER = "[AT/AFTER incident boundary]"
"""What :mod:`maver.tools.snapshot` prints beside a termination it dated against the
incident's observation window. The guard below reads that line and nothing else."""


def _asset(name: str) -> str:
    """Read one of this adapter's own prompt assets.

    Args:
        name: The file name under ``prompts/``.

    Returns:
        The asset's text, stripped, as the core's loader strips its own.
    """
    return read_asset(f"prompts/{name}", package=PROMPT_PACKAGE)


def task_of(alert: Mapping[str, Any]) -> str:
    """Name the AIOpsLab task this incident is graded on.

    Args:
        alert: The incident's alert payload.

    Returns:
        The task type, lowercased, or ``"unknown"`` when the driver named none.
    """
    return str(alert.get("benchmark_task") or "unknown").strip().lower()


def role_overlay(
    role: str,
    state: IncidentState,
    *,
    config: AiopsLabConfig,
    transport: ToolTransport | None,
    authority_mode: str,
) -> RoleOverlay | None:
    """Narrow one role to the AIOpsLab task this incident is graded on.

    Args:
        role: The role about to run.
        state: The incident as it stands, which carries the alert and the task.
        config: The deployment's adapter configuration.
        transport: The bridge the detection guard reads live state through, or None
            when this process supplied none.
        authority_mode: The deployment's approval mode, which decides which of the two
            approval sentences the remediating role is given.

    Returns:
        The overlay for this role and task, or None when there is nothing to narrow.
    """
    alert: Mapping[str, Any] = state.get("alert") or {}
    task = task_of(alert)

    if role == "rca":
        if task == _DETECTION:
            return _detection(alert, state, config=config, transport=transport)
        if task in _CONTRACT_TASKS:
            return RoleOverlay(
                system_suffix=render(_asset("rca_contract.txt"), {"TASK_TYPE": task})
            )
        return None
    if task != _MITIGATION:
        return None
    if role == "planning":
        return RoleOverlay(system_suffix=_asset("remediation.txt"))
    if role == "execution":
        return RoleOverlay(
            system_suffix=f"{_asset('remediation.txt')}\n\n{_scope(alert, authority_mode)}"
        )
    if role == "verify":
        return _verify(alert, config=config)
    return None


def _scope(alert: Mapping[str, Any], authority_mode: str) -> str:
    """Tell the remediating role which namespace it is confined to, and who authorises.

    The bridge enforces the namespace independently, but naming it here avoids spending
    turns on commands the bridge must reject. The approval sentence is chosen rather
    than written because an unattended run must not be told to wait for a human: human
    review remains an enforcement boundary, not a task for the model to block on.

    Args:
        alert: The incident's alert payload.
        authority_mode: The deployment's approval mode.

    Returns:
        The incident-scope block.
    """
    namespace = str(alert.get("namespace") or "").strip() or "the incident namespace"
    approval = "approval_auto.txt" if authority_mode == "auto" else "approval_gated.txt"
    return render(
        _asset("execution_scope.txt"),
        {"NAMESPACE": namespace, "APPROVAL": _asset(approval)},
    )


def _verify(alert: Mapping[str, Any], *, config: AiopsLabConfig) -> RoleOverlay | None:
    """Give the checking role the application's own entry point to probe.

    Static deployment facts, keyed only by namespace and carrying no task, fault or
    diagnosis information: verification needs a reliable user-facing Service to probe
    and must stay blind to the incident's causal answer.

    Args:
        alert: The incident's alert payload.
        config: The deployment's adapter configuration.

    Returns:
        The overlay carrying the contract block, or None for a namespace no application
        was configured for — an unknown namespace has no entry point to name, and
        inventing one would send the probe somewhere nothing serves.
    """
    contract = config.connectivity_contracts.get(str(alert.get("namespace") or ""))
    if contract is None:
        return None
    # Copied on the way out. The configuration's own mapping is process-wide and every
    # incident reads it, so handing out the stored object would let one incident's
    # rendering path mutate what the next one is told.
    block = render(
        _asset("verify_contract.txt"), {"CONTRACT_JSON": json.dumps(dict(contract), indent=2)}
    )
    # The prototype substituted this between the alert JSON and the task text; MAVER
    # prepends a context block to the role's opening turn. Same words, earlier position.
    return RoleOverlay(context_blocks=(block,))


def _detection(
    alert: Mapping[str, Any],
    state: IncidentState,
    *,
    config: AiopsLabConfig,
    transport: ToolTransport | None,
) -> RoleOverlay:
    """Replace the diagnosis with the bounded yes-or-no question detection actually asks.

    Deep endpoint enumeration and local general-purpose execution were the main sources
    of malformed and wandering calls in the AIOpsLab detection transcript, so they are
    withheld; localization and analysis keep the full diagnostic toolset.

    Args:
        alert: The incident's alert payload.
        state: The incident as it stands, for the directory this run already owns.
        config: The deployment's adapter configuration.
        transport: The bridge the guard reads live state through, or None.

    Returns:
        The detection overlay.
    """
    return RoleOverlay(
        # Replaces the composed role prompt only. MAVER still prepends the current time
        # and appends the tool-call protocol, which is exactly what the prototype's
        # `agent_system_prompt` did around this same text.
        system_replacement=_asset("detection.txt"),
        max_turns=config.detection_max_turns,
        drop_tools=config.detection_dropped_tools,
        submit_guard=_detection_consistency_guard(alert, state, transport),
    )


def _claims_no(args: Mapping[str, Any]) -> bool:
    """Whether a submitted ``RootCause`` says the episode held no anomaly."""
    summary = str(args.get("summary") or "").lower().replace(" ", "")
    explicit_none = (
        str(args.get("fault_type") or "").strip().lower() == "none"
        and str(args.get("affected_resource") or "").strip().lower() == "none"
    )
    return explicit_none or "has_anomaly=no" in summary


def _detection_consistency_guard(
    alert: Mapping[str, Any], state: IncidentState, transport: ToolTransport | None
) -> SubmitGuard:
    """Reject a No submission that contradicts post-boundary live evidence.

    This is a benchmark-semantics consistency check, not an answer oracle: it compares
    the trusted neutral observation boundary with Kubernetes container status and never
    receives a problem ID, injector arguments, or expected label. The model must still
    produce the corrected RootCause submission itself.

    Args:
        alert: The incident's alert payload, for the namespace and the boundary.
        state: The incident as it stands, for the directory this run already owns.
        transport: The bridge the snapshot is read through, or None.

    Returns:
        The guard, bound to this incident.
    """
    workdir = Path(state.get("incident_dir") or tempfile.gettempdir())
    session_id = str(state.get("session_id") or "")

    async def guard(args: dict[str, Any], _calls: int, _messages: list[Any]) -> str | None:
        # A "Yes" is never second-guessed, and the snapshot is not even read for one:
        # the guard exists to catch a claim the live evidence contradicts, and reading
        # the cluster to confirm an answer it would accept either way is a cluster read
        # and a minute of wall clock spent to change nothing.
        if not _claims_no(args):
            return None
        live = await _live_snapshot(alert, workdir, session_id, transport)
        decisive = next(
            (line.strip() for line in live.splitlines() if _BOUNDARY_MARKER in line), None
        )
        if decisive is None:
            return None
        return render(_asset("detection_guard_rejection.txt"), {"DECISIVE": decisive})

    return guard


async def _live_snapshot(
    alert: Mapping[str, Any], workdir: Path, session_id: str, transport: ToolTransport | None
) -> str:
    """Read the incident namespace as the diagnosing role's own snapshot renders it.

    Through the bound transport and the shipped renderer rather than a second reader,
    because the marker the guard keys on is produced by that renderer's comparison of a
    container's last termination against the alert's observation window: a second
    implementation would be a second definition of the boundary.

    Args:
        alert: The incident's alert payload.
        workdir: The directory this incident already owns.
        session_id: The incident this read belongs to.
        transport: The bridge to read through, or None.

    Returns:
        The rendered snapshot, or an empty string when there is nothing to read it
        with. Empty carries no decisive line, so the submission stands — a guard that
        refused because it could not look would reject on its own outage.
    """
    namespace = str(alert.get("namespace") or "").strip()
    if transport is None or not namespace:
        return ""
    context = ToolContext(
        transport=transport,
        # Required by the context and never written by this path: the snapshot composes
        # through `read_kubectl`, which returns the whole result rather than offloading
        # it. The incident's own directory is what it is pointed at so that the guard
        # creates nothing of its own to clean up.
        workspace=SessionWorkspace(workdir),
        role="rca",
        session_id=session_id,
        journal=Journal(session_id=session_id, stage="rca"),
        # The boundary line the guard reads exists only when the scope carries the
        # incident's observation window and names the namespace being snapshotted.
        scope=IncidentScope.from_alert(alert),
    )
    rendered: str = await namespace_snapshot(context, namespace)
    return rendered
