"""One deterministic, read-only view of a whole namespace.

The failure this exists for is fault *localization*: a diagnosing role names the wrong
component, often one it never looked at — a sibling Deployment, a Service binding, a
ConfigMap that pod-centric attention never reaches — and prompt text telling it to look
does not fix that. This gathers the namespace in code, computes the binding, reference
and recency anomalies deterministically, and renders one compact view in which the
faulty object is guaranteed to appear.

Four rules constrain every line of the rendering:

- Deterministic, no model involved. The differences (targetPort against containerPort, a
  dangling ConfigMap/Secret reference, empty endpoints, generation drift) are computed
  here.
- Generic Kubernetes correctness only. Every ``[!]`` is a universally-true health signal,
  never "this looks like fault X" — no benchmark or fault hard-coding whatsoever.
- Facts and neutral flags, never a verdict or a ranking. The role still localizes and
  reasons; this only guarantees the right objects are in front of it.
- Bounded output: one line per object, capped lists, transient identifiers stripped, so
  it stays small enough for a small model's context window.

Those rules have a measured cost that is not a defect to fix: a working snapshot is a
decoy generator, 8.6 to 15.0 ``[!]`` flags against exactly one injected fault, some of
them permanent quirks of a chart. Flagging fewer things would mean encoding which fault
was injected. It does mean snapshot-on is never neutral context when arms are compared,
which is why every arm file pins the pipeline flag that auto-injects it instead of
inheriting the default.
"""

import asyncio
import json
import logging
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from kira.tools.context import ToolContext
from kira.tools.errors import tool_error
from kira.tools.kubernetes import read_kubectl
from kira.tools.policy import read_only
from kira.tools.registry import tool

__all__ = ["namespace_snapshot"]

_log = logging.getLogger(__name__)

# Output bounds — keep the rendered snapshot small for small-model contexts.
_MAX_NOT_READY = 12
_MAX_SERVICES = 30
_MAX_EVENTS = 15
_MAX_REF_OK = 8
_MSG_LEN = 160

_READ_TIMEOUT_S = 90.0

_NAMESPACE = re.compile(r"^[a-z0-9][a-z0-9.-]{0,251}$")

# Container/pod states that mean "not healthy" even if phase is Running.
_BAD_WAITING = {
    "CrashLoopBackOff",
    "ImagePullBackOff",
    "ErrImagePull",
    "CreateContainerConfigError",
    "CreateContainerError",
    "RunContainerError",
    "InvalidImageName",
    "CONFIG",
}

_READS: dict[str, str] = {
    "workloads": "kubectl get deployments,statefulsets,daemonsets -n {ns} -o json",
    "pods": "kubectl get pods -n {ns} -o json",
    "services": "kubectl get services -n {ns} -o json",
    "endpoints": "kubectl get endpoints -n {ns} -o json",
    "configmaps": "kubectl get configmaps -n {ns} -o json",
    "secrets": "kubectl get secrets -n {ns} -o json",
    "events": "kubectl get events -n {ns} -o json",
    "nodes": "kubectl get nodes -o json",
}

_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "namespace": {
            "type": "string",
            "description": "Namespace to snapshot, e.g. 'shop'.",
        },
        "since_minutes": {
            "type": "integer",
            "description": "Lookback window for recent changes/events (default 30).",
        },
    },
    "required": ["namespace"],
}


def _load_items(text: str) -> list[Any] | None:
    """Parse one ``-o json`` read into a list of objects.

    Three states, and the difference between the last two is the whole point: ``None``
    means the read failed, ``[]`` means it succeeded and found nothing. A section
    rendered from ``None`` must say ``(unavailable)``. Before the distinction existed a
    failed read printed "(no ConfigMap/Secret references found)" and "(no Warning events
    in window)" — asserted false negatives, which are worse than a gap.
    """
    if not text or text.startswith("error:"):
        return None
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        items = data.get("items")
        if isinstance(items, list):
            return items
        if data.get("kind"):
            return [data]
    return []


def _now() -> datetime:
    return datetime.now(UTC)


def _parse_ts(ts: Any) -> datetime | None:
    if not ts or not isinstance(ts, str):
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def _fmt_age(ts: Any) -> str:
    dt = _parse_ts(ts)
    if dt is None:
        return "?"
    secs = (_now() - dt).total_seconds()
    secs = max(secs, 0)
    if secs < 90:
        return f"{int(secs)}s"
    if secs < 5400:
        return f"{int(secs // 60)}m"
    if secs < 172800:
        return f"{int(secs // 3600)}h"
    return f"{int(secs // 86400)}d"


def _within(ts: Any, since_minutes: int) -> bool:
    dt = _parse_ts(ts)
    if dt is None:
        return False
    return (_now() - dt).total_seconds() <= since_minutes * 60


def _short(text: object, limit: int = _MSG_LEN) -> str:
    s = " ".join(str(text or "").split())
    return s if len(s) <= limit else s[: limit - 1] + "…"


def _workload_key(pod: Mapping[str, Any]) -> tuple[str, str] | None:
    """Best-effort (kind, name) of the workload owning *pod*.

    ReplicaSet owners are collapsed to their Deployment by stripping the
    pod-template-hash suffix; StatefulSet/DaemonSet/Job owners are used directly.
    """
    owners = (pod.get("metadata") or {}).get("ownerReferences") or []
    for owner in owners:
        kind, name = owner.get("kind"), owner.get("name")
        if not name:
            continue
        if kind == "ReplicaSet":
            return ("Deployment", name.rsplit("-", 1)[0])
        if kind in ("StatefulSet", "DaemonSet", "Job"):
            return (kind, name)
    return None


def _relative_to_boundary(timestamp: str, boundary: str | None) -> str:
    observed = _parse_ts(timestamp)
    start = _parse_ts(boundary)
    if observed is None or start is None:
        return ""
    relation = "AT/AFTER" if observed >= start else "BEFORE"
    return f" [{relation} incident boundary]"


def _pod_summary(pod: Mapping[str, Any], boundary: str | None = None) -> dict[str, Any]:
    """Reduce one pod to the fields the rendering needs.

    A pod that is Ready right now can still carry the only evidence of a short-lived
    failure in ``lastState.terminated``, so current health and prior termination are
    kept apart rather than merged: the model compares ``finishedAt`` against the neutral
    incident boundary instead of treating every old restart as live.
    """
    meta = pod.get("metadata") or {}
    status = pod.get("status") or {}
    spec = pod.get("spec") or {}
    cstats = status.get("containerStatuses") or []
    restarts = sum(int(c.get("restartCount") or 0) for c in cstats)
    ready = bool(cstats) and all(c.get("ready") for c in cstats)
    current_reason = ""
    last_termination = ""
    last_termination_at = ""
    for c in cstats:
        waiting = (c.get("state") or {}).get("waiting") or {}
        terminated = (c.get("state") or {}).get("terminated") or {}
        last_term = (c.get("lastState") or {}).get("terminated") or {}
        if not current_reason and waiting.get("reason"):
            current_reason = _short(f"{waiting['reason']}: {waiting.get('message', '')}", 120)
        if (
            not current_reason
            and terminated.get("reason")
            and terminated.get("reason") != "Completed"
        ):
            current_reason = _short(
                f"{terminated['reason']}(exit {terminated.get('exitCode', '?')})", 80
            )
        if not last_termination and last_term.get("reason"):
            last_termination_at = str(last_term.get("finishedAt") or "")
            timestamp = (
                f" finishedAt={last_termination_at}"
                f"{_relative_to_boundary(last_termination_at, boundary)}"
                if last_termination_at
                else ""
            )
            last_termination = _short(
                f"last {last_term['reason']}(exit {last_term.get('exitCode', '?')}){timestamp}",
                120,
            )
    phase = status.get("phase", "?")
    healthy = ready and phase in ("Running", "Succeeded") and not current_reason
    return {
        "name": meta.get("name", "?"),
        "phase": phase,
        "ready": ready,
        "restarts": restarts,
        "node": spec.get("nodeName", "?"),
        "reason": current_reason or last_termination,
        "current_reason": current_reason,
        "last_termination": last_termination,
        "last_termination_at": last_termination_at,
        "labels": meta.get("labels") or {},
        "start": status.get("startTime") or meta.get("creationTimestamp"),
        "healthy": healthy,
    }


def _container_ports(pods: list[Any]) -> tuple[set[int], set[str]]:
    nums: set[int] = set()
    names: set[str] = set()
    for pod in pods:
        spec = pod.get("spec") or {}
        for container in (spec.get("containers") or []) + (spec.get("initContainers") or []):
            for port in container.get("ports") or []:
                declared = port.get("containerPort")
                if isinstance(declared, int):
                    nums.add(declared)
                if port.get("name"):
                    names.add(port["name"])
    return nums, names


def _selector_matches(labels: Mapping[str, Any], selector: Mapping[str, Any]) -> bool:
    return bool(selector) and all(labels.get(k) == v for k, v in selector.items())


def _collect_refs(workloads: list[Any]) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Map each referenced ConfigMap/Secret name to the workloads referencing it."""
    cm_refs: dict[str, list[str]] = {}
    sec_refs: dict[str, list[str]] = {}

    def add(refs: dict[str, list[str]], name: str | None, who: str) -> None:
        if name:
            refs.setdefault(name, [])
            if who not in refs[name]:
                refs[name].append(who)

    for workload in workloads:
        who = f"{workload.get('kind', '?')}/{(workload.get('metadata') or {}).get('name', '?')}"
        spec = ((workload.get("spec") or {}).get("template") or {}).get("spec") or {}
        for secret in spec.get("imagePullSecrets") or []:
            add(sec_refs, secret.get("name"), who)
        for container in (spec.get("containers") or []) + (spec.get("initContainers") or []):
            for env_from in container.get("envFrom") or []:
                add(cm_refs, (env_from.get("configMapRef") or {}).get("name"), who)
                add(sec_refs, (env_from.get("secretRef") or {}).get("name"), who)
            for env in container.get("env") or []:
                value_from = env.get("valueFrom") or {}
                add(cm_refs, (value_from.get("configMapKeyRef") or {}).get("name"), who)
                add(sec_refs, (value_from.get("secretKeyRef") or {}).get("name"), who)
        for volume in spec.get("volumes") or []:
            add(cm_refs, (volume.get("configMap") or {}).get("name"), who)
            add(sec_refs, (volume.get("secret") or {}).get("secretName"), who)
            for source in (volume.get("projected") or {}).get("sources") or []:
                add(cm_refs, (source.get("configMap") or {}).get("name"), who)
                add(sec_refs, (source.get("secret") or {}).get("name"), who)
    return cm_refs, sec_refs


def _render(
    ns: str,
    since_minutes: int,
    raw: Mapping[str, str],
    boundary: str | None = None,
) -> str:
    """Render the six sections from the eight reads, flagging generic health problems."""
    flags: list[str] = []  # collected anomaly count
    out: list[str] = []

    parsed = {key: _load_items(raw.get(key, "")) for key in _READS}

    # A section rendered "(unavailable)" says nothing about *why*, which is how a total
    # read outage stayed invisible for months. Name the failing read and show what came
    # back instead.
    for key, items in parsed.items():
        if items is None:
            text = raw.get(key) or ""
            _log.warning(
                "[snapshot] ns=%s read=%s unusable (%d chars): %s",
                ns,
                key,
                len(text),
                " ".join(text[:200].split()) or "<empty>",
            )

    workloads = parsed["workloads"]
    pods_raw = parsed["pods"]
    pods = pods_raw or []
    services = parsed["services"]
    endpoints = parsed["endpoints"]
    configmaps = parsed["configmaps"]
    secrets = parsed["secrets"]
    events_raw = parsed["events"]
    events = events_raw or []
    nodes = parsed["nodes"] or []

    pod_sums = [_pod_summary(p, boundary) for p in pods]
    by_workload: dict[str, list[dict[str, Any]]] = {}
    for pod, summary in zip(pods, pod_sums, strict=True):
        owner = _workload_key(pod)
        if owner:
            by_workload.setdefault(owner[1], []).append(summary)

    # --- WORKLOADS ---------------------------------------------------------- #
    out.append("WORKLOADS  (kind/name · ready/desired · restarts · age · image)")
    if workloads is None:
        out.append("  (unavailable)")
    elif not workloads:
        out.append("  (none)")
    else:
        if pods_raw is None:
            # Restart counts and the not-ready list below both come from the pod read;
            # without it a workload looks restart-free rather than unknown.
            out.append("  (pod detail unavailable — restart counts and readiness omitted)")
        for workload in workloads:
            meta = workload.get("metadata") or {}
            spec = workload.get("spec") or {}
            status = workload.get("status") or {}
            kind = workload.get("kind", "?")
            name = meta.get("name", "?")
            desired = spec.get("replicas")
            if desired is None:  # DaemonSet
                desired = status.get("desiredNumberScheduled")
            ready = status.get("readyReplicas")
            if ready is None:
                ready = status.get("numberReady", 0)
            ready = ready or 0
            desired = 0 if desired is None else desired
            template = (spec.get("template") or {}).get("spec") or {}
            images = [c.get("image", "?") for c in template.get("containers") or []]
            image = images[0].split("@")[0] if images else "?"
            restarts = sum(s["restarts"] for s in by_workload.get(name, []))
            generation = meta.get("generation")
            observed = status.get("observedGeneration")
            mark = ""
            if desired and ready < desired:
                mark = f"  [!] {ready}/{desired} ready"
                flags.append(name)
            if generation is not None and observed is not None and observed < generation:
                mark += "  [!] rollout in progress"
                if name not in flags:
                    flags.append(name)
            out.append(
                f"  {kind}/{name}  {ready}/{desired}  rst={restarts}  "
                f"{_fmt_age(meta.get('creationTimestamp'))}  {image}{mark}"
            )

    # --- POD STATE / RESTART HISTORY ---------------------------------------- #
    pod_issues = [s for s in pod_sums if not s["healthy"] or s["last_termination"]]
    if pods_raw is None:
        # Announced here rather than only inside the workloads branch: when both reads
        # failed, the prototype mentioned the pod outage nowhere at all.
        out.append("")
        out.append("POD STATE / RESTART HISTORY  (current · restarts · node · reason and time)")
        out.append("  (unavailable)")
    elif pod_issues:
        out.append("")
        out.append("POD STATE / RESTART HISTORY  (current · restarts · node · reason and time)")
        for summary in pod_issues[:_MAX_NOT_READY]:
            current = (
                "Ready" if summary["healthy"] else f"{summary['phase']}/ready={summary['ready']}"
            )
            details = [x for x in (summary["current_reason"], summary["last_termination"]) if x]
            reason = " · ".join(details) or "no container reason reported"
            tag = "[!] " if (summary["restarts"] or any(b in reason for b in _BAD_WAITING)) else ""
            out.append(
                f"  {tag}{summary['name']}  current={current}  rst={summary['restarts']}  "
                f"node={summary['node']}  {reason}"
            )
            if tag:
                # Pods listed only for a historical termination carry no [!] of their
                # own, so counting them made the header claim more flagged components
                # than the body shows.
                flags.append(summary["name"])
        if len(pod_issues) > _MAX_NOT_READY:
            out.append(f"  … {len(pod_issues) - _MAX_NOT_READY} more pod issues")

    # --- SERVICES / BINDINGS ------------------------------------------------ #
    out.append("")
    out.append("SERVICES / BINDINGS  (selector → endpoints · port→targetPort)")
    if services is None:
        out.append("  (unavailable)")
    elif not services:
        out.append("  (none)")
    else:
        ep_count: dict[str, int] = {}
        for endpoint in endpoints or []:
            name = (endpoint.get("metadata") or {}).get("name", "")
            addresses = 0
            for subset in endpoint.get("subsets") or []:
                addresses += len(subset.get("addresses") or [])
            ep_count[name] = addresses
        for svc in services[:_MAX_SERVICES]:
            meta = svc.get("metadata") or {}
            spec = svc.get("spec") or {}
            name = meta.get("name", "?")
            selector = spec.get("selector") or {}
            stype = spec.get("type", "ClusterIP")
            if stype in ("ExternalName",):
                out.append(f"  svc/{name}  type={stype}  {spec.get('externalName', '')}")
                continue
            matched = [
                p
                for p in pods
                if _selector_matches((p.get("metadata") or {}).get("labels") or {}, selector)
            ]
            nums, names = _container_ports(matched)
            # A failed endpoints read is not an endpoints count of zero. Collapsing
            # the two makes every selector-bearing Service look broken, which is a
            # fabricated finding in the one section whose purpose is binding
            # correctness — and it is manufactured from a gap, not observed.
            eps = ep_count.get(name, 0) if endpoints is not None else None
            port_strs: list[str] = []
            port_flags: list[str] = []
            for port in spec.get("ports") or []:
                declared = port.get("port")
                target = port.get("targetPort", declared)
                port_strs.append(f"{declared}→{target}")
                if matched:  # only judge when we can see the backing pods
                    ok = (target in nums) if isinstance(target, int) else (target in names)
                    if not ok:
                        port_flags.append(
                            f"targetPort {target} matches no containerPort "
                            f"{sorted(nums) or sorted(names)}"
                        )
            sel = ",".join(f"{k}={v}" for k, v in selector.items()) or "(none)"
            marks = []
            if selector and eps == 0:
                marks.append("0 endpoints")
            marks.extend(port_flags)
            mark = ("  [!] " + "; ".join(marks)) if marks else "  [OK]"
            if marks:
                flags.append(name)
            out.append(
                f"  svc/{name}  {sel}  eps:{'?' if eps is None else eps}  "
                f"{' '.join(port_strs) or '(no ports)'}{mark}"
            )
        if len(services) > _MAX_SERVICES:
            out.append(f"  … {len(services) - _MAX_SERVICES} more services")

    # --- CONFIG REFERENCES -------------------------------------------------- #
    out.append("")
    out.append("CONFIG REFERENCES  (ConfigMap/Secret ← referencing workloads)")
    cm_refs, sec_refs = _collect_refs(workloads or [])
    cm_names = (
        {(c.get("metadata") or {}).get("name") for c in configmaps}
        if configmaps is not None
        else None
    )
    sec_names = (
        {(s.get("metadata") or {}).get("name") for s in secrets} if secrets is not None else None
    )
    ok_lines: list[str] = []
    bad_lines: list[str] = []
    for refs, existing, kindlabel in ((cm_refs, cm_names, "cm"), (sec_refs, sec_names, "secret")):
        for name, who in sorted(refs.items()):
            users = ", ".join(who[:4]) + (" …" if len(who) > 4 else "")
            if existing is not None and name not in existing:
                bad_lines.append(f"  [!] {kindlabel}/{name}  NOT FOUND  ← {users}")
                flags.append(name)
            else:
                ok_lines.append(f"  {kindlabel}/{name}  ← {users}")
    out.extend(bad_lines)
    out.extend(ok_lines[:_MAX_REF_OK])
    if len(ok_lines) > _MAX_REF_OK:
        out.append(f"  … {len(ok_lines) - _MAX_REF_OK} more resolved references")
    if not bad_lines and not ok_lines:
        # References are derived from the workload specs, so a failed workload read
        # yields the same empty result as a namespace that genuinely references nothing.
        # Saying "none found" there is a false negative, not an absence.
        out.append(
            "  (unavailable)" if workloads is None else "  (no ConfigMap/Secret references found)"
        )

    # --- RECENT CHANGES ----------------------------------------------------- #
    out.append("")
    out.append(f"RECENT CHANGES  (Warning events, last {since_minutes}m)")
    warn: list[tuple[Any, str, Any, str]] = []
    for event in events:
        if event.get("type") != "Warning":
            continue
        ts = (
            event.get("lastTimestamp")
            or event.get("eventTime")
            or (event.get("metadata") or {}).get("creationTimestamp")
        )
        if not _within(ts, since_minutes):
            continue
        involved = event.get("involvedObject") or {}
        warn.append(
            (
                event.get("count", 1),
                f"{involved.get('kind', '?')}/{involved.get('name', '?')}",
                event.get("reason", "?"),
                _short(event.get("message", "")),
            )
        )
    if warn:
        for count, obj, reason, msg in warn[:_MAX_EVENTS]:
            # The multiplication sign is the event count separator the model has been
            # read this rendering with since the tool shipped; it is not a stray glyph.
            out.append(f"  [!] {reason} ×{count}  {obj}  {msg}")  # noqa: RUF001
        if len(warn) > _MAX_EVENTS:
            out.append(f"  … {len(warn) - _MAX_EVENTS} more warnings")
    elif events_raw is None:
        out.append("  (unavailable)")
    else:
        out.append("  (no Warning events in window)")
    recent_pods = [s["name"] for s in pod_sums if _within(s["start"], since_minutes)]
    if recent_pods:
        out.append(
            f"  recently (re)started pods: {', '.join(recent_pods[:8])}"
            + (" …" if len(recent_pods) > 8 else "")
        )
    recent_terminations = [
        f"{s['name']}@{s['last_termination_at']}"
        for s in pod_sums
        if _within(s["last_termination_at"], since_minutes)
    ]
    if recent_terminations:
        out.append(
            f"  recent container terminations: {', '.join(recent_terminations[:8])}"
            + (" …" if len(recent_terminations) > 8 else "")
        )

    # --- NODES -------------------------------------------------------------- #
    out.append("")
    out.append("NODES  (cluster-scoped context)")
    if not nodes:
        out.append("  (unavailable)")
    else:
        pressure_seen = False
        for node in nodes:
            meta = node.get("metadata") or {}
            conds = {
                c.get("type"): c.get("status")
                for c in (node.get("status") or {}).get("conditions") or []
            }
            ready = conds.get("Ready")
            pressures = [
                t
                for t in ("MemoryPressure", "DiskPressure", "PIDPressure", "NetworkUnavailable")
                if conds.get(t) == "True"
            ]
            mark = ""
            if ready != "True" or pressures:
                mark = "  [!] " + ", ".join(
                    ([f"Ready={ready}"] if ready != "True" else []) + pressures
                )
                pressure_seen = True
                flags.append(meta.get("name", "node"))
            out.append(f"  {meta.get('name', '?')}  Ready={ready}{mark}")
        if not pressure_seen:
            out.append("  (no MemoryPressure / DiskPressure / NotReady)")

    boundary_line = (
        f"Incident observation boundary: {boundary}; termination "
        "timing labels are deterministic timestamp comparisons.\n"
        if boundary
        else ""
    )
    header = (
        f"NAMESPACE SNAPSHOT: {ns}   (window: last {since_minutes}m; "
        f"{len(set(flags))} component(s) flagged [!])\n"
        + boundary_line
        + "Deterministic read-only facts — generic Kubernetes health flags only, NOT a "
        "root-cause verdict. The faulty component is among these; localize it yourself.\n"
    )
    return header + "\n".join(out)


@tool(
    name="namespace_snapshot",
    safety="read_only",
    requires={"kubernetes"},
    parameters=_PARAMETERS,
)
async def namespace_snapshot(ctx: ToolContext, namespace: str, since_minutes: int = 30) -> str:
    """Snapshot one namespace: workloads, pods, bindings, references, events and nodes.

    Args:
        ctx: The run's context.
        namespace: The namespace to read.
        since_minutes: Lookback window for recent events and restarts, 1 to 1440.

    Returns:
        The rendered snapshot, or a refusal beginning with ``error:``.
    """
    ns = (namespace or "").strip()
    if not _NAMESPACE.match(ns):
        return tool_error(f"invalid namespace '{namespace}'")
    try:
        since = int(since_minutes)
    except (TypeError, ValueError):
        since = 30
    since = max(1, min(since, 1440))

    cmds = {key: template.format(ns=ns) for key, template in _READS.items()}
    keys = list(cmds)

    async def bounded(cmd: str) -> str:
        # The transport's own per-call timeout cannot be relied on to fire: SSE
        # keep-alives held an MCP stream open while kubectl hung server-side, and an API
        # server under load can hang a read whatever the transport. This one is
        # unconditional.
        try:
            return await asyncio.wait_for(read_kubectl(ctx, cmd), timeout=_READ_TIMEOUT_S)
        except TimeoutError:
            return tool_error(f"timed out after {_READ_TIMEOUT_S}s")

    results = await asyncio.gather(*(bounded(cmds[key]) for key in keys))
    raw = dict(zip(keys, results, strict=True))

    # A discovery endpoint that intermittently 404s while an application is still rolling
    # out ("couldn't get current server API group list") is exactly what a session-start
    # snapshot runs into. Retry the unusable reads once, sequentially, so a retry cannot
    # collide with the remaining fan-out.
    for key in keys:
        if _load_items(raw[key]) is None:
            raw[key] = await bounded(cmds[key])

    # The incident observation boundary, and with it the BEFORE / AT-AFTER labels on
    # termination times, arrives once the incident scope is a field on the context.
    return _render(ns, since, raw)


read_only("namespace_snapshot")
