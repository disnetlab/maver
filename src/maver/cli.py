"""Command line entry point.

Subcommands are added alongside the features they drive: ``serve`` with the HTTP
API, ``approvals`` with the human approval workflow, ``config`` with the settings
model. Keeping them together means a feature and the way an operator reaches it land
in the same commit.

The approval commands read the configured store directly, which is what works from
inside the pod, and ``--url`` switches them to the API for a remote operator. Both paths
render a request through the same projection, so the two cannot describe one request
differently.
"""

import base64
import json
import logging
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any, cast

import httpx
import typer
from pydantic import ValidationError

from maver import __version__
from maver.adapters.registry import available_adapters
from maver.api.app import enabled_adapters, validate_deployment
from maver.api.feedback import FeedbackRequest
from maver.api.hitl import request_payload
from maver.api.incidents import IncidentRequest
from maver.audit.export import export_incident
from maver.configuration import Configuration, load_configuration
from maver.memory.corpus import MalformedRecord, import_records, parse_records
from maver.memory.factory import build_storage
from maver.memory.ports import ApprovalStatus, ApprovalStore
from maver.roles.registry import RoleRegistryError
from maver.settings.root import Settings, arm_overrides, legacy_names_in_use, load_settings

app = typer.Typer(
    name="maver",
    help="Kubernetes Incident Recovery Agent.",
    no_args_is_help=True,
    add_completion=False,
)
config_app = typer.Typer(
    help="Inspect and validate the resolved configuration.", no_args_is_help=True
)
approvals_app = typer.Typer(
    help="Review and decide the writes waiting on a person.", no_args_is_help=True
)
memory_app = typer.Typer(help="Load and inspect the experience corpus.", no_args_is_help=True)
incidents_app = typer.Typer(help="Submit incidents to a running deployment.", no_args_is_help=True)
audit_app = typer.Typer(
    help="Export complete private audit evidence offline.", no_args_is_help=True
)
queue_app = typer.Typer(help="Inspect and retry quarantined queue records.", no_args_is_help=True)
app.add_typer(audit_app, name="audit")
app.add_typer(config_app, name="config")
app.add_typer(approvals_app, name="approvals")
app.add_typer(memory_app, name="memory")
app.add_typer(incidents_app, name="incidents")
app.add_typer(queue_app, name="queue")


@queue_app.command("list")
def queue_list() -> None:
    """List records the local deployment quarantined."""
    storage = build_storage(_resolved())
    try:
        records = storage.sessions.list_quarantine()
    finally:
        storage.close()
    typer.echo(
        json.dumps(
            [
                {
                    "record_id": r.record_id,
                    "source": r.source,
                    "key": r.key,
                    "reason": r.reason,
                    "created_at": r.created_at,
                    "retry_session_id": r.retry_session_id,
                }
                for r in records
            ],
            indent=2,
        )
    )


@queue_app.command("show")
def queue_show(record_id: str) -> None:
    """Show a local quarantine record and its original message bytes as text.

    Args:
        record_id: The record to inspect.
    """
    storage = build_storage(_resolved())
    try:
        record = storage.sessions.get_quarantine(record_id)
    finally:
        storage.close()
    if record is None:
        typer.echo("quarantine record not found", err=True)
        raise typer.Exit(code=1)
    result = asdict(record)
    result["raw_text"] = base64.b64decode(record.raw_base64).decode("utf-8", errors="replace")
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


@queue_app.command("retry")
def queue_retry(
    record_id: str,
    corrected: Annotated[
        Path | None, typer.Option("--corrected", help="Corrected incident JSON.")
    ] = None,
    url: Annotated[
        str, typer.Option("--url", help="The running MAVER deployment.")
    ] = "http://127.0.0.1:8080",
) -> None:
    """Explicitly retry a quarantined delivery through the running server.

    Args:
        record_id: The local quarantine record id.
        corrected: Replacement incident request, needed for malformed JSON.
        url: The running deployment with access to the same store.
    """
    body = _json_object(corrected) if corrected else None
    try:
        receipt = _api(url, "POST", f"/quarantine/{record_id}/retry", json=body)
    except httpx.HTTPStatusError as exc:
        typer.echo(_detail(exc.response), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(receipt, indent=2, sort_keys=True))


UrlOption = Annotated[
    str | None,
    typer.Option(
        "--url",
        help=(
            "Act through a running server's API instead of the local store, so a "
            "decision continues its incident immediately."
        ),
    ),
]
JsonOption = Annotated[bool, typer.Option("--json", help="Print the raw record.")]
ReasonOption = Annotated[str, typer.Option("--reason", help="Why, recorded on the request.")]
ByOption = Annotated[
    str, typer.Option("--by", help="Who is deciding. A claimed name, not a verified one.")
]

_STATUSES = ("pending", "approved", "denied", "answered", "timed_out", "cancelled")


def _print_version(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit


@app.callback()
def main(
    *,
    show_version: bool = typer.Option(
        False,
        "--version",
        callback=_print_version,
        is_eager=True,
        help="Print the version and exit.",
    ),
) -> None:
    """Kubernetes Incident Recovery Agent.

    The callback exists so that the application stays a command group. Without it,
    Typer collapses an application holding a single command into that command, and
    every subcommand added later silently changes the CLI's shape.
    """


@app.command()
def version() -> None:
    """Print the installed version."""
    typer.echo(__version__)


@app.command()
def serve(
    host: Annotated[str | None, typer.Option("--host", help="Bind address.")] = None,
    port: Annotated[int | None, typer.Option("--port", help="Bind port.")] = None,
    reload: Annotated[bool, typer.Option("--reload", help="Restart on a source change.")] = False,
) -> None:
    """Run the HTTP API.

    This rather than a bare ``uvicorn`` line, because the configuration is resolved and
    validated here, before the port is bound. A process that starts on a configuration
    it cannot honour produces runs nobody can reconstruct.

    Args:
        host: Overrides the configured bind address.
        port: Overrides the configured bind port.
        reload: Restart on a source change, for local development.
    """
    import uvicorn  # noqa: PLC0415 - a server dependency no other command needs

    settings = _resolved()
    logging.basicConfig(level=settings.observability.log_level)
    bind_host = host or settings.server.host
    bind_port = port or settings.server.port
    typer.echo(f"maver {__version__} listening on {bind_host}:{bind_port}")
    uvicorn.run(
        "maver.api.app:create_app",
        factory=True,
        host=bind_host,
        port=bind_port,
        reload=reload,
    )


@audit_app.command("export")
def audit_export(
    session: Annotated[str, typer.Argument(help="Incident identifier to export.")],
    output: Annotated[
        Path, typer.Option("--output", help="Absent or empty private bundle directory.")
    ],
    audit_dir: Annotated[
        Path | None, typer.Option("--audit-dir", help="Audit root; defaults to WORKDIR/audit.")
    ] = None,
    bridge_log: Annotated[
        Path | None,
        typer.Option("--bridge-log", help="Optional host bridge log to join by invocation."),
    ] = None,
) -> None:
    """Export an incident without a running server, database, model or cluster."""
    try:
        root = audit_dir if audit_dir is not None else load_settings().server.workdir / "audit"
        manifest = export_incident(root, session, output, bridge_log=bridge_log)
    except (OSError, ValueError) as exc:
        typer.echo(f"could not export audit: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        json.dumps(
            {
                "output": str(output.resolve()),
                "session_id": session,
                "event_count": manifest["event_count"],
                "evidence_integrity": manifest["evidence_integrity"],
                "capture_coverage": manifest["capture_coverage"],
                "database_coverage": manifest["database_projection"]["coverage"],
                "started_without_terminal": manifest["started_without_terminal"],
                "database_pending_events": len(manifest["database_pending_event_ids"]),
            },
            indent=2,
        )
    )


@incidents_app.command("start")
def incidents_start(
    alert: Annotated[Path, typer.Argument(help="JSON file containing the alert.")],
    pipeline: Annotated[
        str | None,
        typer.Option("--pipeline", help="Configured pipeline; overrides selection middleware."),
    ] = None,
    metadata: Annotated[
        Path | None,
        typer.Option(
            "--metadata", help="JSON file of private run metadata, separate from the alert."
        ),
    ] = None,
    target_stage: Annotated[
        str, typer.Option("--target-stage", help="How far to run: full, diagnose, or rca.")
    ] = "full",
    url: Annotated[
        str, typer.Option("--url", help="The running MAVER deployment.")
    ] = "http://127.0.0.1:8080",
) -> None:
    """Submit an incident and print its acceptance receipt.

    Args:
        alert: File containing the model-visible incident input.
        pipeline: An explicit configured workflow, or the deployment's selector/default.
        metadata: Private run context, never merged into the alert.
        target_stage: How far the workflow should run.
        url: The running deployment to submit to.
    """
    body: dict[str, Any] = {"alert": _json_object(alert), "target_stage": target_stage}
    if pipeline is not None:
        body["pipeline"] = pipeline
    if metadata is not None:
        body["metadata"] = _json_object(metadata)
    try:
        IncidentRequest.model_validate(body)
    except ValidationError as exc:
        for error in exc.errors():
            typer.echo(f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}", err=True)
        raise typer.Exit(code=2) from exc
    try:
        receipt = _api(url, "POST", "/incidents", json=body)
    except httpx.HTTPStatusError as exc:
        typer.echo(_detail(exc.response), err=True)
        raise typer.Exit(code=1) from exc
    except httpx.RequestError as exc:
        typer.echo(f"could not reach MAVER: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(receipt, indent=2, sort_keys=True))


@incidents_app.command("reconcile-show")
def incidents_reconcile_show(session_id: str) -> None:
    """Inspect private checkpoint and last tool evidence from the configured local store."""
    storage = build_storage(_resolved())
    try:
        record = storage.sessions.get_session(session_id)
        if record is None:
            typer.echo("incident not found", err=True)
            raise typer.Exit(code=1)
        if record.run_status != "needs_reconciliation":
            typer.echo(f"incident is {record.run_status}", err=True)
            raise typer.Exit(code=1)
        events = [
            event
            for run in storage.runs.runs_for(session_id)
            for event in storage.runs.events_for(run.agent_run_id)
            if event.event_type in {"tool_call_started", "tool_call"}
        ]
        last = events[-1] if events else None
        result = next(
            (event for event in reversed(events) if event.event_type == "tool_call"), None
        )
        typer.echo(
            json.dumps(
                {
                    "session_id": session_id,
                    "interrupted_stage": record.current_stage,
                    "checkpoint": record.checkpoint,
                    "root_cause": record.root_cause,
                    "plan": record.plan,
                    "execution_result": record.execution_result,
                    "verify_result": record.verify_result,
                    "last_tool_action": None if last is None else last.payload,
                    "last_recorded_tool_result": None if result is None else result.payload,
                    "evidence_limit": "The external write may have completed without a recorded result. Inspect the live system.",
                },
                indent=2,
                sort_keys=True,
                default=str,
            )
        )
    finally:
        storage.close()


@incidents_app.command("reconcile-decide")
def incidents_reconcile_decide(
    session_id: str,
    action: Annotated[str, typer.Option("--action", help="close or replan")],
    observed_state: Annotated[str, typer.Option("--observed-state")],
    reason: Annotated[str, typer.Option("--reason")],
    decided_by: Annotated[str, typer.Option("--by")],
    url: Annotated[str, typer.Option("--url")] = "http://127.0.0.1:8080",
) -> None:
    """Record the live observation and decide through the running server."""
    if action not in {"close", "replan"}:
        raise typer.BadParameter("action must be close or replan")
    try:
        result = _api(
            url,
            "POST",
            f"/incidents/{session_id}/reconciliation",
            json={
                "action": action,
                "observed_state": observed_state,
                "reason": reason,
                "decided_by": decided_by,
            },
        )
    except httpx.HTTPStatusError as exc:
        typer.echo(_detail(exc.response), err=True)
        raise typer.Exit(code=1) from exc
    except httpx.RequestError as exc:
        typer.echo(f"could not reach MAVER: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


@incidents_app.command("feedback")
def incidents_feedback(
    session_id: str,
    verdict: Annotated[str, typer.Argument(help="confirmed, rejected, or corrected")],
    reason: Annotated[str, typer.Option("--reason", help="Why this verdict was reached.")],
    evidence: Annotated[str, typer.Option("--evidence", help="Observation supporting it.")],
    by: Annotated[str, typer.Option("--by", help="Operator identity.")],
    corrected_cause: Annotated[
        str, typer.Option("--corrected-cause", help="Correct cause for corrected verdicts.")
    ] = "",
    url: Annotated[
        str, typer.Option("--url", help="The running MAVER deployment.")
    ] = "http://127.0.0.1:8080",
) -> None:
    """Review a completed incident's outcome, preserving its original artifacts."""
    try:
        body = FeedbackRequest.model_validate(
            {
                "verdict": verdict,
                "reason": reason,
                "evidence": evidence,
                "decided_by": by,
                "corrected_cause": corrected_cause,
            }
        ).model_dump()
    except ValidationError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc
    try:
        receipt = _api(url, "POST", f"/incidents/{session_id}/feedback", json=body)
    except httpx.HTTPStatusError as exc:
        typer.echo(_detail(exc.response), err=True)
        raise typer.Exit(code=1) from exc
    except httpx.RequestError as exc:
        typer.echo(f"could not reach MAVER: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(receipt, indent=2, sort_keys=True))


def _json_object(path: Path) -> dict[str, Any]:
    """Read one object without echoing private content in a parsing error."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        typer.echo(f"{path}: could not read a JSON object", err=True)
        raise typer.Exit(code=2) from exc
    if not isinstance(data, dict):
        typer.echo(f"{path}: expected a JSON object", err=True)
        raise typer.Exit(code=2)
    return data


@config_app.command("show")
def config_show(as_json: JsonOption = False) -> None:
    """Print the resolved configuration, with secrets masked.

    Args:
        as_json: Print the snapshot record rather than one line per setting.
    """
    settings = _resolved()
    snapshot = settings.snapshot()
    if settings.customisation is not None:
        snapshot["customisation"] = _configuration(settings).manifest()
    if as_json:
        typer.echo(json.dumps(snapshot, indent=2, sort_keys=True))
        return
    for name, value in sorted(_flatten(snapshot["settings"]).items()):
        typer.echo(f"{name} = {value}")
    if "customisation" in snapshot:
        typer.echo("configured agents and pipelines:")
        typer.echo(json.dumps(snapshot["customisation"], indent=2, sort_keys=True))
    if snapshot["arm_digest"]:
        typer.echo(f"arm digest: {snapshot['arm_digest']}")
    for legacy, modern in sorted(legacy_names_in_use().items()):
        typer.echo(f"legacy name in use: {legacy} -> {modern}")


@config_app.command("check")
def config_check() -> None:
    """Validate the configuration the server would start on.

    Exits non-zero with one line per problem, so a misconfigured deployment is caught by
    an operator running a command rather than by an incident arriving.

    A setting the environment takes back from the named arm is printed rather than
    refused: overriding one key without editing a frozen file is the deployment
    contract working. Refusing is reserved for a path the arm claims to hold constant,
    which ``load_settings`` does, because there the override makes the arm's own claim
    false and the comparison it licenses stops being single-variable.
    """
    try:
        settings = load_settings()
        validate_deployment(settings)
        overridden = arm_overrides(settings).allowed
    except ValidationError as exc:
        for error in exc.errors():
            typer.echo(f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}", err=True)
        raise typer.Exit(code=2) from exc
    except (FileNotFoundError, RoleRegistryError, RuntimeError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc
    for path, (declared, actual) in sorted(overridden.items()):
        typer.echo(f"arm override: {path} = {actual} (arm {settings.arm} resolves {declared})")
    typer.echo("ok")


@config_app.command("pipelines")
def config_pipelines(as_json: JsonOption = False) -> None:
    """List configured pipeline names and the deployment default.

    Args:
        as_json: Print the complete resolved, redacted composition manifest.
    """
    configuration = _configuration(_resolved())
    if as_json:
        typer.echo(json.dumps(configuration.manifest(), indent=2, sort_keys=True))
        return
    for name, pipeline in sorted(configuration.pipelines.pipelines.items()):
        default = " (default)" if name == configuration.pipelines.default else ""
        typer.echo(f"{name}{default}: {pipeline.workflow.name}")


@config_app.command("adapters")
def config_adapters() -> None:
    """List the environment adapters this install has, and which ones are enabled.

    An adapter installed but unloadable is the failure this exists for: without it the
    service starts on the default transport and the run grades nothing, with the import
    error visible nowhere.
    """
    settings = _resolved()
    enabled = enabled_adapters(settings)
    for name, status in sorted(available_adapters().items()):
        mark = "enabled" if name in enabled else "available"
        typer.echo(f"{name}: {status}" if status != "ready" else f"{name}: {mark}")
    for name in sorted(enabled - set(available_adapters())):
        typer.echo(f"{name}: ENABLED BUT NOT INSTALLED", err=True)


@memory_app.command("import")
def memory_import(
    path: Annotated[Path, typer.Argument(help="JSON Lines, one record per line.")],
    corpus: Annotated[
        str | None,
        typer.Option(
            "--corpus",
            help="Which corpus to load, defaulting to the configured memory.corpus_id.",
        ),
    ] = None,
    replace: Annotated[
        bool, typer.Option("--replace", help="Overwrite a session the corpus already holds.")
    ] = False,
) -> None:
    """Load distilled records from a file into the configured store.

    For a campaign whose corpus was written by an earlier system: the records go in
    through the same port the distilling pass writes through, so the backend's own index
    is built by the normaliser that retrieval queries with rather than by whatever wrote
    the file. ``docs/adapters/sregym.md`` carries the export the archive is read out by.

    The file is read and validated whole before the store is opened, so a bad line leaves
    nothing behind. Keyed by session: without ``--replace`` a session already in the
    corpus is left exactly as it is, which makes an interrupted import safe to re-run and
    stops a second file quietly revising a record a run has already retrieved.

    Args:
        path: The JSON Lines file.
        corpus: The corpus to write into. The partition is the database itself, so this
            decides which campaign's records these become.
        replace: Overwrite the records the corpus already holds.
    """
    settings = _resolved()
    if corpus is not None:
        settings = settings.model_copy(
            update={"memory": settings.memory.model_copy(update={"corpus_id": corpus})}
        )
    try:
        records = parse_records(path.read_text(encoding="utf-8"))
    except (OSError, MalformedRecord) as exc:
        typer.echo(f"{path}: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    try:
        storage = build_storage(settings)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc
    try:
        report = import_records(storage.index, records, replace=replace)
    finally:
        storage.close()
    typer.echo(
        f"{settings.memory.corpus_id}: imported {report.imported}, "
        f"replaced {report.replaced}, skipped {report.skipped} already present"
    )
    if report.unknown_states:
        # The ranker demotes by state, and a state its table cannot name is worth the
        # default weight. A typo in an export is therefore a re-weighting of the corpus,
        # and this is the only place it is visible.
        typer.echo(
            "lifecycle state(s) this build does not name, stored as given and ranked at "
            f"the default weight: {', '.join(report.unknown_states)}"
        )


@approvals_app.command("list")
def approvals_list(
    status: Annotated[str, typer.Option("--status", help="One status, or 'all'.")] = "pending",
    limit: Annotated[int, typer.Option("--limit", help="How many to show.")] = 100,
    as_json: JsonOption = False,
    url: UrlOption = None,
) -> None:
    """List the requests waiting on a person.

    Args:
        status: One status, or ``all`` for every status.
        limit: How many to show.
        as_json: Print the raw records.
        url: A running server to ask instead of the local store.
    """
    if url is not None:
        payloads = _api(url, "GET", "/hitl", params={"status": status, "limit": limit})["requests"]
    else:
        with _approvals() as store:
            selected = None if status in ("all", "") else _status(status)
            payloads = [request_payload(r) for r in store.list_requests(status=selected)[:limit]]
    if as_json:
        typer.echo(json.dumps(payloads, indent=2, sort_keys=True))
        return
    if not payloads:
        typer.echo("no requests")
        return
    for payload in payloads:
        typer.echo(
            f"{payload['request_id']}  {payload['status']:<9}  {payload['stage']:<9}  "
            f"{payload['created_at']}  {payload['command']}"
        )


@approvals_app.command("show")
def approvals_show(request_id: str, as_json: JsonOption = False, url: UrlOption = None) -> None:
    """Show one request in full: the command, its targets, the rule, and the expiry.

    Args:
        request_id: Which request.
        as_json: Print the raw record.
        url: A running server to ask instead of the local store.
    """
    payload = _one(request_id, url)
    if as_json:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    typer.echo(_render(payload))


@approvals_app.command("approve")
def approvals_approve(
    request_id: str,
    reason: ReasonOption = "",
    by: ByOption = "generic human",
    url: UrlOption = None,
) -> None:
    """Allow the write this request is holding.

    Args:
        request_id: Which request.
        reason: Why, recorded on the request.
        by: Who is deciding.
        url: A running server to decide through instead of the local store.
    """
    _decide(request_id, approve=True, reason=reason, by=by, url=url)


@approvals_app.command("deny")
def approvals_deny(
    request_id: str,
    reason: ReasonOption = "",
    by: ByOption = "generic human",
    url: UrlOption = None,
) -> None:
    """Refuse the write this request is holding.

    The role sees a refusal as a tool result and carries on: a denied write is something
    to report or work around, not a crash.

    Args:
        request_id: Which request.
        reason: Why, recorded on the request.
        by: Who is deciding.
        url: A running server to decide through instead of the local store.
    """
    _decide(request_id, approve=False, reason=reason, by=by, url=url)


def _decide(request_id: str, *, approve: bool, reason: str, by: str, url: str | None) -> None:
    """Record one decision, locally or through a running server."""
    if url is not None:
        body = {"approve": approve, "reason": reason, "decided_by": by}
        _api(url, "POST", f"/hitl/{request_id}/decide", json=body)
    else:
        with _approvals() as store:
            record = store.decide(
                request_id,
                status="approved" if approve else "denied",
                decided_by=by,
                reason=reason,
            )
        if record is None:
            typer.echo(f"{request_id}: not found, or already decided", err=True)
            raise typer.Exit(code=1)
    typer.echo(f"{request_id}: {'approved' if approve else 'denied'}")


def _one(request_id: str, url: str | None) -> dict[str, Any]:
    """Read one request, locally or through a running server."""
    if url is not None:
        payload: dict[str, Any] = _api(url, "GET", f"/hitl/{request_id}")
        return payload
    with _approvals() as store:
        record = store.get_request(request_id)
    if record is None:
        typer.echo(f"{request_id}: not found", err=True)
        raise typer.Exit(code=1)
    return request_payload(record)


@contextmanager
def _approvals() -> Iterator[ApprovalStore]:
    """Open the configured store for the length of one command."""
    storage = build_storage(_resolved())
    try:
        yield storage.approvals
    finally:
        storage.close()


def _api(url: str, method: str, path: str, **kwargs: Any) -> Any:
    """Call one API route, turning a refusal into an exit code rather than a traceback."""
    with httpx.Client(base_url=url.rstrip("/"), timeout=30.0) as client:
        response = client.request(method, f"/api/v1{path}", **kwargs)
    if response.status_code == 404:
        typer.echo(_detail(response), err=True)
        raise typer.Exit(code=1)
    response.raise_for_status()
    return response.json()


def _detail(response: httpx.Response) -> str:
    """Read the reason a route refused, falling back to its status line."""
    try:
        detail = response.json().get("detail")
    except ValueError:
        detail = None
    return str(detail or f"{response.status_code} {response.reason_phrase}")


def _configuration(settings: Settings) -> Configuration:
    """Resolve user definitions with the same concise refusal as config check."""
    try:
        return load_configuration(settings)
    except (OSError, RoleRegistryError, RuntimeError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc


def _resolved() -> Settings:
    """Load settings, reporting a bad configuration as a message rather than a traceback."""
    try:
        return load_settings()
    except (ValidationError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc


def _status(value: str) -> ApprovalStatus:
    """Accept only a status the store knows, so a typo is not a silent empty list."""
    if value not in _STATUSES:
        typer.echo(f"unknown status {value!r}", err=True)
        raise typer.Exit(code=2)
    return cast(ApprovalStatus, value)


def _render(payload: Mapping[str, Any]) -> str:
    """Lay out one request for a person deciding on it.

    Everything a reviewer needs is here, including the arguments. The prototype showed
    the first six argument tokens, so a patch named the resource and never the patch
    body — the only part that says what changes. Approving that is not review.
    """
    lines = [
        f"request   {payload['request_id']}",
        f"session   {payload['session_id']}",
        f"stage     {payload['stage']}",
        f"tool      {payload['tool_name']} ({payload['action']})",
        f"status    {payload['status']}",
        f"rule      {payload['rule'] or '-'}",
        f"created   {payload['created_at']}",
        f"expires   {payload['expires_at'] or 'never'}",
    ]
    if payload["decided_at"]:
        lines.append(f"decided   {payload['decided_at']} by {payload['decided_by']} (claimed)")
    if payload["reason"]:
        lines.append(f"reason    {payload['reason']}")
    lines += ["", "command", f"  {payload['command']}"]
    lines += _block("targets", payload["targets"])
    lines += _block("selectors", payload["selectors"])
    arguments = payload["payload"].get("arguments")
    if arguments:
        lines.append("arguments")
        lines += [
            f"  {line}" for line in json.dumps(arguments, indent=2, sort_keys=True).splitlines()
        ]
    return "\n".join(lines)


def _block(title: str, values: Sequence[str]) -> list[str]:
    """Render one titled list, or nothing when there is nothing to show."""
    return [title, *(f"  {value}" for value in values)] if values else []


def _flatten(tree: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Render nested settings groups as dotted names."""
    flat: dict[str, Any] = {}
    for key, value in tree.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat |= _flatten(value, f"{name}.")
        else:
            flat[name] = value
    return flat


if __name__ == "__main__":  # pragma: no cover
    app()
