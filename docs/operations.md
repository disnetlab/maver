# Incident operations

MAVER runs one process against one deployment. The HTTP API, optional Kafka intake,
session store and audit directory belong to that deployment. Configure local CLI
commands with the same storage settings as the server; commands using `--url`
contact the server instead.

The API does not authenticate callers. Restrict access using the deployment's
network controls. Operator names are recorded as supplied, not verified identities.
See [Kubernetes deployment](../deploy/k8s/README.md) for access and persistence setup.

## Submit and inspect an incident

```bash
curl --fail-with-body http://127.0.0.1:8080/api/v1/incidents \
  -H 'Content-Type: application/json' \
  -H 'Idempotency-Key: monitoring-event-001' \
  --data '{"alert":{"alertname":"HighErrorRate","namespace":"shop"},"target_stage":"full"}'
```

The response includes a `session_id`. Poll it with:

```bash
curl --fail-with-body http://127.0.0.1:8080/api/v1/incidents/SESSION_ID
```

`target_stage: "diagnose"` stops after diagnosis. An optional `pipeline` selects a
configured workflow; optional `metadata` is private runtime context. See
[customisation](customisation.md) for selection rules and agent configuration.

The incident response separates two concepts:

| Field | Meaning |
|---|---|
| `run_status` | Execution progress: `pending`, `running`, `awaiting_human`, `resuming`, `needs_reconciliation`, `completed`, `failed`, or `cancelled`. |
| `status` | Memory lifecycle: `pending`, `tentative_success`, `human_verified`, `invalidated`, or `halted`. A completed run does not itself verify a lesson. |

`awaiting_human` requires an approval decision. `needs_reconciliation` requires an
operator to inspect an interrupted run. Neither status means the incident is actively
making progress. `POST /api/v1/incidents/SESSION_ID/cancel` cancels a run carried by
the current process; it does not undo operations already dispatched to the cluster.

## Retries and duplicate alerts

An HTTP `Idempotency-Key` identifies one delivery. MAVER stores its receipt with the
initialized session in one database transaction. Repeating the same key and request
returns the original session, including after completion or restart. Reusing a key
with a different request returns HTTP 409. Keys contain 1–255 characters.

Choose a new key for a genuinely new incident. Without a key, matching alerts are
deduplicated only while their runs are live in the current process. An exact-key
retry does not restart or resume the original run.

Admission returns HTTP 429 when capacity is exhausted and HTTP 503 when draining.
Temporary initialization refusals can also return HTTP 503. New deliveries for an alert awaiting reconciliation
return HTTP 409 with `Retry-After`; exact-key retries still return their saved receipt.

Shutdown tracks admission from pipeline selection through session initialization,
as well as running incidents and delayed Learning. Work already accepted shares the
configured drain window. An admission that finishes after that window cannot start a
pipeline; with restart recovery enabled it is retained for reconciliation, otherwise
it is recorded as cancelled. Database work that is still settling keeps its storage
open until process exit rather than using a store already closed by shutdown.

## Decide a pending approval

With prompt-based write authority, an incident can park before a tool executes.
Inspect the request before approving or denying it:

```bash
uv run maver approvals list --url http://127.0.0.1:8080
uv run maver approvals show REQUEST_ID --url http://127.0.0.1:8080
uv run maver approvals approve REQUEST_ID --by on-call \
  --reason 'Reviewed the command and its targets' --url http://127.0.0.1:8080
```

Use `deny` instead of `approve` to refuse the operation. Without `--url`, these
commands use the configured local store; the server's approval sweep picks up the
decision. A server API decision schedules continuation immediately. Approval permits
the recorded operation under the current authority policy and does not confirm that
the incident has recovered.

## Kafka intake and quarantine

Install the optional dependency with `uv sync --extra kafka`, or build the container
with `--build-arg EXTRAS=kafka`. Configure:

```bash
export MAVER_INTAKE__KAFKA__ENABLED=true
export MAVER_INTAKE__KAFKA__BOOTSTRAP_SERVERS=broker:9092
export MAVER_INTAKE__KAFKA__TOPIC=alerts
export MAVER_INTAKE__KAFKA__GROUP_ID=maver
```

Kafka is disabled by default. Each message uses the same JSON body as HTTP incident
submission. Set `MAVER_INTAKE__HTTP_ENABLED=false` to disable HTTP incident submission
while retaining read and operator routes. At least one intake must remain enabled.

Delivery is at least once. A receipt keyed by consumer group, topic, partition and
offset prevents redelivery from opening a duplicate session. MAVER commits the
specific offset only after durable admission or quarantine. Capacity and temporary
admission refusals leave the message available for retry.

Malformed JSON, tombstones and permanent admission refusals enter the session
database's quarantine. Original bytes are retained as base64; a tombstone has an
empty byte payload and a distinct null-value reason. A quarantine write failure
leaves the offset uncommitted. Inspect records locally:

```bash
uv run maver queue list
uv run maver queue show RECORD_ID
```

Retry through a running server connected to the same store:

```bash
uv run maver queue retry RECORD_ID --url http://127.0.0.1:8080
uv run maver queue retry RECORD_ID --corrected incident.json \
  --url http://127.0.0.1:8080
```

`incident.json` must contain the whole incident request, including its `alert`
object. A corrected body is required for malformed messages and tombstones. Retry
uses normal admission and a durable quarantine key. Repeating the same retry returns
its original session; a different body with that key returns HTTP 409. The record
retains the original delivery and the replacement session reference.

## Restart recovery

Enable application-owned checkpoints explicitly:

```bash
export MAVER_SERVER__RESUME_INTERRUPTED_INCIDENTS=true
```

MAVER saves incident state before each workflow stage. On startup it serially
replays eligible interrupted RCA, Planning and Verify stages with the saved pipeline
and deployment configuration. Work within the interrupted stage may run again;
recovery is not continuation from the last individual model or tool call.

Execution, adapter-owned episodes, changed configurations, incomplete checkpoints
and unvetted custom tools require reconciliation. Recovery mode withholds
`bash_exec` and `code_exec` from replayable stages, including their initial execution.
It does not reattach external benchmark episodes. Keep one process and retain the
configured SQLite volume or PostgreSQL database across restarts.

An interrupted approval continuation also requires reconciliation. Once continuation
has begun, its approved operation may have taken effect even if no result was saved.
With recovery enabled, startup parks interrupted `resuming` sessions for review;
MAVER does not automatically replay their approved operations. An incident still
waiting for an approval decision retains the usual approval workflow.

The flag defaults to false. Durable intake receipts prevent duplicate admission
independently of this flag; a receipt alone does not resume interrupted work.

### Reconcile an interrupted run

Inspect the saved checkpoint, artifacts and last recorded tool action locally:

```bash
uv run maver incidents reconcile-show SESSION_ID
```

Read the affected resources and service state before choosing a disposition. A
missing tool result does not establish whether its cluster operation took effect.
Record the fresh observation and the reason for the decision:

```bash
uv run maver incidents reconcile-decide SESSION_ID --action close \
  --observed-state 'Service is healthy; the resource change is present' \
  --reason 'No further remediation required' --by on-call

uv run maver incidents reconcile-decide SESSION_ID --action replan \
  --observed-state 'Service is still unhealthy' \
  --reason 'Reassess from current cluster state' --by on-call
```

Both commands accept `--url`; the default is `http://127.0.0.1:8080`. `close` ends
the uncertain attempt. `replan` starts a fresh linked incident from the stored alert
and pinned pipeline; it does not replay the old write or start directly at Planning.
The old closure and initialized replacement commit together. A repeated decision
returns HTTP 409. If the process stops before scheduling the replacement, its durable
record remains available; enabled restart recovery parks its missing checkpoint for
reconciliation.

Replanning requires the saved deployment and pipeline composition to match. An
adapter-owned episode must be closed and restarted through its driver. The API's
`GET /api/v1/incidents/SESSION_ID/reconciliation` returns status and evidence-presence
metadata; detailed evidence is available through local inspection.

## Review an outcome

Tool approval permits an operation. Outcome review records what later evidence says
about an incident that has reached `completed` or `failed`:

```bash
uv run maver incidents feedback SESSION_ID confirmed --by on-call \
  --reason 'Recovery remained stable' --evidence 'Health probes pass'

uv run maver incidents feedback SESSION_ID corrected --by on-call \
  --reason 'Inspected the service and application listener' \
  --evidence 'The service targeted a port with no listener' \
  --corrected-cause 'Service targetPort was wrong'
```

`confirmed` requests `human_verified`; `rejected` and `corrected` request
`invalidated`. Only `corrected` accepts and requires `corrected_cause`. Lifecycle
transition rules still apply: incompatible reviews return HTTP 409, including direct
confirmation of an unclassified `pending` lesson.

The API equivalent is `POST /api/v1/incidents/SESSION_ID/feedback`, with `verdict`,
`decided_by`, `reason`, `evidence`, and optional `corrected_cause`. `GET` on the same
path returns the append-only history. Reviews are ordered by a sequence allocated
under the session lock. Session lifecycle and existing memory metadata change in the
same transaction, leaving original agent artifacts and observation text intact.

Delayed Learning receives the latest review as external feedback. Review text is
therefore model-visible to Learning. If a review changes before the lesson is committed,
that pass does not write its stale lesson and is retried with the latest review. Existing
stored lessons retain their original text when reviewed.
Both storage backends check the review sequence inside the corpus write transaction,
including reviews that arrive after Learning's final application-level check.

## Delayed Learning and retries

Eligible delayed Learning records an intent in the session database before scheduling
the job. The intent retains its due time and supplied external feedback. Startup
restores these intents, and a periodic sweep repairs missing scheduled jobs. Failed
passes retry after one minute, with exponential backoff capped at fifteen minutes;
retries continue until successful distillation. Restored jobs run one at a time, and
an already distilled session does not incur another model call.

Temporarily disabling Learning retains pending intents and their feedback. Re-enable
it on a later restart to restore that work. A long-running pass is not queued again
by the repair sweep. Kafka shutdown also leaves unfinished admission and quarantine
writes tracked while their storage operations settle.

Eligibility remains part of the workflow: diagnosis-only runs are excluded, and
adapter-owned episodes keep their synchronous Learning pass after external grading.
MAVER does not backfill every undistilled session. Existing scheduled Learning jobs
are adopted with their saved feedback during upgrade. The saved deployment and
pipeline configuration must still match; restore that configuration to unblock work
that was deferred because behavior changed.

## Health, storage and upgrades

`GET /healthz` is a dependency-free liveness check. `GET /readyz` checks startup
configuration, draining state, storage and the enabled queue consumer. A stopped consumer makes readiness fail;
inspect the server log, correct the cause and restart the service. Readiness is not
a complete check of model-provider or telemetry availability.

Storage migrations run when the configured store is opened. Layout version 9 includes
durable intake receipts, quarantine, ordered operator feedback and Learning intents. Older layouts
upgrade automatically; a binary that encounters a newer layout refuses to open it.
Back up durable state before upgrading and restore a matching database backup when
rolling back across layout versions.

The workspace also holds scheduled jobs and audit evidence, including when sessions
use PostgreSQL. Retain it across restarts. See [audit evidence](audit.md) for capture,
export, integrity checks and access boundaries.
