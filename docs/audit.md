# Audit evidence

The HTTP service writes canonical audit events and full referenced evidence under
`<server.workdir>/audit`. This directory is separate from each incident workspace.
JSONL events and content-addressed JSON blobs are written before events are offered
to the existing database sink. The database remains a best-effort query projection:
queue limits and payload limits can omit or shorten its records. Database coverage
is reported as unknown; there is no automatic database replay from the file mirror.
Custom applications that construct a `Deployment` directly must supply their own
sink, such as `DurableAuditSink(StoreSink(storage.runs), audit_dir=...)`.

## Export an incident

```bash
uv run maver audit export INCIDENT_ID --output ./incident-audit
uv run maver audit export INCIDENT_ID --output ./incident-audit \
  --audit-dir /path/to/workdir/audit --bridge-log /path/to/bridge.log
```

Export is offline: it needs neither a model nor a database connection. Use an empty
or nonexistent output directory. The bundle includes ordered events, recursively
referenced evidence and a manifest. References carry SHA-256 and byte length, and
export refuses references that escape their evidence root. Sequence allocation is
persisted across process restarts and allocator eviction; missing sequence positions
are reported instead of silently renumbered.

File integrity and capture coverage are separate. Valid checksums mean the exported
bytes match their references. They do not prove every operation was observed. Open
calls, missing request/response evidence, unavailable retry observations and missing
bridge correlation are reported as capture limitations. Export after an incident and
its asynchronous learning finish for a settled view; an active incident can legitimately
contain unfinished calls. Abrupt process termination can leave a started-only call.

## What is captured

- Model calls retain the ordered neutral request, actual LiteLLM invocation arguments
  when available, normalized response and public provider response body. Repeated
  messages and tool schemas share immutable blobs. This is SDK-level evidence, not
  the HTTP wire payload. Each observed provider attempt has its own duration, usage,
  response or error and links to one logical call. Logical usage describes the final
  attempt; do not add it to attempt usage when calculating totals.
- Custom completion implementations provide neutral request/response evidence.
  `attempts_observed=0` means retries were not observable through the native client,
  not that the provider performed no work. Provider-omitted usage remains unknown.
- Workflow and learning records retain separate conversations and agent identity.
  Terminal lifecycle evidence distinguishes completion, pause, cancellation and
  failure, including failures to persist the normal runtime projection.
- Tool event schema 4 supports invocation IDs, measured duration, transport, full
  result references, truncation and incomplete-observation details. These fields are
  optional: legacy or uninstrumented producers cannot supply them. Normal dispatch,
  automatic snapshots, Verify health sweeps and approval replay record started/terminal
  attempts. HITL refusals are recorded as refusals; approving a replay does not imply its
  command succeeded. Complete backend and filtered tool outputs remain in private
  artifacts even when the model sees a bounded preview or a saved-file pointer.
  Automatic observations have no provider tool-call ID. Unknown or interrupted work
  is never assigned a fabricated duration or success result.
- AIOpsLab bridge schema 2 independently records received/finished actions and full
  parsed request contracts and response bodies. Correlation IDs are retained when
  callers supply them. Client disconnection is distinct from dispatch failure.
  Legacy host logs cannot recover arguments or outputs that were never saved.

Telemetry parsing now distinguishes missing/malformed data from a valid empty result.
Snapshot coverage names unavailable reads. Health checks retain known unhealthy
resources while flagging incomplete observations. They cannot recover data that an
upstream service already discarded or turn an unknown observation into proof of health.

The dispatch observer supplies a context with a protocol-compatible transport proxy.
Custom tools should use `ToolTransport` methods and properties; relying on a concrete
transport type or mutating its attributes through `ctx.transport` is not supported
by this observer. The wrapped methods receive the original arguments and return the
original values. Directly calling an undecorated tool function from custom workflow
code requires `observe_tool` to obtain the same audit coverage.

## Storage and privacy

Evidence directories use mode 0700 and files use 0600. Full local audit bodies may
contain incident telemetry, model reasoning and private runtime metadata. They are
not automatically added to prompts or the retrieval corpus. Back up and retain this
directory with the incident database; no automatic audit retention policy is installed.
The service account and administrators can read it. Filesystem isolation from agent
shell tools depends on the configured sandbox: Landlock is unavailable on non-Linux
development hosts, so directory placement alone is not an access boundary there.

Credentials and transport headers are excluded from model and bridge captures.
Captured application bodies can still contain sensitive incident data. Optional
OTLP spans carry operational attributes and exception types, without exception
messages or stack traces; the full local evidence store is separate. Model and OTLP
endpoints are deployment choices, so private model serving requires private endpoints
and the corresponding network policy.

`observability.event_log_enabled` and `observability.event_log_path` remain accepted
legacy fields but are ignored by the service composition root. They do not disable
or relocate this mirror. Set `server.workdir` for the service's storage location, or
supply a different sink when composing a custom application.

Audit schema version 4 does not change the structured submission artifact schemas.
Do not rewrite old measurements with newly inferred durations, success flags or
missing bodies. Changed telemetry observations define a new experimental condition;
compare results using the schema and software revision recorded for each run.
