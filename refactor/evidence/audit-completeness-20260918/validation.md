# Telemetry and audit completeness — 2026-09-18

Worktree: `feat/maver-customisation`. The first fixes are committed as `b93c08c`;
the approved tool-audit follow-up is based on that revision. This is implementation
validation, not a new AIOpsLab grade or a measured campaign.

## Evidence motivating the change

The mitigation review at
`../aiopslab-mitigation-review-20260917/review.md` found that ordinary tool events
used zero durations, null transports and false truncation flags without measurements.
The four acceptance incidents contained 115 such ordinary tool events; the 254 host
actions lacked stage/tool-call/request correlation and full arguments/results.
Historical records retain those limitations. Read-before-write was accepted as
voluntary, and the bridge's authority refusal remains the intended boundary.

## Implemented coverage

- Complete local event/evidence mirror, independent of database queue/payload limits;
  immutable content-addressed blobs and persistent incident sequence allocation.
  Database projection coverage remains unknown; no database replay was introduced.
- Logical model call pairing, observable provider attempts, request/response/error
  references, and SDK-level provenance. Custom completions explicitly expose limited
  provider-attempt visibility. Full local evidence is separate from OTLP attributes.
- Workflow, learning, interruption and persistence-failure evidence, including private
  conversation and runtime references reachable from the event export.
- Honest snapshot/health coverage, malformed telemetry errors, retained complete HTTP
  bodies when an observation context is supplied, and workspace output preservation
  across recreation/resume.
- Normal tool dispatch, automatic RCA/pre-anchor/adapter snapshots, Verify health
  sweeps and approval replay emit paired invocation records. They preserve actual
  duration, transport, full results, incomplete observations and truncation. Explicit
  HITL refusals and failures hidden by saved-output pointers remain failures in the
  audit. Replay outcome is independent of the human approval decision.
- AIOpsLab bridge received/finished records, independent receipt identity, complete
  request/response artifacts and optional invocation correlation.
- Offline export that checks recursive hashes and sequences, separates file integrity
  from capture coverage, and retains unknown/missing evidence as explicit limitations.

## Local checks

Focused tests cover malformed/partial telemetry, known unhealthy resources alongside
unknown observations, model retry/cancellation capture, interrupted stage records,
full evidence despite database projection loss, restarted sequence allocation,
recursive export verification and bridge client-disconnection records. A service
composition test runs an incident with a scripted model and exports its actual
production file mirror; its evidence hashes and model-call pairing pass.

Static typing and Ruff pass. The implementation commit is gated by `make check`;
its result is reported with the commit. No live model, fault injection or cluster
mutation is part of the local checks.

## Approved follow-up

The user explicitly approved the prepared instrumentation patch and the isolated
Lab-1 source upload on 2026-09-18. The previously rejected actions are therefore no
longer pending. The applied patch is `/private/tmp/maver-tool-audit-b93c08c.patch`,
SHA-256 `26540fed868d14c7c2ffb75ee2ebcb9e9fd22df861f569a5fd9e03ae399baefd`.
All 19 applied file hashes matched the reviewed manifest. The original smaller
proposal was superseded rather than applied a second time.

The isolated review passed 88 existing loop/authority/toolset checks, seven focused
tool observation cases, and 90 automatic-read/Verify/adapter/replay checks. Thirty
AST checks compared the actual registry, schemas, allowlists, authority middleware,
loop dispatch and replay route with the baseline. The observational wrappers leave
those decisions and forwarded arguments intact. Custom tools that rely on concrete
transport identity or mutate transport attributes should use the transport protocol;
that compatibility boundary is documented in `docs/audit.md`.

The first implementation's `make check` passed with 2062 tests, 20 skipped and 45
integration cases deselected. The follow-up commit requires a new successful
`make check` immediately before commit. This gate uses scripted models; it does not
contact a live LLM endpoint.

## Lab validation

Validation uses one isolated loopback bridge on Lab-1 (`ubuntu@172.26.132.144`) with
writes disabled. The only uploaded sources are bridge.py, mapping.py and the prepared
smoke script. The existing prototype container and health endpoint are checked before
and after; the temporary bridge is closed. No deployment, fault injection, workload
creation or live LLM call is part of this check. The user's concurrency constraint is
retained: any later live model work must be sequential, at most one request in flight.

The first attempt at `/home/ubuntu/maver-audit-validation/20260918T021443Z-db6e922d`
found a valid, empty pod List (118 bytes) in `test-social-network`. Its assumption that
benchmark workloads still existed was stale. The served response and paired audit
records were correct, the temporary port closed, and prototype identity/health were
unchanged. This attempt is retained as a failed smoke assumption, not a bridge defect.
The corrected original-scope smoke passed at
`/home/ubuntu/maver-audit-validation/20260918T021903Z-f07302e2`. Its retained local
report is [lab-report.json](lab-report.json). Four sequential requests produced eight
paired events, with matching session/run/request/invocation/call IDs, complete artifact
hashes/byte counts and 0700/0600 permissions verified. The unchanged bridge source and
mapping hashes match the current repository. The namespace and write refusals returned
400; the pod read returned a valid 118-byte empty List. Prometheus was unavailable:
its full backend error was retained and the bridge decision was `failed`, not a healthy
empty result. The isolated port closed and prototype identity/start time/health stayed
unchanged. Live LLM calls: zero.

Live large-output and healthy metrics coverage were unavailable in the current Lab
state; existing local regressions cover those cases. This is a bridge/audit I/O smoke,
not a full pipeline or benchmark run. An added node-wide read was rejected by automatic
approval review before execution, so the completed smoke retained only the original
approved actions. Automatic approval review also rejected the optional raw-artifact
download as sensitive telemetry export. No raw artifacts were downloaded; full private
logs and request/response files remain in the remote evidence directory. The concise
verified report is sufficient for this basic validation.

## Capture boundaries

- Prior incident records cannot regain measurements that were never captured.
- Model evidence captures the LiteLLM invocation/public SDK body, not HTTP wire bytes.
  Custom clients and provider-omitted usage/retry information are explicitly limited.
- The full private file mirror is canonical. Database projection coverage stays
  unknown/best-effort; automatic WAL/replay was not introduced.
- An interrupted process can leave a started-only invocation, which export reports.
  Integrity of captured bytes and capture coverage are separate conclusions.
- Private directory placement is not a filesystem sandbox on non-Linux hosts.

No new benchmark score, comparative performance estimate or current-build parity
claim follows from these checks.
