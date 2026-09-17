# Telemetry and audit completeness — 2026-09-18

Worktree: `feat/maver-customisation`, based on `776ea5c`. This is implementation
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

## Remaining blocked work

Automatic approval review rejected the generic tool instrumentation patch twice as
potentially affecting execution. The prepared patch remains unapplied:
`/private/tmp/maver_tool_observation_patch.py`. Its isolated review copy passed 118
focused authority, approval, allowlist and loop checks; an AST comparison preserved
the middleware dispatch call and registry/decorator definitions. Generic dispatch,
automatic snapshots and approval replay still need observation integration, so full
per-tool timing/transport/result coverage is not claimed. User approval is pending.

Automatic approval review also rejected uploading the prepared isolated bridge smoke
to Lab-1 (`ubuntu@172.26.132.144`) as private-source egress. No remote mutation or live
smoke took place. The exact preparation is retained at
`/private/tmp/maver_bridge_live_upload.py` and
`/private/tmp/maver_bridge_live_smoke.py`; it copies bridge.py, mapping.py and the test
script, starts a temporary loopback bridge with writes disabled, checks real reads
and refusals, then stops it. Specific user approval for that upload is pending.

A rejected database WAL/replay proposal remains outside the repository at
`/private/tmp/maver-audit-wal/rejected-storage-proposal.patch`. The applied additive
file mirror uses a different, approved design and leaves database semantics intact.
