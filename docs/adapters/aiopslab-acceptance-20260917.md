# AIOpsLab acceptance, 2026-09-17

Four explicit tasks ran sequentially on Lab-1, one attempt each. All reached an
external AIOpsLab grade: three passed and one failed. No graded result was retried.

| Task | Problem | Grade | Wall time, including setup and cleanup |
|---|---|---|---|
| Detection | `noop_detection_hotel_reservation-1` | Pass | 6m 23s |
| Localization | `k8s_target_port-misconfig-localization-1` | Pass | 10m 15s |
| Analysis | `k8s_target_port-misconfig-analysis-1` | Fail | 14m 19s |
| Mitigation | `k8s_target_port-misconfig-mitigation-1` | Pass | 28m 39s |

The suite ran from 01:45:23 to 02:44:59 UTC. The healthy detection case and common
fault family were selected to cover the four task contracts, not for historical
scores. Four tasks do not estimate benchmark-wide accuracy or a Verify effect.

## Tested condition

- MAVER source: `f33ddc8d4bdcf8f9557084fc8dbaa473b081c2af`, version `0.1.0`.
- Source archive SHA-256: `488af6eba0a4a5e65b2987d7420aaf34ab2d1367b3c220f286ebfa3028ebc691`.
- Runtime image digest: `sha256:c8526ca6c8294db45127bc9597a3ff3ccc215ae4723cb879f3a1d64d24cf60bc`.
- AIOpsLab checkout: `ddf7e40619689dad75eaf8f2174e263c4157ec76`, with its
  existing external fixes retained and captured in `evidence/benchmark.patch`.
- Lab-1 controlled the shared three-node Lab cluster; there was one problem at a time.
- Shipped `qwen3.8-27b` profile on the private endpoint, unchanged sampling:
  temperature 1, top-p 0.95, top-k 20, min-p 0, maximum 8,192 output tokens,
  seed 42, presence/frequency penalties 0, repetition penalty 1.
  Qwen3.6's endpoint was unavailable. No named historical arm was claimed.
- Multi-agent workflow, memory/retrieval/learning/second-pass off, automatic
  namespace snapshot on, per-agent maximum 100 turns before task-specific limits.
- Verify enabled, health gate off, inconclusive retries 1, shared remediation
  retries 2, Verify handback cap 1. Authority mode `auto`, environment `local`.
- Fresh embedded SQLite workspace, isolated loopback API on port 18080 and bridge
  on 18190; the prototype container remained available on 8080.
- Named YAML pipeline `acceptance_recovery`, implementation `multi`, with
  `verify_handback_retries: 1`. The three read-only tasks selected `diagnose`;
  mitigation selected `acceptance_recovery` at admission.
- Agent timeout 7,200 seconds, problem timeout 7,800 seconds, maximum attempts 1.
  Buildx was extracted into the acceptance directory to build the image; system
  Docker configuration and the prototype image were not replaced.

## What the evidence establishes

All four incident audit streams have contiguous, unique sequence numbers and their
persisted pipeline selections match admission. Conversations remain separate per
agent invocation. Model usage is taken from MAVER's ledger, not AIOpsLab's outer
single-submit token placeholders: 64 model calls, 812,080 prompt tokens and 54,368
completion tokens across the four incidents. Reasoning/cache token counts were not
reported separately by the provider.

The three diagnosis-only tasks recorded no write approvals. Mitigation recorded an
automatically approved Service patch and deployment restart, followed by successful
namespace-scoped tool results. Bridge entries carry the same incident ID and matching
dispatch times. Verify's attempted generic `kubectl exec` was refused by the bridge;
its successful entry-point probes returned HTTP 200. A separate malformed compose
GET returned HTTP 400 and is not itself proof of a successful compose transaction.
The external mitigation grader independently returned success.

The new Execution prompt was present in the tested source. Its submitted execution
result confined its success claim to the applied resource changes and delegated
end-to-end recovery to Verify. Verify submitted a recovered verdict.

The configured Verify cap was recorded, but no handback was taken and no submission
normalization occurred. This run therefore does not validate either boundary on a
live incident; their automated regression coverage remains separate evidence.

Re-running the identical completed sweep command skipped all four graded cases,
including the failure. SHA-256 hashes of all eight result/submission JSON files were
unchanged. This verifies completed-result resume behavior, not interruption during an
active incident or checkpoint recovery.

## Analysis failure

MAVER correctly identified `Service/user-service` forwarding to target port 9999 while
the application listened on 9090. It submitted **Application / Misconfiguration**.
The task grader expects **Virtualization / Misconfiguration**, so
`system_level_correct=false`, `fault_type_correct=true`, `success=false`.

The submission sidecar records `source=model`, `rule=null`: no adapter fallback
reclassified the answer. This is a model taxonomy error despite a correct causal
description. The original failing grade is retained.

## Defect found and separate correction check

Successful large bridge responses were cut at 10,000 characters before internal
consumers parsed them. Snapshot reads became invalid JSON of exactly 10,013 characters
including the truncation marker, and trace tools reported unparseable telemetry.
The model worked around missing snapshots with narrower kubectl reads.

The correction keeps successful kubectl and structured telemetry complete until tools
parse, scope and save them. Model-facing display bounds and bridge authority remain
in place. Snapshot parsing, automatic/explicit output saving, trace filtering and
alert parsing regressions all fail on the original implementation and pass with the
correction.

A separate read-only check on Lab-1 used the existing image's dependencies and the
corrected source, with bridge writes disabled and no new fault injection. It parsed
57,929 characters of node JSON into three nodes, rendered a namespace snapshot with
zero unavailable sections, and saved all 57,929 characters through `save_as`.
The source archive SHA-256 was
`75ec9551fb77c0292ca9eb88be17f8a9b3f7b8452d6063f673c25df755f8906f`;
the tested transport file SHA-256 was
`610791ba232fefe96e2c5dbceeb0e9253238ef33df281456d4ca582042ea91e6`.

This read-only check does not re-grade the four tasks or establish that the correction
improves model outcomes. Large-trace scoping is covered locally with the real transport
and tool, using a synthetic bridge response.

## Evidence and remaining acceptance scope

Raw evidence remains on Lab-1 under:

```text
/home/ubuntu/maver-acceptance/20260917T001942Z-f33ddc8/
  evidence/plan.json                  # exact configuration and provenance
  evidence/first-run-status.tsv       # original durations and grades
  evidence/default.sqlite            # consistent final session/audit backup
  evidence/audit/                     # ordered events and per-incident summaries
  evidence/benchmark.patch            # existing external harness modifications
  evidence/resume.json                # skip verification and unchanged result hashes
  evidence/complete-output-readback.json
  results/smoke/results/              # grader JSON, submissions and bridge logs
  console.log
```

The isolated MAVER test container is stopped and its workspace/results are retained.
The prototype container remains running. Benchmark recovery and teardown completed.

This covers acceptance C1 (all four task types graded) and C3 (audited mitigation
write path). C2's interrupted-sweep scenario remains untested; only completed-result
resumption was checked. The full re-baseline (C4), Verify replicate study (C5), live
cap exhaustion and live normalization remain pending. Historical SREGym parity
sign-off and manuscript results are unchanged.
