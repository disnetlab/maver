# Gemini 3.5 Flash-Lite mitigation validation — 2026-09-21

One AIOpsLab mitigation problem, `k8s_target_port-misconfig-mitigation-1`, ran on
Lab-1 (`ubuntu@172.26.132.144`) against an isolated MAVER image. The Gemini API key
was stored separately on Lab-1 and was never included in the uploaded source or
local evidence. The agent used Google's AI Studio Gemini API through LiteLLM's
native `gemini/gemini-3.5-flash-lite` route. This is a hosted diagnostic condition:
incident telemetry left the cluster. It is not a local-model thesis arm.

## First condition: quota failure after remediation

The initial profile set temperature 0.2 and had no request pacing. Its complete
source archive was SHA-256
`eea4c039adea092d9c3432c8308854d232eeb408553cb0e8bb2c2dfbe9de3270`;
the image ID was
`sha256:adf60d9acc74d622316fc2395bda46cfb0b8a49d93aa79c7a7e03b6be0d2b477`.
Incident `d962d1dc-e384-44de-b401-3b983bb1cf0b` started at 13:38:01 UTC.
RCA, Planning and Execution submitted artifacts; Execution made two approved write
calls. Verify then encountered Google's 429 free-tier request limit (15 requests
per minute), and the agent ended `invalidated` / `failed` without a Verify artifact.
The harness nevertheless graded the repaired workload successful. That external
grade does not establish successful agent completion. Its audit export contains 106
contiguous events with complete file integrity and observed-call coverage.

The first record remains at
`/home/ubuntu/maver-gemini-20260921/` on Lab-1, including
`workspace/export-first/`, `results/`, `console.log` and `run.exit`. It was not
replaced by the second condition.

## Corrected condition: completed recovery

Google recommends leaving temperature, top-p and top-k at Gemini 3 defaults. The
corrected profile omits these parameters in both the main and auxiliary sampling
sets. A profile-specific 5,000 ms minimum interval serializes provider requests
within the agent process. Other profiles keep their existing behavior. The profile,
client and profile-schema file hashes are respectively
`ed82ab03891825780baddb4e2a9c55c12a6559d15fc1b49d92ab03c410c894f6`,
`7cb5950de7b989cbd83c92c9bbe94f05b94a4e9cc19fecbf660ea36f35d0fe30`,
and `6e56a414cb323039449f8b1961ca807a9d752e2fb2e6cf3f06a376d2a39e2ad4`.
The update archive was SHA-256
`4a027e5701931e5706dd0d8c5d1eaecbb18a254f416c9154a5514a0f1cfbbd2f`;
image ID was
`sha256:316de75c4d4106cd9f0b658d9a2f076c15f4208b9839148dc767b1fc6d0097c8`.

The separate incident `c88e2616-0f93-4daa-aa70-ef0a05a8d189` ran from
13:53:49 to 13:56:39 UTC. Execution patched `Service/user-service` targetPort
from 9999 to 9090 and completed a restart of `Deployment/user-service`. Verify
reported `recovered` after repeat namespace and entry-point checks. MAVER ended
`tentative_success` / `completed` with all four structured artifacts and no error.
AIOpsLab independently graded mitigation successful (TTM 160.1 s); this is one
validation case, not an effectiveness estimate. The 26 observed provider attempts
all succeeded. Model-call log gaps had a median of 5.0 s (minimum 4.89 s at the
logging point); there was no 429 or overlapping provider call.

The offline audit export, with the bridge log supplied, passed: 153 contiguous
MAVER events, 23 paired tool invocations, 26 paired bridge actions / 52 bridge
artifacts, no unmatched invocation IDs, no open calls, no missing sequence ranges,
and complete evidence integrity and observed-call capture coverage. Database
projection coverage remains `unknown`, as documented by the audit contract.
The full private evidence remains on Lab-1 at
`/home/ubuntu/maver-gemini-defaults-20260921/`: `workspace/export-final/`,
`results/bridge.log`, `results/bridge.log.artifacts/`,
`results/maver-submission.json`, `console.log` and `run.exit`.

Both isolated test containers were stopped. The original prototype container
`k8s-agent-aiopslab` retained ID `35a7fa77099204136d11e0dedca47e1ce08e24014b88ba5627af4fb5bbc1b308`
and its original start time, and answered `/healthz` with HTTP 200. The Lab
benchmark's own teardown completed. The Gemini key file stays on Lab-1 at
`~/.config/maver/gemini.env` with mode 0600; it is not committed here.

Local validation of the final code: `make check` passed Ruff, mypy and
2,078 tests (20 skipped, 45 integration cases deselected). The request-pacing
regression checks serialization, and the profile test checks Google's defaults
for both sampling sets. AIOpsLab records one grade per condition; neither record
is substituted into a paper campaign or historical parity ledger.

Google guidance: [Gemini 3.x defaults](https://ai.google.dev/gemini-api/docs/whats-new-gemini-3.5),
[Gemini API rate limits](https://ai.google.dev/gemini-api/docs/rate-limits).
