# The AIOpsLab adapter

MAVER runs its own pipeline against [AIOpsLab](https://github.com/microsoft/AIOpsLab)
without putting AIOpsLab's admin kubeconfig inside the agent container.

## Two processes, opposite directions

AIOpsLab is the inverse of a harness that spawns the agent. The **driver** imports
AIOpsLab's orchestrator and drives it in the harness's own Python environment, and the
orchestrator pulls exactly one action back out of it — an action the driver produces by
posting a blinded incident to the **agent** over HTTP and waiting for the whole pipeline
to finish.

| | Driver (`src/maver/adapters/aiopslab/host/`) | Agent (the container) |
|---|---|---|
| Runs in | AIOpsLab's Poetry environment, loaded by path | its own image |
| Holds | the cluster-admin kubeconfig, the problem id, the bridge | a bearer token and nothing else |
| Imports | the standard library only, plus `aiopslab` | MAVER |
| Reaches the cluster by | `kubectl`, directly | one `POST /v1/action` to the bridge |

The driver half's standard-library-only rule is enforced by an AST test. It is not
tidiness: an import of `maver` there crashes on a benchmark host part-way through a sweep,
where nothing in this repository is installed.

## Ownership boundary

AIOpsLab keeps owning application deployment, workload generation, fault injection,
grading, recovery and cleanup. The adapter submits a blinded incident and maps the
agent's structured result back onto AIOpsLab's task contract:

| AIOpsLab task | Agent stages | Submitted value |
|---|---|---|
| detection | diagnosis only | `"Yes"` or `"No"` |
| localization | diagnosis only | exact component-name list |
| analysis | diagnosis only | `system_level` / `fault_type` taxonomy |
| mitigation | diagnosis → planning → execution → optional verification | `submit()` after the pipeline |

Read-only tasks stop after diagnosis, so the agent cannot remediate a problem before
AIOpsLab grades the diagnosis. The bridge derives its own write authority from the task
type before the episode starts, so a detection episode is refused a write whatever the
request claims about itself.

The bridge serves exactly five things: active-namespace `kubectl`, Prometheus instant
queries and firing alerts, namespace-scoped Jaeger, an execution-only MongoDB access
repair whose service and script are both allowlisted, and a verification-only
`probe_service` that port-forwards one named Service and makes one HTTP request through
it. Namespace-less reads are scoped to the active namespace, writes must name it, and
generic `kubectl exec` is rejected because an in-container command cannot be proven
read-only. This bridge refusal is expected transport enforcement; the general
approval gate may approve an in-container command whose effects it cannot classify,
while the bridge still refuses it.

The adapter replaces the role's kubectl prompt block and tool description with the
same transport guidance. RCA, Planning and Verify receive read-only instructions;
Execution is explicitly allowed to apply planned, scoped mutations through the
runtime's authorization path. Generic `kubectl exec`, shell command chains and
kubectl through local sandbox tools remain unsupported. The dedicated MongoDB
recovery tool is still the only supported in-container repair for its allowlisted
faults. Execution's read-before-write behavior and approval policy are unchanged.

## Start the agent

```bash
make image                                            # builds maver:local
MAVER_MODEL_ENDPOINT=http://vllm.internal:8000/v1 \
  sudo -E scripts/aiopslab/deploy-agent.sh
```

The AIOpsLab adapter needs no packaging extra: its transport speaks HTTP to the bridge
with the client library the core already depends on. `make image-eval` is SREGym's.

The script generates `/var/lib/aiopslab/agent-config/bridge-token` (32 random bytes, mode
0600) if it is absent, writes the environment file beside it, and starts the container on
the host network with no kubeconfig, a read-only root filesystem, and the token mounted
read-only at `/run/secrets/aiopslab-bridge-token`. It waits for `/healthz` and then
`/readyz`, and reports which of the two failed.

| Variable | Default | What it selects |
|---|---|---|
| `MAVER_MODEL_ENDPOINT` | *required* | the private model endpoint |
| `AIOPSLAB_ARM` | `ma-v` | an arm file in `configs/arms/`, baked into the image |
| `AIOPSLAB_MODEL_PROFILE` | `qwen3.6-35b-a3b` | a profile in `configs/models/` |
| `AIOPSLAB_AGENT_IMAGE` | `maver:local` | the image `make image` tags |
| `AIOPSLAB_AGENT_DATA_DIR` | `/var/lib/aiopslab/agent-workspace` | the workspace volume |
| `AIOPSLAB_AGENT_CONFIG_DIR` | `/var/lib/aiopslab/agent-config` | the token and environment file |

`ma-v` enables verification with a shared remediation budget of two, verification
blind to the diagnosis, no health gate, one inconclusive retry, and memory off.
With no explicit Verify cap, its hand-backs share that limit of two. A different arm
is a different experiment; deploy it into a fresh `AIOPSLAB_AGENT_DATA_DIR` so the
two do not share a workspace.

What the script writes into the environment file, which is also what a hand-written
`docker run` needs:

| Name | Value | Why |
|---|---|---|
| `MAVER_INTAKE__ADAPTERS` | `aiopslab` | binds the adapter for the run |
| `MAVER_ADAPTERS__AIOPSLAB__ENABLED` | `true` | makes the bridge credential mandatory rather than optional |
| `MAVER_ADAPTERS__AIOPSLAB__BRIDGE_URL` | `http://127.0.0.1:1819` | where the driver serves the bridge |
| `MAVER_ADAPTERS__AIOPSLAB__BRIDGE_TOKEN_FILE` | `/run/secrets/aiopslab-bridge-token` | a file, so the token is not in `docker inspect` |
| `MAVER_ADAPTERS__AIOPSLAB__BRIDGE_TIMEOUT_S` | `180` | one bridge action's bound; the shipped default is 120 |
| `MAVER_AUTHORITY__MODE` | `auto` | nobody is waiting to approve a write during a sweep |
| `MAVER_SERVER__ENVIRONMENT` | `local` | `production` plus `auto` is refused at startup, and a benchmark host is not production |
| `MAVER_SERVER__HOST` | `127.0.0.1` | with `--network host` there is no published-port boundary, and the API has no authentication |

`MAVER_ADAPTERS__AIOPSLAB__BRIDGE_TOKEN` sets the token inline instead of by file. The
prototype's own names — `AIOPSLAB_BRIDGE_URL`, `AIOPSLAB_BRIDGE_TOKEN`,
`AIOPSLAB_BRIDGE_TOKEN_FILE`, `AIOPSLAB_BRIDGE_TIMEOUT_SECONDS` — still work as legacy
aliases, and each one that decides a value says so in the log and in the run's
configuration snapshot.

Do not carry the prototype's `agent.env` across wholesale. It set
`RCA_AUTO_SNAPSHOT_ENABLED=false` under a name the prototype never read, so every archived
AIOpsLab run had the RCA configuration snapshot on; MAVER honours that name as an alias,
which would switch the snapshot off and make a new sweep incomparable with the archive.
The deploy script names `MAVER_PIPELINE__AUTO_SNAPSHOT=true` for that reason.

## Complete structured reads

Successful Kubernetes and structured telemetry responses remain complete inside the
transport. Snapshot parsing and telemetry namespace filtering run before model-facing
tools apply their 16,384-character display limit. Full scoped results are saved to the
session workspace; `save_as` also retains the complete kubectl output. Direct
unstructured bridge actions and error messages retain their 10,000-character bound.

## Verify hand-back cap and current behavior

The agent accepts `MAVER_VERIFY__HANDBACK_RETRIES=1` to permit at most one Verify
return to RCA or Planning, still within the shared remediation budget. Zero permits
none; unset inherits the shared limit. The prototype's `VERIFY_HANDBACK_RETRIES`
spelling is not an alias. Inconclusive checks and Execution replans retain their
existing accounting.

The shipped deployment script writes a fixed environment file and does not forward
arbitrary host variables. To configure this cap with that script, add the setting
to its environment-generation block when adapting it locally. For your own
`docker run`, put it in the container's `--env-file` and recreate the container.
Editing a file alone does not change a running container's environment.

The current Execution prompt confirms applied changes at the resource level;
Verify judges recovery. Verify can fill a missing recovery entry-point field from
this adapter's successful `probe_service` result, and missing hand-back fields from
its submitted evidence. Those repairs preserve the verdict and raw transcript and
were introduced in audit schema version 3; the current audit version is 4. See [the migration record](../../CHANGELOG.md)
and [configuration guide](../customisation.md#bound-verify-hand-backs). New behavior
defines a new experimental condition; a historical arm name does not restore older
prompt or control behavior.

## Run one problem

From the pinned AIOpsLab checkout, in its Poetry environment. `PYTHONPATH` is
deliberately not set: the entry point loads the driver from its files rather than
importing it by package name, so this project does not have to be installed there.

```bash
cd /var/lib/aiopslab/src/AIOpsLab
poetry run python /path/to/maver/scripts/aiopslab-driver.py run \
  --problem noop_detection_hotel_reservation-1 \
  --bridge-token-file /var/lib/aiopslab/agent-config/bridge-token \
  --results-dir /var/lib/aiopslab/results/maver/noop_detection_hotel_reservation-1
```

It prints the AIOpsLab session, the agent's incident id, the submitted action and the
harness's outcome, and exits 1 on any refusal the sweep should record.

Check the host before a run:

```bash
poetry run python /path/to/maver/scripts/aiopslab-driver.py doctor \
  --bridge-token-file /var/lib/aiopslab/agent-config/bridge-token
```

`doctor` reports four preconditions separately: the agent answers, `aiopslab` is
importable in this interpreter, `kubectl` is on PATH, and the token resolves.

## Run a sweep

```bash
poetry run python /path/to/maver/scripts/aiopslab-driver.py sweep \
  --sweep-dir /var/lib/aiopslab/results/sweeps/my-sweep \
  --bridge-token-file /var/lib/aiopslab/agent-config/bridge-token
```

With no `--problems` file it reads AIOpsLab's registry, runs every Kubernetes-backed
problem once, and records the Docker-backed ones as excluded. `--task-types mitigation`
narrows it to one subset; `--problems <file>` takes an explicit list, one id per line,
`#` starting a comment.

**Reusing the same `--sweep-dir` resumes**: a problem AIOpsLab has already graded is
skipped, so an interrupted sweep continues rather than repeating. `--rerun` overrides
that, and it is a new experiment rather than a repair.

The sweep applies these retry rules:

- **A result AIOpsLab graded is final.** It is never run again — not on a resume, and not
  when the runner exits nonzero cleaning up after the grade was persisted. Retrying a
  graded failure would replace one cell with a second sample of the model, which is a
  better score manufactured out of variance.
- **Only a non-grade is retried**, at most `--max-attempts` times (default 2), with
  `--retry-delay` seconds between attempts. That is a deployment failure, a host timeout,
  or a crash.

Each problem is bounded by `--problem-timeout` (default 7,800 s, covering its deployment
and grading) and the agent's own incident by `--agent-timeout` (default 7,200 s). A
problem stopped by the host bound records exit code 124.

Detached, with a watcher on the side:

```bash
tmux new-session -d -s aiopslab-sweep "cd /var/lib/aiopslab/src/AIOpsLab && \
  poetry run python /path/to/maver/scripts/aiopslab-driver.py sweep \
  --sweep-dir /var/lib/aiopslab/results/sweeps/my-sweep \
  --bridge-token-file /var/lib/aiopslab/agent-config/bridge-token 2>&1 | \
  tee -a /var/lib/aiopslab/results/sweeps/my-sweep/console.log"

scripts/aiopslab/watch-sweep.sh /var/lib/aiopslab/results/sweeps/my-sweep
```

The watcher only watches. It prints the grade tally, the last rows of `status.tsv` and
the tail of `sweep.log`, and exits when the sweep records `completed_utc`. It does not
kill a slow problem — `--problem-timeout` is that boundary — and it does not restart a
dead sweep, because restarting one is the resume command above.

One sweep at a time per host, enforced by a lock on `<sweep-root>/.sweep.lock`. AIOpsLab
mutates cluster-wide state, so two concurrent sweeps produce two spoiled arms rather than
two arms.

## Where the records land

```
<sweep-dir>/
  metadata.env                    what this sweep set out to do, which build answered,
                                  and completed_utc once it has
  problem-list.txt                the problems selected, with their task types
  excluded-docker-problems.txt    the Docker-backed problems, named rather than dropped
  status.tsv                      one row per problem
  sweep.log                       the sweep's own log
  results/<problem>/
    <AIOpsLab's own result JSON>  the graded outcome; the sweep reads it and nothing else
    maver-submission.json          what was asked of the agent, and what was submitted
    bridge.log                    one JSON line per cluster action, from the host side
```

`status.tsv` records these columns:

| Column | |
|---|---|
| `index` | position in the selected list |
| `problem` | AIOpsLab problem id |
| `task_type` | the graded task (new) |
| `attempts` | how many runs it took (new) |
| `exit_code` | 0, 1, or 124 for the host bound |
| `grade` | `pass`, `fail`, `unknown`, or `error` when the harness wrote nothing |
| `agent_incident_id` | the incident that produced the answer (new) |
| `started_utc`, `ended_utc` | wall clock |
| `duration_seconds` | (new) |
| `result_file` | the AIOpsLab result file the grade was read from |

`bridge.log` is the credential-holding process's independent account of each action.
Bridge log schema 2 writes a `received` record before dispatch and a `finished` record
on completion, refusal or failure, including client disconnection. A bridge receipt ID
links the pair; session, request, invocation, model tool-call and run IDs are retained
when supplied. Duration, HTTP status, decision and bounded previews remain in JSONL.
Full parsed request contracts and response bodies are stored by SHA-256 under
`bridge.log.artifacts`, with private directory/file permissions (0700/0600). Headers
and bridge credentials are excluded. Older logs contain a single row per action;
when counting schema-2 actions, count `finished` rows. A received-only action remains
explicitly unfinished. See [audit evidence](../audit.md) for export and coverage limits.

`maver-submission.json` holds the alert as posted, the terminal incident record, the
encoded answer, and — for the analysis task — the answer's provenance: `source` is
`model`, `normalized` or `fallback`, and `rule` names the hand-written rule when one
overrode the model's classification. **An analysis accuracy reported without that split
is the rules' accuracy reported as the agent's.** The same caution applies more strongly
to the detection consistency guard and, weakly, to the localization suffix stripping.

## Scope and research notes

- The problem id is retained only by AIOpsLab, for setup and grading. It never enters the
  alert or any prompt. The one thing that does cross is the moment fault injection began,
  so the agent can separate deployment noise from the observation window.
- AIOpsLab sees one outer adapter action (`submit(...)`); internal multi-agent steps and
  tool calls are audited in this agent's session store. Consequently, AIOpsLab's `steps`
  and token counters are not directly comparable to its native single-LLM clients. Use
  the agent-run ledger for process-cost comparisons.
- Docker-backed Flower problems are intentionally rejected. They fall outside this
  Kubernetes agent's tool/RBAC/HITL model; silently granting Docker socket or host shell
  access would be an unsafe and thesis-confounding extension.
- AIOpsLab deploys no Loki, so the log tools are withdrawn from every role before the run
  and the reason is recorded. Logs stay reachable through scoped `kubectl logs`, which is
  what the role prompts already say.
- MongoDB access recovery is a narrow allowlisted action, not arbitrary exec, and it still
  passes the approval gate. `probe_service` is available to verification only, and cannot
  run an in-container command or mutate anything.
- What verification is told to probe comes from the adapter's own configuration
  (`connectivity_contracts`, keyed by namespace), not from the incident: a reliable
  user-facing Service is deployment metadata, and verification stays blind to the causal
  answer.
- The embedded store is fine for a smoke run. Use the PostgreSQL backend for a campaign
  whose retrieval state has to outlive a container.
