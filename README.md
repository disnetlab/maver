# MAVER

**M**ulti-**A**gent **V**erification and **E**xperience-informed **R**ecovery: a
memory-augmented multi-agent framework for diagnosing and repairing live microservice
incidents. MAVER is designed to be cloned, configured and adapted to a deployment;
its default model profile targets a privately served open-weight language model.

An alert enters over HTTP or optional Kafka intake. A diagnosing role investigates
cluster state, metrics, logs and traces and emits a typed `RootCause`; its ordinary
tools are read-only, while `kubectl exec` is a separate action that requires human
approval by default because it can mutate application state. A planning role turns
the diagnosis into a typed `Plan`. An execution role applies it under a check-and-set
discipline and a write-authority gate. Two optional mechanisms sit around that spine:
an independent verification role that can hand work back, and a gated second
diagnostic opinion informed by an experience store of past incidents.

The intended deployment serves the model on your own infrastructure. Hosted model
profiles and optional OTLP export are also supported; the selected endpoints and
network policy determine where data goes.

## Quick start

Python 3.12+ and [uv](https://docs.astral.sh/uv/) are required. Install dependencies
and validate the checkout without contacting a model or cluster:

```bash
uv sync --all-extras
make check                       # lint, types, unit and contract tests
uv run maver config check        # validate local configuration
```

For an incident run, configure an accessible model endpoint and a kubeconfig for the
managed cluster. The shipped model endpoint is a placeholder; select your own:

```bash
export MAVER_MODEL__ENDPOINT_OVERRIDE=http://your-model-host:8000/v1
export KUBECONFIG=/path/to/kubeconfig
uv run maver serve               # listen on 0.0.0.0:8080
```

From another terminal, submit a diagnosis-only incident:

```bash
uv run python examples/run_incident.py examples/alerts/high-error-rate.json \
  --target-stage diagnose
```

Model profiles in `configs/models/` specify provider routing and sampling. Configure
telemetry and write authority for your deployment using
[the environment example](deploy/docker/maver.env.example). The
[operations guide](docs/operations.md) covers incident submission, approvals,
durable retries, quarantine, restart recovery and outcome feedback.

## What this artifact demonstrates

**D** — demonstrable here, by the command shown, on a laptop.
**I** — instrumented here: the mechanism and the measurement apparatus are in this
repository, but reproducing a number needs the benchmark, a served model and tens of
hours.
**A** — archived: the number comes from retained measurement data and is recomputed in
the data archive, not here.

| Claim | Kind | Command |
|---|---|---|
| Roles exchange typed artifacts, never transcripts, and one incident runs end to end under either executor | D | `uv run pytest tests/unit/test_runner.py` |
| Only execution writes through classified Kubernetes mutations: a stage-aware gate refuses them from other roles, and the kubectl tool refuses them again on its own; diagnostic roles may use separately approved `kubectl exec` | D | `uv run pytest tests/unit/test_write_gate.py tests/unit/test_kubernetes_tool.py` |
| A stage-aware gate classifies read, write, exec and blocked, and refuses a write outside the stages that may issue one | D | `uv run pytest tests/unit/test_write_gate.py tests/unit/test_kubernetes_tool.py` |
| Any write can be held for a person, and `kubectl exec` requires one by default even under automatic write authority: the run parks durably, and a decision resumes the same incident from the role that was interrupted | D | `uv run pytest tests/unit/test_approval_pause.py` |
| Execution opens on a forced read, and a submission with no command behind it is refused for the submit guard's budget of two rejections; a third is accepted, recorded as a bypass on the run row and in the event log | D | `uv run pytest tests/unit/test_roles_execution.py tests/unit/test_loop.py` |
| Verification returns to planning or diagnosis within the shared remediation budget and an optional tighter Verify cap | D | `uv run pytest tests/unit/test_routing.py tests/unit/test_roles_verify.py` |
| Records carry lifecycle states, and retrieval demotes an invalidated record without hiding it | D | `uv run pytest tests/unit/test_lifecycle.py` |
| Distillation writes bounded evidence and lesson projections; a raw transcript never enters retrieval | D | `uv run pytest tests/unit/test_projections.py tests/unit/test_distil.py` |
| The gated second opinion: a retrieval-free first diagnosis, a bounded gate that sees only the candidate, precedents retrieved by a harness-built query, a blind second diagnosis, a label-shuffled selector, and a fall back to the first answer on any failure | D | `uv run pytest tests/unit/test_secondpass.py tests/unit/test_secondpass_wiring.py` |
| Four runtime controls: a generation cap, bounded nudges, a repeat breaker, one finalizer rescue | D | `uv run pytest tests/unit/test_loop.py` |
| An ordered per-incident log records every model call with its token counts and the sampling set it actually ran on, every tool call, every submission outcome and every intervention — including the distilling pass, which is a model call like any other | D | `uv run pytest tests/unit/test_store_sink.py tests/unit/test_observability.py tests/unit/test_learning_job.py` |
| Each experiment arm is a single-variable change over a shared base; naming one configures the run, the environment still outranks it, and an environment that contradicts what the arm holds constant fails at startup | D | `uv run pytest tests/unit/test_profiles_and_arms.py tests/unit/test_settings.py` |
| An environment's own fault injection is filtered out of every result before the model sees it, not only out of the ones a tool remembered to filter | D | `uv run pytest tests/unit/test_evidence_reaches_every_tool.py tests/unit/test_sregym_adapter.py` |
| Private model endpoints and a deployment network policy support keeping telemetry on the operator's infrastructure | D | `configs/models/*.yaml` names the endpoint; `deploy/k8s/base/networkpolicy.yaml` is the egress boundary |
| The reported recovery rates, the effect of verification, the denser-model comparison, and the throughput figures | A | the data archive that accompanies the manuscript |
| An archived row can be re-measured end to end using its recorded source revision and configuration | I | the retained inputs, SREGym adapter, served model, benchmark cluster and accompanying data archive |

One row deserves its own sentence, because it is the most easily overstated. **In every
measured campaign the approval gate ran in automatic mode, so no request ever reached a
person.** What is demonstrated above is that the mechanism works — it parks, persists,
and resumes the same incident on a decision, and that path is tested. It is not a claim
that a human approved anything during the measurements.

**The manuscript's results come from the research prototype.** They are recomputed
from its data archive. This repository provides the reusable framework, mechanism
tests and benchmark adapters. Its current defaults include later behavior changes,
so the manuscript's measurements should not be attributed to the current release.

`make test-parity` checks retained inputs and mechanisms. [The changelog](CHANGELOG.md)
identifies changes to prompts, submission handling and other experimental conditions.
Reproducing a study requires its recorded source revision, configuration and data.

## Running it

```
deploy/docker/    the image, and compose files for local and evaluation use
deploy/k8s/       manifests: base plus a local and a production overlay
```

The image is non-root, runs on a read-only root filesystem, and carries `kubectl`
because the default transport executes it. RBAC is derived from the verbs the tool
layer can actually issue: one cluster-wide read-only role, and namespaced read and
write roles an overlay binds per managed namespace. Secrets are in neither role.

**The approval and incident routes are unauthenticated.** Restrict them with the
shipped NetworkPolicy and do not expose them through an ingress.

See [incident operations](docs/operations.md) for durable intake, operator review,
and recovery, and [Kubernetes deployment](deploy/k8s/README.md) for manifests,
namespace access and persistent state.

## Configuration

The package and command are now `maver`, and settings use the `MAVER_` prefix.
Existing `KIRA_` setting names remain aliases; `MAVER_` wins when both are set.
The PostgreSQL schema still defaults to `kira_memory` so the rename opens existing
data. Narrow compatibility imports restore previously scheduled approval and learning
jobs into the MAVER runtime. Compose keeps the old physical workspace-volume names by default; set
`MAVER_WORKSPACE_VOLUME` or `MAVER_EVAL_WORKSPACE_VOLUME` when the previous deployment
used a custom Compose project name. Kubernetes adopters should retain their existing
namespace and PVC bindings in their overlay when upgrading. The repository is now
[disnetlab/maver](https://github.com/disnetlab/maver); update an existing clone with
`git remote set-url origin git@github.com:disnetlab/maver.git`.
Frozen prototype assets retain their original bytes.

Every setting is a field on a nested model, and an unrecognised `MAVER_`-prefixed name
stops the process rather than being ignored. `uv run maver config show` prints the
resolved configuration with secrets masked; `configs/arms/` holds the experiment
configurations, each declaring what it holds constant.

| Setting | Default | Purpose |
|---|---|---|
| `MAVER_INTAKE__HTTP_ENABLED` | `true` | Accept HTTP incident submissions. Read and operator routes remain available when disabled. |
| `MAVER_INTAKE__KAFKA__ENABLED` | `false` | Consume Kafka alerts; requires the Kafka extra and broker configuration. |
| `MAVER_SERVER__RESUME_INTERRUPTED_INCIDENTS` | `false` | Checkpoint and recover eligible interrupted stages; park uncertain effects for reconciliation. |

HTTP callers can use `Idempotency-Key` for durable retries. Kafka uses a durable
message identity automatically. Deduplication and restart recovery are separate:
returning an existing receipt never starts another execution. Configuration examples,
quarantine commands, recovery boundaries and API behavior are in
[incident operations](docs/operations.md).

## Customising your deployment

Clone and adapt the repository for your environment. Python agent subclasses define
supported behaviour; YAML configures complete named agents and binds them to
Python-defined workflows. Set `MAVER_CUSTOMISATION` to your YAML file, and register
local implementations in `src/maver/custom.py`.

Agents can inherit model, prompt, tool, middleware, memory and execution settings,
with typed schemas rejecting unsupported options. Each invocation has its own
conversation and private state inside one shared pipeline-run state. Middleware can
add explicitly scoped prompt context without exposing run metadata automatically.

A request selects a configured pipeline, optional selector middleware can choose when
no explicit name is supplied, and the deployment supplies a default. Approval resumes
retain the selected composition. Customisation requires no separately packaged plugin.

Verify returns can be bounded with `MAVER_VERIFY__HANDBACK_RETRIES`; individual
`multi` pipelines can tighten that cap through `config.verify_handback_retries`.
The shared remediation budget still applies, and approval resumes retain the count.
[examples/verify-handbacks.yaml](examples/verify-handbacks.yaml) configures two built-in
recovery pipelines without requiring custom Python registrations.

See [the customisation guide](docs/customisation.md), or run the complete offline
example with `uv run python examples/customisation.py`. It exercises agent inheritance,
YAML binding, a custom two-stage workflow, middleware and isolated conversations.

## Audit evidence

The service mirrors full audit events and referenced evidence under
`<workdir>/audit`, separately from incident workspaces. The database remains a
best-effort query projection. Export an incident with
`uv run maver audit export INCIDENT_ID --output ./incident-audit`; add
`--bridge-log /path/to/bridge.log` to include AIOpsLab host evidence.

The export reports file integrity and capture coverage separately. Unavailable
provider details, uninstrumented tool fields and unfinished calls are limitations,
not evidence of zero work. See [audit evidence](docs/audit.md) for capture boundaries,
privacy and historical records.

## Layout

```
src/maver/
  contracts/     the typed artifacts, incident state and audit events
  settings/      nested settings, model profiles, arm files
  llm/           the provider client, with sampling pinned and asserted
  agents/        inheritable Python implementations and validated named definitions
  pipelines/     Python workflow definitions, YAML bindings and admission selection
  roles/         the bounded model/tool loop and the existing role implementations
  custom.py      local registration of agents, middleware, workflows and selectors
  configuration.py  the optional MAVER_CUSTOMISATION YAML loader
  middleware/    the hook protocol, and the write gate and approval authority
  tools/         the registry, the capability modules, and the transports
  memory/        ports, ranking, lifecycle, projections, backends
  secondpass/    the gated second opinion, as five composable stages
  orchestration/ routing, the graph, the runner, outcomes, and the offline pass
  adapters/      the environment seam, SREGym and AIOpsLab adapters
  api/           intake, health, approvals, feedback and reconciliation

configs/         model profiles, experiment arms, learning trigger settings
deploy/          the container image, and Kubernetes manifests with two overlays
examples/        alert payloads, HTTP client, and an offline customisation example
scripts/         what an evaluation host runs outside this package
tests/           unit, contract and integration suites
```

The scripts are worth naming because nothing in the package imports them.
`sregym-driver.py` is the path the benchmark's own agent configuration invokes; it loads
the driver from its file rather than by package name, because it runs in the harness's
interpreter where this project is not installed. `sregym-mcp-port-forward.sh` holds open
the port-forward that adapter's transport speaks through. Both are load-bearing and both
fail on a remote host part-way through a run, so the contract between the entry point and
the driver it loads is pinned by a test. `aiopslab-driver.py` is the same arrangement for
the other adapter, and `scripts/aiopslab/` holds the two things its host runs by hand.

Under SREGym, kubectl reaches the cluster through the harness's own filtering proxy: the
sandbox shell runs a real kubectl against a credential-free endpoint the harness serves,
which is the route every measured run took. The MCP kubectl tools are hidden by default,
so the model reaches the cluster from the shell and telemetry still arrives over MCP.
The proxy shell cannot pause for approval, so it refuses `kubectl exec` by default and
directs the role to `exec_kubectl_cmd_safely`; that structured tool is also hidden by
default. Set `MAVER_ADAPTERS__SREGYM__MCP_KUBECTL_TOOLS=true` to expose it for
human-approved exec. Despite the setting's historical name, the tool uses the configured
proxy transport. Reproducing historical unattended shell exec requires the explicit
authority opt-out in
[examples/sregym-unattended.yaml](examples/sregym-unattended.yaml); selecting the adapter
does not enable it.
[The SREGym guide](docs/adapters/sregym.md) covers that route's settings, loading an archived corpus with
`maver memory import`, and the leave-one-out map an imported corpus needs.

Two environment adapters ship, and the difference between them is instructive. Under
SREGym the harness starts the agent and hands it a problem. Under AIOpsLab the direction
reverses: a driver runs inside the benchmark's own interpreter, holds the cluster-admin
kubeconfig that the agent container is deliberately not given, and lets the agent reach
the cluster only through a token-scoped loopback bridge that re-screens every action.
[The AIOpsLab guide](docs/adapters/aiopslab.md) covers deploying the agent,
running one problem and running a resumable sweep.

`STYLE_GUIDE.md` carries the conventions a linter cannot express: the layer boundaries,
the tool-allowlist rule, the write-gate invariant, the artifact-schema freeze, and the
rule that keeps the graph library replaceable.

See [CONTRIBUTING.md](CONTRIBUTING.md) for local checks and the protected
branch-and-pull-request workflow.

## Limitations and known issues

The manuscript's numbers remain prototype results. Mechanism tests demonstrate
implementation behavior; they do not establish comparative recovery rates.

**Deployment boundaries.** Account for these constraints when adapting the framework.

- **No authentication or authorisation on the API.** Anyone who can reach the port can
  approve a cluster write. The shipped deployment restricts access with a NetworkPolicy
  and no ingress, and a decision's `decided_by` is recorded as a claim rather than a
  verified identity.
- **Single tenant.** `memory.corpus_id` partitions the retrieval corpus, but nothing
  authenticates the caller who chose it. It is a research partition, not a security
  boundary; the supported deployment is one MAVER per tenant.
- **No rate limiting** beyond the intake concurrency bound. HTTP refuses excess alerts
  with 429; Kafka leaves them on the broker until capacity returns. The turn budgets,
  the generation cap and the repeat breaker bound what one incident can consume, not
  what a client can ask for.
- **Single process. No scale-out and no leader election.** The execution lock is
  in-process, so `replicas: 1`, the `Recreate` strategy and a ReadWriteOnce volume are
  load-bearing together rather than defaults.
- **No backup or restore automation.** The state is one volume with a documented layout
  and one namespaced PostgreSQL schema, so both are addressable by hand. What is missing
  is a schedule and a tested restore.
- **No prompt-injection hardening.** Telemetry the agent reads is an untrusted channel,
  and the paper scopes this out by name. What contains an injected instruction is
  structural — per-role allowlists, namespace-scoped RBAC, the write gate — and not
  anything in the prompt.

**Behaviour a reader should not be surprised by.**

- An execution submission with no command behind it is refused for the submit guard's
  budget of two rejections and the third is accepted, recorded as a bypass on the run row
  and in the event log. That row is in the table above; the bypass is deliberate, so that
  a guard cannot end a run by refusing it forever.
- The host-recovery tool is offered to the remediating role whether or not recovery
  nodes are configured, and the node allowlist is empty by default, so a deployment that
  configures none is offering a tool that refuses every call. It is offered anyway
  because every measured campaign offered it on an empty allowlist, and what a role is
  shown is an input to the result; the startup log says so.
- Retrieval is one retriever across both backends by construction, and the normalisation
  both index and query with is a documented approximation of PostgreSQL's text parser,
  not a copy of it. `src/maver/memory/lexemes.py` states what it reproduces and what it
  does not, measured against a live server.
- The store layout is versioned. A build that opens a store written by a later build
  refuses it and names both versions rather than writing past columns it does not know.
- Eligible delayed Learning persists its intent and external feedback before
  scheduling. Startup restores pending work, and failed passes retry serially with
  backoff, including when operator feedback changes during distillation. Recovery
  does not backfill diagnosis-only runs or take over externally graded episodes.
  See [incident operations](docs/operations.md#delayed-learning-and-retries).

**Evaluation posture.** The approval gate ran in automatic mode in every measured
campaign. The paragraph under the claims table above says what that does and does not
imply.

## Citing

See [CITATION.cff](CITATION.cff).

## Licence

MIT. See `LICENSE`.
