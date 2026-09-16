# MAVER

**M**ulti-**A**gent **V**erification and **E**xperience-informed **R**ecovery: a memory-augmented multi-agent
recovery framework whose default pipeline diagnoses and repairs live microservice
incidents using a privately served open-weight language model.

An alert enters over HTTP. A read-only diagnosing role investigates cluster state,
metrics, logs and traces and emits a typed `RootCause`. A planning role turns that
record into a typed `Plan`. An execution role, the only one holding write-capable
tools, applies it under a check-and-set discipline. Two optional mechanisms sit around
that spine: an independent verification role that can hand work back, and a gated
second diagnostic opinion informed by an experience store of past incidents.

No telemetry leaves the cluster. The model runs on your own infrastructure.

## Quick start

```bash
uv sync --all-extras
make check                       # lint, types, and the whole test suite
make test-parity                 # only the checks that pin the published numbers
uv run maver config check         # validate the configuration a server would start on
uv run maver serve                # listen on 0.0.0.0:8080
python examples/run_incident.py examples/alerts/high-error-rate.json
```

Nothing above needs a cluster or a model endpoint except the last line, which needs
both. `configs/models/` holds the model profiles; a profile is the complete statement
of how one model is called, including the sampling set every published measurement was
taken under.

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
| Only execution writes: a stage-aware gate refuses a write from any other role, and the kubectl tool refuses it again on its own | D | `uv run pytest tests/unit/test_write_gate.py tests/unit/test_kubernetes_tool.py` |
| A stage-aware gate classifies read, write, exec and blocked, and refuses a write outside the stages that may issue one | D | `uv run pytest tests/unit/test_write_gate.py tests/unit/test_kubernetes_tool.py` |
| Any write can be held for a person: the run parks durably, and a decision resumes the same incident from the role that was interrupted | D | `uv run pytest tests/unit/test_approval_pause.py` |
| Execution opens on a forced read, and a submission with no command behind it is refused for the submit guard's budget of two rejections; a third is accepted, recorded as a bypass on the run row and in the event log | D | `uv run pytest tests/unit/test_roles_execution.py tests/unit/test_loop.py` |
| Verification hands back to planning on an insufficient fix and to diagnosis on a wrong cause, under one shared budget | D | `uv run pytest tests/unit/test_routing.py tests/unit/test_roles_verify.py` |
| Records carry lifecycle states, and retrieval demotes an invalidated record without hiding it | D | `uv run pytest tests/unit/test_lifecycle.py` |
| Distillation writes bounded evidence and lesson projections; a raw transcript never enters retrieval | D | `uv run pytest tests/unit/test_projections.py tests/unit/test_distil.py` |
| The gated second opinion: a retrieval-free first diagnosis, a bounded gate that sees only the candidate, precedents retrieved by a harness-built query, a blind second diagnosis, a label-shuffled selector, and a fall back to the first answer on any failure | D | `uv run pytest tests/unit/test_secondpass.py tests/unit/test_secondpass_wiring.py` |
| Four runtime controls: a generation cap, bounded nudges, a repeat breaker, one finalizer rescue | D | `uv run pytest tests/unit/test_loop.py` |
| An ordered per-incident log records every model call with its token counts and the sampling set it actually ran on, every tool call, every submission outcome and every intervention — including the distilling pass, which is a model call like any other | D | `uv run pytest tests/unit/test_store_sink.py tests/unit/test_observability.py tests/unit/test_learning_job.py` |
| Each experiment arm is a single-variable change over a shared base; naming one configures the run, the environment still outranks it, and an environment that contradicts what the arm holds constant fails at startup | D | `uv run pytest tests/unit/test_profiles_and_arms.py tests/unit/test_settings.py` |
| An environment's own fault injection is filtered out of every result before the model sees it, not only out of the ones a tool remembered to filter | D | `uv run pytest tests/unit/test_evidence_reaches_every_tool.py tests/unit/test_sregym_adapter.py` |
| The model is served privately and no telemetry leaves the operator's infrastructure | D | `configs/models/*.yaml` names the endpoint; `deploy/k8s/base/networkpolicy.yaml` is the egress boundary |
| The reported recovery rates, the effect of verification, the denser-model comparison, and the throughput figures | A | the data archive that accompanies the manuscript |
| Any of the archived rows can be re-measured end to end | I | the SREGym adapter, a served model, and a benchmark cluster |

One row deserves its own sentence, because it is the most easily overstated. **In every
measured campaign the approval gate ran in automatic mode, so no request ever reached a
person.** What is demonstrated above is that the mechanism works — it parks, persists,
and resumes the same incident on a decision, and that path is tested. It is not a claim
that a human approved anything during the measurements.

**No measured campaign has been run on this code.** Every number in the manuscript was
produced by the research prototype and is recomputed from the data archive that
accompanies the paper, not from anything here. What this repository reproduces is the
mechanisms behind those numbers, and its parity checks pin them against the prototype's
behaviour one mechanism at a time (`make test-parity`). A paired parity run on the
benchmark is the gate before this code is used to measure anything, and it has not been
run. The limitations section below repeats this where a reader looking for caveats will
find it.

## Running it

```
deploy/docker/    the image, and compose files for local and evaluation use
deploy/k8s/       manifests: base plus a local and a production overlay
```

The image is non-root, runs on a read-only root filesystem, and carries `kubectl`
because the default transport executes it. RBAC is derived from the verbs the tool
layer can actually issue: one cluster-wide read-only role, and namespaced read and
write roles an overlay binds per managed namespace. Secrets are in neither role.

**The approval and incident routes have no authentication yet.** Until they do, the
deployment guidance is the shipped NetworkPolicy and no ingress.

## Configuration

The package and command are now `maver`, and settings use the `MAVER_` prefix.
Existing `KIRA_` setting names remain aliases; `MAVER_` wins when both are set.
The PostgreSQL schema still defaults to `kira_memory` so the rename opens existing
data. Narrow compatibility imports restore previously scheduled approval and learning
jobs into the MAVER runtime. Compose keeps the old physical workspace-volume names by default; set
`MAVER_WORKSPACE_VOLUME` or `MAVER_EVAL_WORKSPACE_VOLUME` when the previous deployment
used a custom Compose project name. Kubernetes adopters should retain their existing
namespace and PVC bindings in their overlay when upgrading. The repository address
has not changed, and historical experiment configurations retain their original bytes.

Every setting is a field on a nested model, and an unrecognised `MAVER_`-prefixed name
stops the process rather than being ignored. `uv run maver config show` prints the
resolved configuration with secrets masked; `configs/arms/` holds the experiment
configurations, each declaring what it holds constant.

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

See [the customisation guide](docs/customisation.md), or run the complete offline
example with `uv run python examples/customisation.py`. It exercises agent inheritance,
YAML binding, a custom two-stage workflow, middleware and isolated conversations.

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
  adapters/      the environment seam, and the SREGym adapter
  api/           incident intake, health, and approval decisions

configs/         model profiles, experiment arms, adapter settings
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
`docs/adapters/sregym.md` covers that route's settings, loading an archived corpus with
`maver memory import`, and the leave-one-out map an imported corpus needs.

Two environment adapters ship, and the difference between them is instructive. Under
SREGym the harness starts the agent and hands it a problem. Under AIOpsLab the direction
reverses: a driver runs inside the benchmark's own interpreter, holds the cluster-admin
kubeconfig that the agent container is deliberately not given, and lets the agent reach
the cluster only through a token-scoped loopback bridge that re-screens every action.
`docs/adapters/aiopslab.md` covers deploying the agent, running one problem, and running
a resumable sweep. That adapter has not yet been run against a live AIOpsLab host from
this repository; no result is claimed for it.

`STYLE_GUIDE.md` carries the conventions a linter cannot express: the layer boundaries,
the tool-allowlist rule, the write-gate invariant, the artifact-schema freeze, and the
rule that keeps the graph library replaceable.

## Limitations and known issues

The provenance statement above is the first of these: no number reported anywhere was
measured on this code.

**Deferred deliberately.** Each of these is tracked, and what stands in for it today is
named so an adopter can judge the risk themselves.

- **No authentication or authorisation on the API.** Anyone who can reach the port can
  approve a cluster write. The shipped NetworkPolicy and no ingress are the interim
  posture, and a decision's `decided_by` is recorded as a claim rather than a verified
  identity.
- **Single tenant.** `memory.corpus_id` partitions the retrieval corpus, but nothing
  authenticates the caller who chose it. It is a research partition, not a security
  boundary; the supported deployment is one MAVER per tenant.
- **No rate limiting** beyond the intake concurrency bound. A burst of alerts is queued,
  not throttled. The turn budgets, the generation cap and the repeat breaker bound what
  one incident can consume, not what a client can ask for.
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

**Evaluation posture.** The approval gate ran in automatic mode in every measured
campaign. The paragraph under the claims table above says what that does and does not
imply.

## Citing

See `CITATION.cff`.

## Licence

MIT. See `LICENSE`.
