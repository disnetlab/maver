# Kira

**K**ubernetes **I**ncident **R**ecovery **A**gent: a memory-augmented multi-agent
pipeline that diagnoses and repairs live microservice incidents using a privately
served open-weight language model.

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
uv run kira config check         # validate the configuration a server would start on
uv run kira serve                # listen on 0.0.0.0:8080
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
| Diagnosis and planning are read-only; only execution holds write-capable tools, enforced when the toolset is built | D | `uv run pytest tests/unit/test_toolsets.py` |
| A stage-aware gate classifies read, write, exec and blocked, and refuses a write outside the stages that may issue one | D | `uv run pytest tests/unit/test_write_gate.py tests/unit/test_kubernetes_tool.py` |
| Any write can be held for a person: the run parks durably, and a decision resumes the same incident from the role that was interrupted | D | `uv run pytest tests/unit/test_approval_pause.py` |
| Execution opens on a forced read and cannot report a plan applied without having touched the cluster | D | `uv run pytest tests/unit/test_roles_execution.py` |
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

Every setting is a field on a nested model, and an unrecognised `KIRA_`-prefixed name
stops the process rather than being ignored. `uv run kira config show` prints the
resolved configuration with secrets masked; `configs/arms/` holds the experiment
configurations, each declaring what it holds constant.

## Layout

```
src/kira/
  contracts/     the typed artifacts, incident state and audit events
  settings/      nested settings, model profiles, arm files
  llm/           the provider client, with sampling pinned and asserted
  roles/         the bounded model/tool loop and the roles that run in it
  middleware/    the hook protocol, and the write gate and approval authority
  tools/         the registry, the capability modules, and the transports
  memory/        ports, ranking, lifecycle, projections, backends
  secondpass/    the gated second opinion, as five composable stages
  orchestration/ routing, the graph, the runner, outcomes, and the offline pass
  adapters/      the environment seam, and the SREGym adapter
  api/           incident intake, health, and approval decisions

configs/         model profiles, experiment arms, adapter settings
deploy/          the container image, and Kubernetes manifests with two overlays
examples/        one alert payload and the shortest script that drives an incident
scripts/         the two things an evaluation host runs outside this package
tests/           unit, contract and integration suites
```

The two scripts are worth naming because nothing in the package imports them.
`sregym-driver.py` is the path the benchmark's own agent configuration invokes; it loads
the driver from its file rather than by package name, because it runs in the harness's
interpreter where this project is not installed. `sregym-mcp-port-forward.sh` holds open
the port-forward that adapter's transport speaks through. Both are load-bearing and both
fail on a remote host part-way through a run, so the contract between the entry point and
the driver it loads is pinned by a test.

`STYLE_GUIDE.md` carries the conventions a linter cannot express: the layer boundaries,
the tool-allowlist rule, the write-gate invariant, the artifact-schema freeze, and the
rule that keeps the graph library replaceable.

## Citing

See `CITATION.cff`.

## Licence

MIT. See `LICENSE`.
