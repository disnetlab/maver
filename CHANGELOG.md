# Changelog

## Unreleased

- **Breaking:** remove previous package imports and prefixed environment-variable
  aliases. The supported names are `maver` and `MAVER_`; unprefixed prototype aliases
  remain available for recorded experiments. PostgreSQL now defaults
  to `maver_memory`, and Compose physical workspace volumes default to
  `docker_maver-workspace` and `docker_maver-eval-workspace`. Set explicit schema and
  volume overrides to reuse existing data. Scheduled job stores referencing removed
  modules cannot continue unchanged: drain pending approvals and scheduled jobs, or
  migrate their stored references before upgrading. Preserve existing state; see
  [deployment upgrade guidance](deploy/k8s/README.md#upgrading-existing-deployments).
- Normalize Execution's string-encoded step arrays with the submission schema's shared
  decoder before checking for observations. Valid submissions no longer consume guard
  retries; command requirements, schema validation and the bounded fallback are unchanged.
- Add selectable `memory.retrieval_policy: historical` for the prototype's text-only
  retrieval: reciprocal ranks, lifecycle weights, identifier boosts, role lanes and
  exclusion after ranking. The `ma-s` and `ma-s-v` arms select it; general deployments
  retain `current` ranking. This restores a retrieval policy, not the complete historical
  experiment environment or its vector retrieval variants.
- Preserve explicit `authoritative_correction` provenance in corpus imports and storage
  (store layout version 10). Existing records default to false; upgrades do not infer
  external authority from a stored correction string.

Schema changes and model-facing text. `SCHEMA_VERSION` in `src/maver/contracts/__init__.py` versions the
submission schemas and the event contract together, and every run records it in its
configuration snapshot.

Changing an artifact schema is a research decision, not a refactor: it invalidates
published comparisons, which are single-variable only while the schema does not move
with the configuration. The procedure is in STYLE_GUIDE.md, "Typed artifacts are
frozen".

## Explicit approval boundary for kubectl exec

`kubectl exec` now requires human approval by default, even when the global authority
mode or a matching rule is `auto`. Deny decisions retain precedence, and approval
timeouts cannot auto-approve exec while the boundary is enabled. Diagnostic roles may
still exec after approval; the boundary governs approval rather than assigning exec to
one workflow stage.

The proxy shell cannot suspend a command for approval, so it refuses exec by default.
SREGym also hides `exec_kubectl_cmd_safely` by default; human-approved exec requires
`MAVER_ADAPTERS__SREGYM__MCP_KUBECTL_TOOLS=true` to expose that structured tool, which
uses the configured proxy transport despite the setting's historical name. Set
`MAVER_AUTHORITY__KUBECTL_EXEC_REQUIRES_APPROVAL=false` together with
`MAVER_AUTHORITY__MODE=auto` only when explicitly reproducing unattended SREGym behavior.
The SREGym adapter does not opt out automatically, and
`examples/sregym-unattended.yaml` records the required Compose configuration.

## Feedback and background-work race fixes

Learning now retains pending intents when temporarily disabled, and its repair sweep
does not queue a second copy of a running pass. The corpus write checks the operator
review sequence in the same transaction, rejecting stale lesson text even when a
review arrives after the final application-level check. Custom `TextSearch` backends
must accept and enforce `expected_feedback_sequence` on `upsert_summary`.

Kafka shutdown reaches the bounded incident drain while admission is still settling;
quarantine writes remain tracked until their storage threads finish. Adapter-owned
approval continuations now share the duplicate-run and draining guards. The storage
layout remains version 9; no submission schemas or prompts changed.

## Admission shutdown, durable Learning, and approval recovery

Shutdown now tracks pipeline selection and session initialization as well as active
runs. Late admissions cannot start a pipeline after the drain. Delayed Learning
persists explicit eligibility and external feedback, restores pending work after a
restart, and retries unsuccessful passes serially with backoff. Storage layout 9
adds Learning intents to both backends.

With restart recovery enabled, interrupted approval continuations require operator
reconciliation; approved operations are not replayed automatically. Concurrent
continuation jobs in one process dispatch once, and new continuations wait while the service drains.
The operations guide documents these behaviors and the upgrade requirements.

## Operational robustness and documentation

Releasing an intake reservation twice no longer frees another incident's capacity
slot. Audit appends preserve subsequent events after an interrupted JSONL line,
while export continues to report the damaged fragment. CLI commands report HTTP
and connection failures consistently, and the incident example stops when operator
reconciliation is required instead of polling until timeout.

The public operations guide covers intake, approvals, quarantine, restart recovery,
outcome review and storage upgrades. Deployment examples include the optional intake
and recovery settings.

## Durable intake, recovery, and operator feedback

HTTP idempotency keys and Kafka source/message keys now identify an accepted incident
durably. Duplicate deliveries return the prior receipt; a conflicting payload for the
same key is refused. Malformed or permanently rejected Kafka records are quarantined for
inspection and explicit retry. Operator feedback can be attached to a completed
incident and used by subsequent learning without rewriting the original agent artifacts.
Eligible interrupted stages can resume from a persisted pre-stage checkpoint when
recovery is enabled; uncertain external effects require operator reconciliation.
Storage layout version 8 includes durable receipts, quarantine and ordered feedback.
These changes affect intake and lifecycle behavior;
record the version and settings used for each experimental run.

## Local tool cancellation cleanup

Cancelling a tool now kills and reaps its local process group across shell, Python,
SSH, and direct/proxied kubectl paths. Timeout cleanup also reaches children whose
parent has already exited. Cancellation does not establish whether an already
submitted remote operation took effect; interrupted writes still need reconciliation.

## Gemini 3.5 Flash-Lite diagnostic profile and request pacing

A hosted Gemini 3.5 Flash-Lite model profile is available for isolated diagnosis
and validation. It leaves Gemini 3 sampling parameters at Google's defaults and
specifies a five-second minimum interval between provider requests. Adjust the profile
to the quota of the selected endpoint. The optional interval is enforced by
one client-side gate across calls and retries; existing profiles are unchanged.
The key belongs in the environment, never in the profile. This condition sends
incident telemetry outside the cluster and is a separate experimental condition
from local-model thesis arms.

## Complete tool invocation observations

Normal tool dispatch, automatic namespace snapshots, Verify health sweeps and approval
replay now record measured started/terminal attempts with full private backend and
result artifacts. The same invocation identity links nested reads to the model call
and bridge request. Workspace previews retain their existing presentation while the
audit keeps the complete filtered result and its pre-presentation failure status.
HITL refusals are explicitly marked refused, and an approved replay is graded by its
actual result rather than by the approval decision. Loop stage/submission events carry
a source label so workflow and role-loop observations can be counted separately.

This changes audit observations, not authority decisions, tool schemas, prompts or
execution arguments. The transport observer delegates through the existing protocol;
custom tools relying on a concrete transport type should use that protocol instead.
Older evidence cannot recover measurements that were never recorded. Submission
artifact schemas and audit schema version 4 are unchanged.

## Audit schema version 4 and honest telemetry coverage

Earlier audit records represented unknown tool measurements as zero/false and
incomplete telemetry as healthy. Evidence was limited to bounded database records
or overwriteable workspace files.

Audit version 4 adds model call IDs, per-provider-attempt events, full request/response
references, tool-start and tool observation fields, and stage source/cancellation
fields. Unknown tool duration and truncation now default to null. Strict audit readers
must accept the new event kinds and fields. Submission artifact schemas are unchanged.
Only instrumented producers populate the new fields; historical records cannot acquire
missing measurements retroactively.

The service now writes a durable private event/evidence mirror under `server.workdir/audit`
before offering events to the existing best-effort database sink. `maver audit export`
collects referenced evidence and optional AIOpsLab bridge artifacts, verifies hashes,
and reports capture limitations separately from file integrity. Full local bodies are
separate from OTLP spans, which exclude exception messages and stack traces. AIOpsLab
bridge log schema 2 records received/finished pairs and complete request/response artifacts.

Malformed telemetry envelopes and unavailable snapshot reads are reported explicitly.
Health checks distinguish unknown observations from healthy resources, while retaining
known failures. These change subsequent model observations; RCA and Verify implementation
versions advance. Generated workspace output names no longer overwrite earlier results
after recreation or resume. Earlier results retain their original measurement
boundaries; this change makes no benchmark performance claim. See
[the audit guide](docs/audit.md) for the current capture boundaries.

## AIOpsLab transport instructions and adapter wording overrides

The remediation suffix restricted kubectl to reads while Execution was tasked with
mutations, and shared diagnostic guidance advertised `kubectl exec` that the bridge
refuses.
The adapter now explicitly permits Execution's scoped mutations and supplies
consistent kubectl guidance and descriptions for RCA, Planning, Execution and
Verify. It uses the generic `RoleOverlay.prompt_blocks` and `tool_descriptions`
hooks; the core has no AIOpsLab branch. Completed workflow runs record effective
description overrides in existing run metadata, independently of prompt digests.

This changes AIOpsLab model inputs. Historical prompts and fixtures retain their
original meaning. Tool schemas, execution permissions, bridge enforcement and
voluntary opening reads are unchanged.

## AIOpsLab complete structured reads

Successful bridge responses were previously cut to 10,013 characters before
namespace snapshots and trace tools could parse them.
The AIOpsLab transport now returns complete successful kubectl and structured
telemetry output. Tools parse and scope those results before applying their shared
16,384-character display limit and saving full scoped output to the session workspace.
Kubectl also preserves complete explicit `save_as` results. Unstructured direct bridge
actions and bridge failures retain their 10,000-character bound.

This changes observations in subsequent AIOpsLab runs: snapshots can inspect large
resource lists and agents can read complete saved results. Earlier results retain the
original behavior. No prompts, artifact schemas, namespace restrictions or write
authority change.

## Execution scope and bounded Verify hand-backs

Port behavior recorded at prototype revisions `cb1ed3b6462feaa090c4c7b3c273753c7aece20b`
(Execution prompt, 2026-09-11) and `00e37eadbc13bdaa8e9dec8794bd4bf3fa9270a6`
(Verify hand-back cap and submission repair, 2026-09-16). The second revision contains
the Verify normalizer; the corresponding prototype loop hook and wiring were not
committed. MAVER therefore supplies its own integration and audit record.

- Execution's default prompt now limits success to mechanical confirmation that
  planned changes applied. Persistent symptoms alone do not justify replanning.
  The earlier prototype and composed-prompt fixtures remain unchanged for historical
  reproduction; the new prompt is pinned separately to its source revision.
- `verify.handback_retries` / `MAVER_VERIFY__HANDBACK_RETRIES` caps Verify's returns
  within the shared remediation budget. Unset inherits that budget; zero disallows
  Verify returns. `multi` pipeline YAML may set `config.verify_handback_retries`
  to tighten the deployment ceiling. Execution replans and inconclusive rechecks
  retain their existing budgets. Counts persist across approval pauses and are
  audited as `verify_handbacks_taken`, excluding the verdict denied by the cap.
- Verify repairs missing entry-point status only from a successful recorded
  `probe_service` result; missing not-recovered routing fields use its submitted
  summary/checks. The model's verdict and raw transcript remain unchanged, and
  health/shape guards still run after repairs.

These are behavioral changes. Existing results retain their original prompts and
control policy; new runs must record the changed condition. Execution, Verify and the
built-in multi workflow advance their versions so pinned approvals cannot silently continue
under changed behavior. Frozen experiment-arm files and artifact schemas are unchanged.

### Schema version 3: audited submission repair

`InterventionEvent` adds `control="submit_normalization"` with the existing
`action="rescued"`. Its detail records original/normalized arguments and the repair
sources. Older strict audit readers must accept this control value. Submission
artifact schemas are unchanged.

## Customisable agents and workflows

Python agent subclasses now expose typed settings to final named YAML agents.
Python workflows bind these agents through YAML, with explicit per-incident pipeline
selection, optional selector middleware, and a deployment default. Shared private
state, scoped prompt additions and separate per-invocation conversations survive
approval checkpoints. See [the customisation guide](docs/customisation.md).

Session storage layout 4 adds a private runtime JSON column to SQLite/PostgreSQL;
existing data migrates forward on open. An approved incident refuses continuation
if its pinned composition or effective behaviour changed. Historical checkpoints
without this metadata retain the legacy continuation path.

### Schema version 2: audit extension

`InterventionEvent` accepts `control="middleware_context"` and `action="injected"`
so explicitly contributed context is recorded verbatim. Older strict event readers
must be updated for these new values. Submission schemas, shipped model prompts,
tool descriptions and frozen experiment arm files are unchanged; this event-only
addition does not invalidate published model-input comparisons. Configuring new
agent behaviour defines a new experimental condition and requires its own evidence.

## Breaking changes

### MAVER naming

- Rename the previous framework's Python package, CLI, entry-point groups and
  deployment examples to MAVER. Use `maver serve` and `MAVER_` settings. The initial
  rename retained compatibility aliases and existing physical data names; the
  compatibility removal and current defaults are documented under Unreleased.
  Frozen prompts and arm files are unchanged; the rename changes no recovery policy.


### Schema version 1

- **Artifacts reject unknown fields.** The submission models set `extra="forbid"`, where
  the archive took pydantic's default `extra="ignore"`. A field removed from an artifact
  left its callers passing a dead keyword that was silently swallowed — `ExecutionResult`
  carried `needs_reassessment=False` past the removal of the field — so the rot survived
  until something read the artifact and found nothing there. Under `forbid` it fails at
  construction. A model whose submission carries a field the schema does not name is now
  rejected and asked again, where the archive accepted it and dropped the field, so the
  two are not interchangeable on a run that produced such a submission.

## Safety

- **`ssh_exec` runs only the host binaries it exists for.** The tool hands its argument
  to the node's login shell, which on a k3s host holds a cluster-admin kubeconfig, and
  the kubectl guard screened by position only: `xargs kubectl`, `sh -c 'kubectl …'`,
  `eval`, `timeout 5 kubectl` and a script wrapping any of them all reached the cluster
  outside the kubectl tool's screen, the approval gate's selectors and the
  ServiceAccount's RBAC. Every simple command in the line must now start with one of the
  host binaries named in `src/maver/tools/host.py`, and the tool's description says so.
  The refusal surface is wider than the archive's, and no measured run enabled the tool:
  `MAVER_TOOLS__HOST_RECOVERY_NODES` was empty in every campaign, so every call returned
  the "not an allowed recovery node" refusal before reaching a command.

- **`ssh_exec` is offered whether or not a recovery node is configured.** The toolset
  resolver withheld it from a deployment with an empty host allowlist, on the same rule
  that drops a log tool with no log backend behind it. Every measured campaign ran with
  an empty allowlist and offered the tool regardless, so withholding it removes a tool
  definition from the remediating role's request and changes what the model was shown.
  It is offered again, and a call naming a node the deployment may not reach gets the
  refusal the archive gave: `error: host '<node>' is not an allowed recovery node;
  configured: (none configured)`. The startup warning now says that is what will happen
  rather than saying the tool is withheld.

## Model-facing text

Prompts and tool descriptions are held-constant inputs to every published measurement,
and unlike code they are read by the model rather than executed, so a change to them is
recorded here rather than left to the commit log.

### Role prompts

- **The Learning prompt no longer prescribes the playbook layer.** Roughly 100 of its
  262 lines described a mutable shared playbook: four `playbook_*` tools, none of them
  registered or on any allowlist, and the `domain` and `human` triggers, neither of
  which ships. It reached the model on every distillation pass of every
  experience-enabled configuration, where the only two things it could do with it were
  spend turns on a call that does not exist or ignore that part of its instructions.
  The prompt now describes the one artifact Learning produces, the per-session summary.
  The trigger's own half of that system prompt (`runtime/learning/post_incident.txt`,
  appended to it by the role) went the same way: a `## Playbook Updates` chapter whose
  first instruction was to call `playbook_read` and whose fifth was `playbook_remove`,
  and two sentences elsewhere. Both are deliberate divergences from the prototype's
  assets, registered as such in `tests/unit/test_prompt_assets.py`.

### Tool descriptions

- **`bash_exec`, `code_exec` and `exec_kubectl_cmd_safely` no longer branch on
  `KUBECTL_MODE`.** The two-mode conditional described a `proxy_exec` mode in which
  kubectl resolved to a proxy shim inside the sandbox, and at the time this build had no
  such route: the sandbox carried no cluster credentials and both sandbox tools refused
  kubectl unconditionally, so the model was told in one request both that the sandbox has
  no credentials (the kubectl guidance block) and that a mode exists in which kubectl
  works there. `exec_kubectl_cmd_safely` also dropped "via the kubectl MCP server", which
  named one adapter's transport in a core description. The route has since been restored
  under SREGym, recorded below, and the descriptions branch on it again. They now branch
  on the bound transport rather than on an environment variable.

## Retrieval

The candidate pool a query is ranked over is an input to every published retrieval
number, so a change to how it is built belongs here rather than in the commit log.

- **PostgreSQL's candidate stage moved from `english`-on-raw-text to a vector over
  normalised terms, so that the embedded backend can be identical to it.** The campaign
  was measured on PostgreSQL, whose `to_tsvector('english', ...)` applied its parser, its
  127-word stopword list and Snowball English stemming before BM25 scored the pool in
  Python. SQLite's FTS5 index applied none of that: on a 120-record corpus five of six
  queries were served a different top five, `pod crash` reached a record reading "the
  pods were crashing" on PostgreSQL and nothing on SQLite, and a query of nothing but
  stopwords matched the whole SQLite corpus and none of the PostgreSQL one. An adopter
  who will not run PostgreSQL still has to be able to reproduce the retriever that was
  measured, so both backends now index and query the output of one Python normaliser
  (`maver.memory.lexemes`) and their pools are equal by construction. On the campaign's
  116-record corpus the two return identical candidate sets for all 30 queries tried.

  Against the pool the old PostgreSQL path built, 15 of those 30 queries are unchanged
  and every one of the 118 record-level differences is one of two documented classes:
  115 are compound tokens PostgreSQL's parser keeps whole — `RunContainerError/`
  `CrashLoopBackOff` was one lexeme there and is two terms here, so a query naming either
  half now reaches the record — and 3 come from five words in 4,413 where the vendored
  Snowball stemmer is a later revision than PostgreSQL 17's. Verified against a live
  server over 400 strings of real incident text: 52.5% of strings normalise to exactly
  PostgreSQL's lexemes, 91.5% of distinct lexemes agree, and 100% of the differences fall
  in those classes. Retrieval numbers produced before this change and after it are
  therefore comparable but not identical.

  BM25 scoring is unchanged: it still tokenises the raw text, because raw tokens are what
  was scored. Both backends migrate an existing database on open, deriving the terms from
  the raw text they already hold.

- **A retrieval returns three records, not five.** Five was the prototype's code default
  and a depth nothing recent measured: the archived environment of the campaigns this
  build reproduces sets `MEMORY_SEARCH_TOP_K=3`, and the frozen memory arms pin three. A
  deployment left on the shipped default therefore read two records more per retrieval
  than any published result describes. Recall and precision pull hard against each other
  at this knob. Over the offline corpus, 1 to 20 results moves useful-precedent recall
  from 0.39 to 0.59 while precision falls from 0.39 to 0.04, and three is the precision
  end of that trade, which is the end a small model that anchors easily wants. The
  earlier MA+R arm, whose campaign ran at five, pins five in its own file.

- **An incident can be denied records that the deployment as a whole is not.** The
  exclusion setting is a process-wide floor, and its description already said that an
  evaluation adapter unions a per-incident leave-one-out set on top of it; nothing could,
  because both places the floor was read went straight to settings. An adapter now names
  the records one episode may not retrieve, before any role runs, and that set is unioned
  with the floor at both of them: the gated second pass and every role's tool context.
  It rides on the incident state, so a run that pauses on a human is denied the same
  records after the decision as before it, and it is written to the incident's own
  environment event, so a cell's withheld set is reconstructible rather than merely
  counted.

  The SREGym adapter supplies the published protocol and owns it: the exact problem is
  withheld and its cross-application twins stay retrievable
  (`adapters.sregym.leave_one_out`, on by default). The archive applied the same set per
  cell from outside, as a list of session ids handed to the process; it is now derived
  per episode from the problem the conductor served, which is the same set with nothing
  to keep in step by hand. A deployment with no adapter retrieves exactly what it did
  before.

## Storage

Where a record is kept, as opposed to what it says. `STORE_SCHEMA_VERSION` in
`src/maver/memory/backends/schema.py` versions the store layout — the tables, columns and
indexes a database holds — and is a different number from `SCHEMA_VERSION` above, which
versions the submission schemas and the event contract.

- **Stores are versioned, and migrate themselves on open.** Both backends keep a
  one-row `schema_version` table. A new database is created at the current version and
  stamped with it; a database written before this mechanism has its layout read off its
  own tables once, is stamped with what that says, and is then migrated a step at a
  time. Each step applies its change and stamps the version it produces in one
  transaction, so a migration interrupted part-way leaves the store at the version it
  began at rather than between two, and the next open finishes it. The move to the
  normalised-terms index described under Retrieval is step 1 → 2, so a store from any
  earlier build opens and catches up with nothing for an operator to run.

  A store stamped *above* the version the running build knows is refused at open, naming
  both numbers. An older binary cannot be told what a newer layout means, and one that
  wrote past it would corrupt the corpus quietly rather than fail; the deployment runs
  the newer build, or points the older one at another database.

- **A record carries the environment's own labels, and the store layout is 3.** Step
  2 → 3 adds one column to the record index of both backends — an unindexed
  `labels_json` in SQLite, a `labels JSONB` in PostgreSQL — defaulting to empty, so an
  existing corpus opens, gains the column and says exactly what it said before. The
  labels are written from the underscore-prefixed keys of the session's stored signal,
  which are the keys every prompt builder already drops, and they are never rendered
  into a projection, a prompt or a tool result and never indexed for search: what a
  record says to a model does not change. They exist for one read,
  `TextSearch.session_ids_with_label`, so that an evaluation adapter can ask which
  records came from one of its own episodes without the core knowing what the key means.

## Adapters

An environment adapter decides what a benchmark result means, so the parts of one that
change a reported number are recorded here rather than left to the commit log.

- **Under SREGym, kubectl runs locally against the harness's filtering proxy.** The
  archive reached the cluster this way in every measured run: `KUBECTL_MODE=proxy_exec`
  put a shim on the sandbox's PATH, a real kubectl ran against a credential-free HTTP
  endpoint the harness serves, and `MCP_KUBECTL_TOOLS_ENABLED=false` took the MCP kubectl
  and rollback tools out of every role's toolset. This build had only the MCP route, so
  the model was shown a different set of tools, given different guidance about what the
  shell may do, and served by a kubectl implementation with different limits from the one
  the results describe. The proxy route is back and is what the adapter configures
  (`adapters.sregym.kubectl_proxy_url`, `http://127.0.0.1:16444`); the MCP kubectl tools
  are hidden unless `adapters.sregym.mcp_kubectl_tools` says otherwise, and telemetry
  still arrives over MCP. The shell cannot pause for approval, so `kubectl exec` is
  refused there unless the deployment explicitly disables the exec approval boundary
  as well as selecting automatic authority. The structured tool route can pause, but is
  unavailable unless the deployment enables `mcp_kubectl_tools`; despite that historical
  name, it uses the proxy transport configured by the adapter.

- **The AIOpsLab analysis answer says who decided it.** Two hand-written rules in the
  taxonomy mapper override the model's own classification — an operator mutation to a
  Kubernetes workload is forced to Virtualization/Operation Error, and a database
  credential fault to Application/Authentication Issue — and below them a keyword ladder
  assigns both fields when the model wrote no classification at all. The rules are kept,
  because AIOpsLab's four levels and six fault types are its vocabulary and the agent's
  `RootCause` does not have them, but every episode now writes `source` (`model`,
  `normalized` or `fallback`) and the rule's name into `maver-submission.json` beside the
  graded answer. An analysis accuracy quoted without that split reports the rules'
  accuracy as the agent's. The prototype had the rules and not the accounting.

- **The AIOpsLab bridge keeps an access log.** The prototype's handler discarded it, so
  the process holding the cluster-admin kubeconfig kept no record of what it was asked to
  do and the agent's own audit log was the only account of any cluster action. One JSON
  line per action now lands in the problem's results directory, carrying the session id
  the agent's transport sends, so the two sides can be reconciled on something better
  than a wall clock.

- **The AIOpsLab sweep's retry and resume rules are unchanged and now testable.** A
  result AIOpsLab graded is final: it is never run again, on a resume or after the runner
  exits nonzero cleaning up behind it, because retrying a graded failure replaces one
  cell with a second sample of the model. Only a non-grade is retried. The three
  `python -c` one-liners the bash sweep embedded — registry enumeration, task-type
  inference, grading — are functions with tests rather than strings inside single quotes,
  and `status.tsv` gains four columns (task type, attempts, agent incident id, duration)
  over the archived seven, which keep their names and meanings.
