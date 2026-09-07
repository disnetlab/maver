# Changelog

Schema changes and model-facing text. `SCHEMA_VERSION` in `src/kira/contracts/__init__.py` versions the
submission schemas and the event contract together, and every run records it in its
configuration snapshot.

Changing an artifact schema is a research decision, not a refactor: it invalidates
published comparisons, which are single-variable only while the schema does not move
with the configuration. The procedure is in STYLE_GUIDE.md, "Typed artifacts are
frozen".

## Breaking changes

### Schema version 1

- **Artifacts reject unknown fields.** The submission models set `extra="forbid"`, where
  the archive took pydantic's default `extra="ignore"`. A field removed from an artifact
  left its callers passing a dead keyword that was silently swallowed — `ExecutionResult`
  carried `needs_reassessment=False` past the removal of the field — so the rot survived
  until something read the artifact and found nothing there. Under `forbid` it fails at
  construction. A model whose submission carries a field the schema does not name is now
  rejected and asked again, where the archive accepted it and dropped the field, so the
  two are not interchangeable on a run that produced such a submission.
  See `refactor/PARITY-CHECKLIST.md` P57.

## Safety

- **`ssh_exec` runs only the host binaries it exists for.** The tool hands its argument
  to the node's login shell, which on a k3s host holds a cluster-admin kubeconfig, and
  the kubectl guard screened by position only: `xargs kubectl`, `sh -c 'kubectl …'`,
  `eval`, `timeout 5 kubectl` and a script wrapping any of them all reached the cluster
  outside the kubectl tool's screen, the approval gate's selectors and the
  ServiceAccount's RBAC. Every simple command in the line must now start with one of the
  host binaries named in `src/kira/tools/host.py`, and the tool's description says so.
  The refusal surface is wider than the archive's, and no measured run enabled the tool:
  `KIRA_TOOLS__HOST_RECOVERY_NODES` was empty in every campaign, so every call returned
  the "not an allowed recovery node" refusal before reaching a command.

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
  kubectl resolved to a proxy shim inside the sandbox. That mode is not in this system:
  there is no such setting and no such code path, the sandbox carries no cluster
  credentials, and both sandbox tools refuse kubectl unconditionally. The model was
  told, in one request, both that the sandbox has no credentials (the kubectl guidance
  block) and that a mode exists in which kubectl works there (the description). The
  entries now state the one behaviour: no credentials, kubectl rejected, use
  `exec_kubectl_cmd_safely`. `exec_kubectl_cmd_safely` also drops "via the kubectl MCP
  server", which named one adapter's transport in a core description.

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
  (`kira.memory.lexemes`) and their pools are equal by construction. On the campaign's
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
`src/kira/memory/backends/schema.py` versions the store layout — the tables, columns and
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

- **The AIOpsLab analysis answer says who decided it.** Two hand-written rules in the
  taxonomy mapper override the model's own classification — an operator mutation to a
  Kubernetes workload is forced to Virtualization/Operation Error, and a database
  credential fault to Application/Authentication Issue — and below them a keyword ladder
  assigns both fields when the model wrote no classification at all. The rules are kept,
  because AIOpsLab's four levels and six fault types are its vocabulary and the agent's
  `RootCause` does not have them, but every episode now writes `source` (`model`,
  `normalized` or `fallback`) and the rule's name into `kira-submission.json` beside the
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
