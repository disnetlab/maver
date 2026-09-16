# The SREGym adapter

SREGym starts the agent and hands it one benchmark problem at a time. The adapter takes
the problem from the conductor, builds a blinded incident signal out of the application's
own metadata, supplies the transport the tools speak, hands the answers over for grading,
and releases the environment for teardown.

Roles reach the cluster the way every measured run did, by running a real `kubectl` in the
sandbox against the harness's own credential-free filtering proxy rather than through its
kubectl MCP server; telemetry still arrives over MCP. The README's "Under SREGym" paragraph
describes that route and why it is the route rather than an option. The settings that
address it are `adapters.sregym.kubectl_proxy_url` and the three timeouts beside it.

This page covers recovery-loop configuration and running the memory arms against
a corpus distilled by an earlier system.

## Recovery-loop configuration

For a new verification-enabled run, set these in the agent container's environment
(`deploy/docker/maver.env` for the evaluation Compose deployment):

```dotenv
MAVER_VERIFY__ENABLED=true
MAVER_VERIFY__HANDBACK_RETRIES=1
```

The cap permits one Verify return to RCA or Planning within the shared remediation
budget. Zero permits none; unset inherits the shared limit. Execution replans and
inconclusive rechecks keep their existing budgets. The prototype's flat
`VERIFY_HANDBACK_RETRIES` spelling is not read by MAVER.

Recreate the container after changing its environment file. Per-pipeline YAML can
impose a tighter cap; see [the configuration guide](../customisation.md#bound-verify-hand-backs).
The current Execution prompt and Verify submission repairs adopt later prototype
behavior. Record the revision and effective settings for new runs; selecting a
historical arm name alone does not restore the older prompt or control policy.

## Loading an archived corpus

A memory arm can only be re-run against the corpus it was measured on, and that corpus
lives in the prototype's PostgreSQL table rather than in a MAVER store. `maver memory import`
reads it from a file:

```bash
maver memory import corpus.jsonl --corpus so_gate_corpus_35b
```

The records go in through the same port the distilling pass writes through, so both
backends build their own normalised index — FTS5 terms in the embedded store, a `tsvector`
in the server one — with the normaliser retrieval queries with. Copying rows into a
database instead produces records `open_record` returns and no query ever reaches.

The import is keyed by session and validated whole before the store is opened: a bad line
names itself and leaves nothing behind, and a session the corpus already holds is skipped
unless `--replace` is given. `--corpus` names the corpus to write into, defaulting to the
configured `memory.corpus_id`. The corpus **is** the partition — one SQLite file per corpus
under `<workdir>/memory/`, and a scoped set of rows on PostgreSQL — so the run that reads
these records has to be configured for the same name:

```bash
MAVER_MEMORY__CORPUS_ID=so_gate_corpus_35b   # legacy: MEMORY_EXPERIMENT_ID
```

### The file

JSON Lines, one object per line, whose keys are the record's own field names. Only
`session_id` and at least one of the two projections are required; every other field
defaults to empty, and an unknown key is refused by name.

| Field | Type | What it is |
|---|---|---|
| `session_id` | string, required | Names the record. One session is one row. |
| `evidence_text` | string | What the incident looked like. Matched by an evidence-scope query. |
| `lesson_text` | string | What was concluded. Matched by a lesson-scope query. |
| `incident_summary` | string | Prose for a reader; indexed by neither scope. |
| `lifecycle_state` | string | `pending`, `tentative_success`, `human_verified`, `invalidated`, `halted`. Imported as given; a state this build cannot name is reported and ranks at the default weight. |
| `fault_category` | string | The record's own category, carried into a hit's metadata. |
| `alert_name` | string | |
| `namespace` | string | |
| `affected_resource` | string | |
| `verification_verdict` | string | |
| `evaluation_label` | string | Where the record's grade came from, if it had one. |
| `evaluation_source` | string | |
| `evaluation_diagnosis_success` | bool or null | |
| `evaluation_diagnosis_score` | number or null | |
| `evaluation_mitigation_success` | bool or null | |
| `corrected_cause` | string | Kept out of every projection: it states the answer. |
| `labels` | object of strings | The environment's own keys. `external_key` is the one leave-one-out reads. |

### The export

The prototype stores its records in `agent_memory.memory_summaries`, one campaign per
`experiment_id`. This reads one campaign out as JSON Lines; `-At` is what makes each row a
single unaligned line, which is the format:

```bash
psql "$PROTOTYPE_DSN" -At -c "
SELECT jsonb_build_object(
  'session_id',           session_id,
  'evidence_text',        coalesce(nullif(evidence_text, ''), retrieval_card, ''),
  'lesson_text',          coalesce(lesson_text, ''),
  'incident_summary',     coalesce(incident_summary, ''),
  'lifecycle_state',      coalesce(state, 'pending'),
  'fault_category',       coalesce(fault_category, ''),
  'alert_name',           coalesce(alert_name, ''),
  'namespace',            coalesce(namespace, ''),
  'affected_resource',    coalesce(affected_resource, ''),
  'verification_verdict', coalesce(verification_verdict, '')
)::text
FROM agent_memory.memory_summaries
WHERE experiment_id = 'so_gate_corpus_35b' AND retrievable
ORDER BY started_at" > corpus.jsonl
```

`experiment_id = '<corpus>' AND retrievable` is the whole filter the archived memory arms
retrieved under, and `<corpus>` is the name to import under and to run under.

Two things about that query are decisions rather than syntax. `retrieval_card` is the
fallback for `evidence_text` because records written before the projections were split
carry their text there and nothing else; a record with neither projection is refused by
the importer, since no query could ever match it. And `state` is the lifecycle column: it
is imported as it stands, so an arm that ran with `memory.lifecycle_weights` pinned still
ranks these records the way it did.

**The evaluation fields come out of `metadata_json`, whose keys differ by campaign.** Check
what the rows actually carry before adding them:

```sql
SELECT DISTINCT jsonb_object_keys(metadata_json)
FROM agent_memory.memory_summaries
WHERE experiment_id = 'so_gate_corpus_35b';
```

Then add a line to the object for each key that exists, mapping it to the field name from
the table above. Leave out the ones that do not: an absent field defaults, where a wrong
one is a claim about how the record was graded.

```sql
  'evaluation_label',              coalesce(metadata_json->>'<key>', ''),
  'evaluation_source',             coalesce(metadata_json->>'<key>', ''),
  'corrected_cause',               coalesce(metadata_json->>'<key>', ''),
  'evaluation_diagnosis_success',  metadata_json->'<key>',   -- true, false or null
  'evaluation_diagnosis_score',    metadata_json->'<key>',   -- a number or null
  'evaluation_mitigation_success', metadata_json->'<key>',
```

`->>` renders a JSON value as text and `->` keeps it typed, which is why the flags and the
score use the second: the importer wants a boolean, not the string `"true"`, and says so
with the line number when it gets one.

## Leave-one-out over an imported corpus

An episode withholds the corpus records of its own problem, so a cell is scored on what
generalises rather than on having seen this problem before. The adapter normally derives
that set from the records themselves: the distilling pass copies the episode's problem id
onto every record it writes, under the `external_key` label, and the episode asks the
corpus which records carry its own.

**An imported corpus usually cannot answer.** The archived campaign's sessions recorded a
problem id nowhere, so records exported from them carry no label, the derived set is empty,
and the cell retrieves its own answer — the direction that flatters a memory arm. Two ways
out, and they compose:

- Emit `"labels": {"external_key": "<problem id>"}` on each line of the export, if the
  session-to-problem mapping is recoverable. Nothing else is needed.
- Name the records per problem in a file, which is what the archive did.

```bash
MAVER_ADAPTERS__SREGYM__EXCLUSION_MAP=/etc/maver/exclusions.tsv
```

One row per problem: the problem id, a tab, and a comma-separated list of the session ids
that problem produced. Blank lines are skipped. A row with an empty list is a problem that
produced no records, which is a statement rather than an omission.

```
k8s_target_port-misconfig	sess-0a1f,sess-77c2
misc_config_baseline_hotel_reservation	sess-91bd
pod_cidr_exhaustion_hotel_reservation
```

The file is read at startup, and the ids it names for the current problem are **unioned**
with the label-derived set rather than replacing it, so a corpus that is partly imported
and partly written by this deployment is covered by both at once. What was withheld is
written to the episode's `acquired` audit event, and the count and where each half came
from is logged as the episode opens.

Two failure modes are treated differently, on purpose:

- **A line that cannot be read stops the deployment**, naming the line. An exclusion that
  fails open voids the cell it was measuring.
- **A problem with no row is a warning**, and the episode withholds whatever the labels
  gave it. The archived runner aborted the cell instead; this process serves whatever
  episode the harness starts next, so refusing would take a sweep down over one missing
  row, and the warning is the record that the cell ran with less withheld than the
  protocol says.

`memory.exclude_record_ids` (legacy `MEMORY_EXCLUDE_SESSION_IDS`, the name the archived
runner exported per cell) is still read, as a process-wide floor unioned with all of the
above. It is per process rather than per episode, so it fits a runner that restarts the
agent for every problem and not a service that is handed one episode after another.
