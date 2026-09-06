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
