# Changelog

Schema changes only. `SCHEMA_VERSION` in `src/kira/contracts/__init__.py` versions the
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
