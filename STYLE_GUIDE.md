# Style guide

MAVER follows **PEP 8 as enforced by ruff**, and the **[Google Python Style
Guide](https://google.github.io/styleguide/pyguide.html)** for anything ruff does not
cover. Both are configured in `pyproject.toml`; `make lint` is the arbiter, and no
formatting question is worth arguing about that `ruff format` already answers.

This document carries only what a linter cannot express: the architectural and
research-validity invariants that keep the system correct and its measurements
comparable. Everything here has cost us something to learn.

---

## 1. Layer boundaries

Dependencies point one way. A module may import from the layers below it and never
from the layers above.

```
api  cli  adapters
        |
 orchestration   pipelines   agents
        |
  roles    secondpass
        |
 middleware   tools    memory
        |
      llm    audit
        |
  settings   contracts
```

Rules that follow, and how they are enforced:

| Rule | Enforcement |
|---|---|
| `contracts/` imports nothing from `maver` except `settings` | review |
| Core modules must never import `maver.adapters.<name>` | ruff `TID251` banned-api |
| `memory/backends/*` is constructed only by `memory/factory.py` | review |
| Nothing imports `api/` or `cli.py` | review |
| A role never imports another role | review |

**The adapter rule is the one that matters most.** An environment adapter *supplies*
capability to the core; the core never branches on which environment it is running
under. The prototype violated this: `agent/tools/mcp_client.py` opened its central
call path with `if aiopslab_bridge_enabled():`, so adding a second benchmark meant
editing the transport every module depends on. In MAVER an adapter provides a
`ToolTransport` and the core cannot tell the difference.

If you find yourself wanting `if benchmark == ...` anywhere under `src/maver/` outside
`adapters/`, the design is wrong, not the requirement.

## 2. Tools and authority

- **No new tool without an explicit per-agent tool selection.** Shipped roles retain
  their ordered default allowlists; configured agents can supply their own ordered
  lists. Register a new tool and its description together, and name which agents may
  use it. YAML selection never changes its canonical authority role or deployment
  write policy. Startup validation rejects unknown tools and incompatible submissions.
- **Only Execution issues classified Kubernetes API writes.** This is enforced twice on
  purpose: by the approval gate, which refuses a write from a stage the tool's policy
  does not name, and again inside the tool by a stage check of its own, because a gate
  that was misconfigured or absent should not be enough to let a diagnostic role issue
  one. `kubectl exec` is a separate action class: diagnostic roles may request it, but
  it requires human approval by default because an in-container command can mutate
  application state even without a Kubernetes API write.
- **Structured write and exec calls pass the approval gate.** The gate is a
  `wrap_tool_call` hook, so it covers any write-capable tool in any
  stage rather than a hand-picked set of call sites. Read-only commands auto-approve. A
  denied write or exec fails closed and is recorded in the event log as a policy
  refusal. SREGym's proxy shell cannot pause for HITL, so it refuses shell-based exec
  unless the deployment explicitly selects the unattended exception. Ordinary API
  writes through that shell route require an authorized stage and automatic authority.
- **Tool output is part of the interface.** Prompts are written against the exact
  shape of what tools return. Changing a tool's output format is a behaviour change
  that belongs in a parity run, not a cleanup commit.

## 3. Typed artifacts are frozen

Roles exchange structured artifacts, never transcripts. Planning receives a
`RootCause`, not the diagnostic conversation. Conversation stays in per-invocation agent state and in the audit store. Legacy
per-role message lists remain a compatibility projection; they are not shared context
for a different agent instance.

**The artifact schemas are held constant across every experiment configuration.** The
comparison between configurations is only single-variable if the submission schema
does not move with it. Adding a field to `RootCause` to help one configuration
silently invalidates every published contrast that used it.

Changing an artifact schema is therefore a research decision:

1. State which published results it invalidates.
2. Bump the schema version in `contracts/`.
3. Record it in `CHANGELOG.md` under a breaking-change heading.

A contributor who does not know this rule will improve a schema in good faith. That
is why it is written down here rather than left implicit.

## 4. Prompts

Prompts are `.txt` files under `src/maver/prompts/`, one per role, loaded through
`importlib.resources` and never by filesystem path.

This is a **deliberate deviation** from every major agent framework. LangGraph puts
prompts in `prompts.py` as module constants, Google ADK uses a `prompt.py` per role,
the OpenAI Agents SDK passes them inline, and pydantic-ai uses decorators. None of
them store prompts as files. We do because prompt content is an experimental variable
here: files diff cleanly in review, can be swapped without touching Python, and can be
read by someone assessing the artifact who does not want to read our code.

Rules:

- Templating uses `{{MARKER}}` and is **strict in both directions**: a marker with no
  value raises, and a value with no marker raises. Renaming a marker cannot silently
  leave a prompt half-wired.
- An adopter overrides a prompt through configuration, never by editing a file inside
  the installed package.
- Prompts state directives plainly. They do not reference past failures or explain
  what the model previously got wrong; the model has no access to that history, so
  such text is noise that consumes context.

## 5. Middleware

Cross-cutting concerns are hooks, not conditionals inside role code. The hook
vocabulary deliberately mirrors LangChain 1.x middleware:

```python
class AgentHook(Protocol):
    def before_model(self, ctx: HookContext) -> HookContext | None: ...
    def after_model(self, ctx: HookContext) -> HookContext | None: ...
    async def wrap_model_call(self, req: ModelRequest, handler: ModelHandler) -> ModelResponse: ...
    async def wrap_tool_call(self, req: ToolRequest, handler: ToolHandler) -> ToolResult: ...
```

We adopt the **pattern, not the API**. LangChain's `AgentMiddleware` only exists
inside `create_agent`, which builds its own internal graph; we build ours. Borrowing
the names costs nothing and makes the architecture immediately legible to anyone
arriving from LangChain.

The test for whether something is middleware: would you want to turn it off for one
experiment configuration, or reuse it across roles? Then it is a hook. The prototype's
loop reached 1,269 lines because the answer was always yes and the code always said
otherwise.

What a failing hook costs depends on the phase, and the split is deliberate. A shaping
or reviewing hook that raises is recorded to the incident's event log as a bypassed
intervention and skipped, because a run cut short by a misbehaving accessory is worse
than a run that lost one accessory. A **wrapping** hook that raises is not swallowed: it
has already taken responsibility for the call, and continuing as though it had not is
how an unauthorised cluster write gets dispatched. Do not write a `wrap_tool_call` hook
that catches its own exception and calls `handler(request)` anyway.

## 6. Configuration

- Settings are **nested pydantic-settings groups**, not a flat namespace of accessor
  functions. The prototype reached 198 environment variables across a 1,272-line
  module; roughly half of them existed so that one experiment could run once.
- **New knobs fail closed.** Unset values use documented defaults; invalid
  configured values fail validation at startup rather than being silently replaced.
- A knob ships with its settings group, its default and its documentation **in the
  same commit**. A live variable declared in no config file is invisible to everyone
  but its author.
- Credentials are `SecretStr`. They must not appear in a log line, a trace attribute,
  or `maver config show`.
- Environment variables override YAML, so a Kubernetes ConfigMap remains the
  deployment control surface.

### Agent and workflow customisation

- Python owns agent behaviour and workflow topology. YAML selects registered code and
  fills typed schemas; it never imports arbitrary modules or evaluates expressions.
- Extend an agent by subclassing its implementation and configuration schema. Each
  invocation gets a new implementation and middleware instances. Persist counters and
  trigger state in `AgentState.middleware` or `RunState.middleware`, not class fields.
- Canonical authority roles, workflow stage names, configured agent names and running
  instance IDs are distinct. Reusing a definition in two stages never shares a transcript.
- `RunState.metadata` and agent metadata are private. Prompt additions declare a
  recipient and lifetime. Custom shared artifacts live in `RunState.artifacts`.
- Select a pipeline once at admission and persist its resolved composition. Approval
  checkpoints restore it; they do not rerun selectors or substitute a new default.
- Local Python is trusted deployment code, not sandboxed third-party code. Mandatory
  runtime authority, budget and persistence mechanisms remain outside optional hooks.
- Preserve measured inputs and their provenance. An explicitly approved migration
  may advance the default behavior; retain historical fixtures and record the source
  revision and changed experimental condition in CHANGELOG.md. New components require
  explicit configuration. Bump implementation versions when Python behavior changes
  so a saved approval cannot silently resume under different code.

Workflow definitions may reuse the pure routing and outcome policies in
`orchestration/`; they do not import the runner or construct deployment resources.
`configuration.py` and `custom.py` assemble local registrations for the runtime.

## 7. Sampling is pinned as a set

The full sampling configuration, temperature, top-p, frequency penalty, max tokens,
seed where available, comes from one model profile, is applied together, and is
**asserted at startup and recorded in the run's audit record**.

Never pin one knob and assume the rest. A configuration change that moved temperature
while leaving everything else alone went unnoticed for weeks and made runs on either
side of it incomparable. The assertion exists so that the next such change fails loudly
instead of quietly producing numbers.

Sampling parameters belong to the **model**, not to the harness or the role. Different
endpoints accept different parameters; that is a property of the model being served.

## 8. LangGraph containment

LangGraph is a graph executor. Use `StateGraph`, add nodes, add edges, compile,
invoke. That is the whole permitted surface.

Do not use its checkpointers, savers, interrupt/resume, built-in human-in-the-loop, or
prebuilt agents. Persistence, resume, approval and lifecycle semantics are ours,
implemented in `orchestration/` and `middleware/`, because they are load-bearing for
the system's auditability claims and must not depend on a framework's roadmap.

The target: if LangGraph disappeared, replacing it should take days, not weeks.

## 9. Memory

- Only **distilled records** enter the retrieval corpus, marked with a lifecycle
  state. Raw transcripts stay in the audit store. A retrieval corpus of chat logs is
  not a memory system.
- **Retrieval never suppresses a record for its lifecycle state, but it does demote
  one.** A record of an attempt that failed tells a later incident what was already
  tried and what happened, and most of what is worth learning from an incident corpus is
  a negative result. Excluding `INVALIDATED` records was measured and lost ground:
  coverage fell 88% to 72% and 90% to 68%, and 5.7pp and 9.1pp fewer queries were served
  a true precedent first. Invalidation records what the prior session's remediation did,
  not whether what it observed was true, so an invalidated record can be both the only
  same-family precedent in the corpus and perfectly accurate about what it saw. It is
  therefore ranked at 0.3 of its lexical score and still returned: demotion keeps the
  coverage and costs only rank, where exclusion throws the coverage away. Do not add a
  retrievability filter on lifecycle state. `INVALIDATED` marks a claim that did not
  hold up or could not be confirmed, closer to inconclusive than to false.
- An operational run status is not a verdict on a lesson. Do not promote a record's
  validity from the fact that its run happened to succeed.

## 10. Observability and audit

Every stage emits OpenTelemetry spans, and every model call, tool call, submission
validation outcome and harness intervention is written to the ordered per-incident
event log. The log is what makes a run reconstructible after the fact, which is a
stated property of the system, not a debugging convenience.

If you add a model call anywhere, including inside a gate, a nominator or a selector,
it is counted. The prototype's second-pass helper calls bypassed the usage log, and
the cost figures had to be published with that gap disclosed.

## 11. Testing

- Unit tests need no cluster, database, model endpoint or network. `filterwarnings =
  error` is on; a new warning is a failure.
- Anything requiring real infrastructure is marked `integration` and excluded from the
  default run.
- Backends that share a port share a **contract test suite**. A new backend is done
  when it passes the existing suite unmodified, not when its own tests pass.
- Provider quirks get a regression test with a recorded fixture. A workaround with no
  test is a comment waiting to be deleted by someone who does not believe it.

## 12. Commits

[Conventional Commits](https://www.conventionalcommits.org/) with a scope:
`feat(memory):`, `fix(tools):`, `refactor(roles):`, `docs:`, `chore:`, `test:`.

- Commit code incrementally. Documentation-only commits are avoided; documentation
  lands with the code it describes.
- No `Co-Authored-By` or other attribution trailers.
- A commit body explains why, not what. The diff already says what.
