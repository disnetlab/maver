# Customising a MAVER deployment

MAVER is a system to clone, configure and adapt. Python defines supported behaviour;
YAML creates the final named agents and binds them into Python-defined workflows.
The existing recovery workflow remains available as `multi`, alongside `single`
and `diagnose`.

Run the complete offline example from the repository root:

```sh
uv run python examples/customisation.py
```

It uses a scripted model and temporary SQLite storage. No model server, telemetry
service or cluster is contacted. Two workflow stages use the same configured agent
with separate conversations. Middleware contributes a next-call note, while a
private ticket identifier stays out of model input.

## Configure an existing implementation

Set `MAVER_CUSTOMISATION` to a YAML file:

```yaml
agents:
  short_diagnoser:
    implementation: rca
    config:
      model_profile: qwen3.6-35b-a3b
      limits:
        max_turns: 30
      memory:
        enabled: false

pipelines:
  quick_check:
    implementation: diagnose
    agents:
      rca: short_diagnoser

default_pipeline: quick_check
```

```sh
export MAVER_CUSTOMISATION=configs/local.yaml
uv run maver config check
uv run maver config pipelines
uv run maver serve
```

`implementation` selects registered Python code. `config` accepts exactly the fields
that implementation exposes. Unknown settings, tools, middleware, agents, workflow
stages and mismatched output contracts fail during startup. Duplicate YAML keys also
fail rather than silently replacing earlier declarations.

Common agent settings are `model_profile`, `prompt`, `tools`, `middleware`, `limits`
and `memory`. Omitted values inherit the Python schema's defaults, then the deployment
settings where the default is unspecified. Lists replace the inherited list in full;
tool order and middleware order are preserved. Model profiles retain the complete
sampling configuration as one unit.

`prompt` names a role prompt asset, such as `rca`, or a file stem under the configured
prompt override directory's `roles/` folder. For `prompt: database`, supply
`roles/database.txt` there. It is not a Python import path or an arbitrary filename.
Registered Python implementations own input construction, output schemas, guards
and stopping behaviour. A YAML setting cannot create an unimplemented feature.

Named agent memory configuration selects access to the deployment's experience
store. `enabled: false` disables it for that agent; `true` cannot create a store when
the deployment has disabled memory. Storage backends and credentials remain
deployment resources.

## Bound Verify hand-backs

The built-in `multi` workflow can limit how many times Verify sends an incident
back to RCA or Planning. Set the deployment ceiling in the environment:

```sh
export MAVER_VERIFY__ENABLED=true
export MAVER_PIPELINE__REMEDIATION_RETRIES=2
export MAVER_VERIFY__HANDBACK_RETRIES=1
```

`MAVER_VERIFY__HANDBACK_RETRIES` replaces the prototype's `VERIFY_HANDBACK_RETRIES`.
The old flat name is not an alias for this new setting. Omit it to inherit the
shared remediation limit. Zero prohibits Verify hand-backs; one permits one.
The shared remediation budget still applies, so this cap can only shorten a run.
An Execution-driven replan spends only the shared budget. An inconclusive Verify
recheck spends only `MAVER_VERIFY__INCONCLUSIVE_RETRIES`.

Individual pipelines may impose a tighter limit. The following configuration is
available as [examples/verify-handbacks.yaml](../examples/verify-handbacks.yaml);
set `MAVER_CUSTOMISATION` to that file to use it:

```yaml
pipelines:
  recovery:
    implementation: multi
    config:
      verify_handback_retries: 1
  one_pass:
    implementation: multi
    config:
      verify_handback_retries: 0
default_pipeline: recovery
```

Omitted agent bindings use the registered defaults. Omitted or null pipeline caps
inherit the deployment cap; a larger pipeline value cannot raise that ceiling.
These limits control returns after verification, not whether verification runs:
enable Verify at deployment level as above. Custom Python workflows can inherit
`MultiWorkflowConfig` and reuse the built-in transitions, or use
`RoutingPolicy.verify_handback_limit` in their own routing.

The incident counter `verify_handbacks` and the completion audit field
`verify_handbacks_taken` count returns actually taken. They survive approval pauses
and new remediation cycles. The prototype instead counted hand-back verdicts,
including a final verdict that its cap refused; the permitted routes are the same.

Execution now confirms that each planned change applied at the resource level.
It does not turn a successful apply into a replan merely because the incident's
symptom remains. Verify independently judges recovery. Verify submission repair
fills only missing fields from recorded probe results or the model's submitted
evidence; it retains the model's verdict, preserves the raw transcript, and records
the repair before the existing submission checks run. If the normalizer fails,
the original arguments still go through those checks and the audit records a bypass.

The audit contract is version **3**: repairs appear as
`InterventionEvent(control="submit_normalization", action="rescued")`, with original
and normalized arguments, the tool-call ID and repair sources in its JSON `detail`.
Normalizer failures use `action="bypassed"`. This version is independent of the
arm/model YAML schema versions and the session-store layout version.

## Add behaviour by Python inheritance

Subclass an existing agent and its configuration model. The example defines:

```python
class DatabaseConfig(AgentConfig):
    schema_names: tuple[str, ...] = ("orders",)


class DatabaseAgent(RCAAgent):
    version = "1"
    config_type = DatabaseConfig
    config: DatabaseConfig

    def prepare_context(self, context: RoleContext) -> RoleContext:
        context = super().prepare_context(context)
        # Add database-specific context here, preserving the existing overlay.
        return context
```

See [the runnable implementation](../examples/customisation.py) for the actual
context addition. `prepare_context` retains the inherited role's loop, model
integration, submission checks and tool dispatch. Override `run` only when the
implementation needs different execution behaviour. Output contracts are declared
in the Python `role_spec` and checked against both the submit tool and pipeline stage.

Register the subclass in `src/maver/custom.py`:

```python
def register_agents(registry: AgentRegistry) -> None:
    registry.register("database", DatabaseAgent)
```

The name becomes available to YAML as `implementation: database`. Keep local
implementations in a module under `src/maver/` and import them from `custom.py`.
There is no requirement to package a plugin or modify a global allowlist dictionary.
The sample's `register_agents`, `register_pipelines` and `pipeline_selectors` functions
can be imported from that local module.

Each invocation receives a new implementation object, newly constructed middleware
and an independent configuration copy. Inheritance shares behaviour and defaults;
it does not share conversations, counters or mutable state. The configured name is
distinct from the canonical authority role: naming an RCA agent `repairer` gives it
no additional write permissions.

Keep credentials in deployment settings or environment variables. Custom agent and
middleware schemas should expose environment-variable names, never raw credentials.
Their resolved configuration is persisted for reproducibility. Human-readable
`config show` output masks secret-like field names, but that is not a secret store.

## Middleware and private state

Register a typed factory with `registry.register_middleware(...)`. YAML supplies
the factory's validated configuration, and each invocation constructs a fresh hook.
Factories receive the role context; model hooks receive `TurnContext`.

The shared `RunState` exposes private run metadata, shared artifacts, run-wide
middleware state, per-instance `AgentState` objects and queued prompt additions.
Place custom shared artifacts in `run.artifacts`. Each agent state has its own
conversation, metadata and middleware state. Live model
clients, transports and locks stay in runtime context, outside persisted state.

Middleware can inspect private metadata and deliberately queue model-visible text:

```python
run.add_context(
    "Inspect database availability before proposing a repair.",
    source="operator_note",
    recipient=agent.id,
    lifetime="next_call",
)
agent.middleware["operator_note"] = True
```

Use an instance ID for the current invocation or a workflow stage name to address a
later stage. `next_call` is consumed after a successful model response; `invocation`
remains for the receiving instance. Record trigger state in `agent.middleware` or
`run.middleware` so it survives an approval pause. Metadata is never automatically
rendered into prompts. Only deliberate context additions become model input.

Middleware executes in declared order within the runtime's mandatory write gate.
Budget accounting, submission enforcement, persistence and deployment authority
remain runtime responsibilities. Local Python implementations are trusted deployment
code; these interfaces do not sandbox untrusted Python.

## Define and select workflows

A Python `WorkflowDefinition` declares named stages, their authority and output
contracts, an entry stage, and routing functions with explicit allowed destinations.
YAML binds configured agent names to those stages and supplies the workflow's typed
settings. It does not define edges or execute expressions.

The example's `database_review` workflow runs `investigate` then `review`. Both bind
the same database agent definition; each receives its own execution identity and
conversation. `database_single` instead binds it to the existing `diagnose` workflow.
The same deployment can serve either composition.

A pipeline may also name `learning_agent` beside its `agents` and `config` entries.
Omitting it retains the configured canonical `learning` agent; an explicit name binds
a Learning implementation, and `null` disables the learning binding. Workflows can
also disable post-incident hooks in Python, as the diagnostic example does.

At incident admission the order is:

1. Explicit `pipeline` in the incident request.
2. Optional Python selector middleware, in registered order.
3. YAML `default_pipeline`, or the existing deployment mode when unspecified.

Selectors inspect the incoming alert and private metadata, return a configured
pipeline name, or return `None` to abstain. An explicit name always wins. The chosen
composition is persisted once; resuming an approval does not run selectors again.
This durability covers approval checkpoints and process restarts while parked; it
is not an automatic resume mechanism for an arbitrary mid-call crash.

```json
{
  "alert": {"alertname": "DatabaseUnavailable", "namespace": "shop"},
  "pipeline": "database_single",
  "metadata": {"ticket_id": "INC-123", "team": "database"}
}
```

Changes to bound agents, prompts, profiles, behaviour settings or workflow topology
invalidate a saved composition. Restore its original configuration before resuming.
Bump an agent's `version`, a middleware registration's `version`, or the workflow
version when changing Python behaviour; configuration changes are detected directly.
Changing only the deployment's default selection does not reroute an admitted run.

Adapters can customize environment-specific wording through `RoleOverlay`:

```python
return RoleOverlay(
    prompt_blocks={"KUBECTL_GUIDANCE": "Use this environment's Kubernetes tool."},
    tool_descriptions={"exec_kubectl_cmd_safely": "Inspect resources in the incident namespace."},
)
```

`prompt_blocks` replaces recognized blocks during template composition, including
inside custom prompt templates. Recognized markers absent from a custom template
are ignored; misspelled markers are rejected. `system_replacement` still replaces
the whole composed prompt. `tool_descriptions` replaces the description only after
tool selection, so it cannot add a withheld tool or change its schema, dispatch or
authority. Unknown tool names and blank descriptions are rejected. Overrides are
per role and run. Completed workflow agent runs store effective description overrides
in metadata as `tool_description_overrides`; this field is not a record of tools
offered before a paused or failed run. Prompt digests cover the resulting prompt text.
The AIOpsLab adapter uses these hooks to describe its bridge without changing core
or other adapters' instructions.

Frozen research-arm files and historical prompt fixtures retain their original
bytes. The current default Execution prompt and Verify submission repairs adopt
later prototype behavior, as recorded in [the changelog](../CHANGELOG.md). Historical
composition tests replay the earlier Execution prompt through the override mechanism.
New customisations and migrated behavior must be recorded as experimental conditions;
these extension points do not constitute parity sign-off.
