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

The measured research arms keep their existing settings, prompts, tool order and
behaviour. New customisations are explicit experiment configurations and should be
recorded as such; these extension points do not constitute parity sign-off.
