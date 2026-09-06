import re
from collections.abc import Iterator
from importlib.resources import files
from typing import Any

import pytest
import yaml

from kira.prompts.loader import compose_role_prompt
from kira.settings.root import known_env_names
from kira.tools.allowlists import RETRIEVAL_TOOLS, ROLE_TOOLSETS
from kira.tools.registry import ToolSpec, validate_registry

DESCRIPTIONS: dict[str, Any] = yaml.safe_load(
    files("kira.tools").joinpath("descriptions.yaml").read_text(encoding="utf-8")
)


def stand_in(name: str) -> ToolSpec:
    """A spec that stands in for the real one, so the catalogue can be checked alone."""
    if name.startswith("submit_"):
        return ToolSpec(name=name, parameters={"type": "object"}, safety="submit")

    async def fn(ctx: object, **_: object) -> str:
        return "ok"

    return ToolSpec(
        name=name,
        parameters={"type": "object", "properties": {}},
        safety="read_only",
        dispatch=fn,
    )


def test_the_shipped_catalogue_satisfies_every_registry_rule() -> None:
    """Descriptions and allowlists are two hand-edited files that must agree.

    Every failure mode here is silent at incident time: a tool with no description
    reaches the model uninstructed, and a description written for a role that cannot
    call the tool means the allowlist and the catalogue disagree about the design while
    each looks correct on its own. The prototype shipped one of these — a role name
    typed beside ``default`` instead of under ``agents`` — for the life of the system.

    The registry here is built from the description keys rather than from the live one,
    so this checks the two files against each other. The full bijection against the
    registered tools is enforced at import of ``kira.tools`` once the capability modules
    that declare them land.
    """
    registry = {name: stand_in(name) for name in DESCRIPTIONS}
    validate_registry(descriptions=DESCRIPTIONS, allowlists=ROLE_TOOLSETS, registry=registry)


SETTING_TOKEN = re.compile(r"\b([A-Z][A-Z0-9_]{5,})=")
"""An assignment to a shouting name: how a description spells a setting."""


def shipped_descriptions() -> Iterator[tuple[str, str]]:
    """Every description string the model can be shown, with where it came from."""
    for tool, entry in DESCRIPTIONS.items():
        yield tool, entry["default"]
        for role, text in (entry.get("agents") or {}).items():
            yield f"{tool}/{role}", text


def test_no_description_names_a_setting_the_system_does_not_have() -> None:
    """Descriptions ship to the model, so a stale setting name is a stale instruction.

    ``bash_exec`` and ``code_exec`` branched on ``KUBECTL_MODE=proxy_exec`` for a mode
    the rebuild dropped, telling the model kubectl works in the sandbox while the
    kubectl block in the same request told it the sandbox has no credentials. The
    prompt blocks lost the mode axis and the descriptions did not, because nothing
    checked one hand-edited file against the settings model.

    This only catches a setting written as an assignment, which is how the drift was
    written. A setting named in prose still gets past it.
    """
    accepted = known_env_names()
    named = [
        (where, name)
        for where, text in shipped_descriptions()
        for name in SETTING_TOKEN.findall(text)
        if name not in accepted
    ]
    assert named == []


TOOL_SHAPED = re.compile(r"`([a-z][a-z0-9]*(?:_[a-z0-9]+)+)`")
"""A backticked snake_case word: how a prompt names a tool it wants called."""

NOT_A_TOOL = frozenset(
    {
        # Submission fields and tool arguments, which prompts backtick the same way.
        "affected_resource",
        "attempted_action",
        "attempted_hypothesis",
        "drift_detected",
        "entry_point_status",
        "expected_outcome",
        "fault_category",
        "fault_type",
        "final_state_summary",
        "needs_replan",
        "observed_outcome",
        "reassessment_reason",
        "replan_reason",
        "save_as",
        # The learning trigger.
        "post_incident",
    }
)

NAMED_WITHOUT_BEING_OFFERED = {
    # Planning does not run the command; it writes the step that names it, and
    # Execution is the role holding the tool.
    ("planning", "ssh_exec"),
    # Learning receives the shared read-only kubectl block, which describes the two
    # sandbox tools, while Learning's own Tools section says it has no shell or
    # Python. The block is shared with RCA and Verify, which do hold them.
    ("learning", "bash_exec"),
    ("learning", "code_exec"),
    # Learning's Tools section names the retrieval tools unconditionally, so a
    # composition with retrieval off prescribes three tools it is not offered.
    ("learning", "memory_search"),
    ("learning", "open_memory"),
    ("learning", "session_trace_summary"),
    # Supplied by an environment adapter rather than by a role allowlist, so the
    # core cannot see them from here.
    ("single", "rollback_command"),
}
"""Known cases of a role prompt naming a tool that role cannot call, with the reason.

Listed rather than fixed: each needs a prompt change with a parity decision behind it.
An entry is an admission, not an exemption to reach for.
"""


@pytest.mark.parametrize("role", sorted(ROLE_TOOLSETS))
@pytest.mark.parametrize("retrieval", [True, False], ids=["memory-on", "memory-off"])
def test_a_role_prompt_only_names_tools_that_role_is_offered(role: str, retrieval: bool) -> None:
    """A prompt that prescribes a tool the model was never given costs turns and tokens.

    The Learning prompt prescribed four ``playbook_*`` tools through every distillation
    pass of every experience-enabled configuration. They had been deleted with the
    playbook layer; the prompt was migrated byte-exact and nothing rechecked it against
    the allowlist. The model could only spend turns on a phantom call or ignore the
    instruction.

    This reads the composed system prompt, so it does not see the per-turn runtime
    assets a role appends to it.
    """
    offered = set(ROLE_TOOLSETS[role])
    if not retrieval:
        offered -= RETRIEVAL_TOOLS
    prompt = compose_role_prompt(role, retrieval_enabled=retrieval)
    prescribed = {
        name
        for name in TOOL_SHAPED.findall(prompt)
        if name not in NOT_A_TOOL and (role, name) not in NAMED_WITHOUT_BEING_OFFERED
    }
    assert prescribed <= offered
