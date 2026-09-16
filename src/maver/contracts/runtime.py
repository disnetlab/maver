"""Serializable incident state which is never implicitly exposed to a model.

Resources (clients, locks, tools) belong to RoleContext. This state contains only
JSON values and can be checkpointed across an approval or process restart.
"""

from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, JsonValue


class PromptAddition(BaseModel):
    """An explicit, auditable contribution to one recipient's model context."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    id: str = Field(default_factory=lambda: uuid4().hex)
    source: str = Field(min_length=1)
    recipient: str = Field(min_length=1)
    text: str = Field(min_length=1)
    lifetime: Literal["next_call", "invocation"] = "next_call"


class AgentState(BaseModel):
    """One logical invocation; a resumed audit attempt retains this identity."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    id: str
    definition: str
    stage: str
    role: str
    messages: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    middleware: dict[str, JsonValue] = Field(default_factory=dict)


class RunState(BaseModel):
    """Shared artifacts and private state, with conversations isolated by invocation."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    schema_version: Literal[1] = 1
    deployment_digest: str = ""
    selection: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    artifacts: dict[str, JsonValue] = Field(default_factory=dict)
    middleware: dict[str, JsonValue] = Field(default_factory=dict)
    agents: dict[str, AgentState] = Field(default_factory=dict)
    additions: list[PromptAddition] = Field(default_factory=list)
    steps: int = Field(default=0, ge=0)
    active_stage: str | None = None
    active_agent: str | None = None

    def add_context(
        self,
        text: str,
        *,
        source: str,
        recipient: str,
        lifetime: Literal["next_call", "invocation"] = "next_call",
    ) -> str:
        """Queue text for an instance ID or workflow stage; metadata stays private.

        A next-call addition is consumed after a successful model response. An
        invocation addition binds to the first matching instance and remains there.
        """
        addition = PromptAddition(text=text, source=source, recipient=recipient, lifetime=lifetime)
        self.additions.append(addition)
        return addition.id

    def context_for(self, agent: AgentState) -> tuple[PromptAddition, ...]:
        """Read eligible additions without consuming them before model success."""
        return tuple(a for a in self.additions if a.recipient in (agent.id, agent.stage))

    def delivered(self, agent: AgentState, additions: tuple[PromptAddition, ...]) -> None:
        """Consume one-shot entries, binding persistent entries to this invocation."""
        delivered = {a.id for a in additions}
        self.additions = [
            a for a in self.additions if a.id not in delivered or a.lifetime == "invocation"
        ]
        for addition in self.additions:
            if addition.id in delivered:
                addition.recipient = agent.id
