"""What a tool needs that the model did not supply.

Every dispatch takes this as its first argument. The prototype passed the equivalent
through four separate context variables, which made a tool impossible to call in a test
without a fixture that set all four, and made the set of things a tool could reach
invisible at its call site.

The context grows as the subsystems it carries land; it is deliberately built from what
exists rather than stubbed ahead of them.
"""

import logging
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from maver.audit.context import ToolObservation
from maver.audit.journal import Journal
from maver.memory.ports import RunStore, SessionStore
from maver.memory.retrieval import Retriever
from maver.settings.groups import ToolSettings
from maver.tools.scope import IncidentScope
from maver.tools.transport.base import ToolTransport
from maver.tools.workspace import SessionWorkspace

__all__ = [
    "DEFAULT_KUBECTL_ROUTE",
    "KUBECTL_ROUTES",
    "EvidencePolicy",
    "NullEvidencePolicy",
    "ToolContext",
    "deliver",
    "kubectl_proxy_url",
    "kubectl_route",
    "redact",
    "redact_query",
]

_log = logging.getLogger(__name__)

DEFAULT_KUBECTL_ROUTE = "tool"
"""The route with no proxy: the cluster tool, and a sandbox that refuses kubectl."""

KUBECTL_ROUTES: tuple[str, ...] = (DEFAULT_KUBECTL_ROUTE, "proxy_exec")
"""Every way a deployment's roles can reach the cluster.

The vocabulary is here because three layers spell it: the sandbox tools, the description
catalogue and the prompt loader. The loader keys its block maps by these names rather
than importing them, since prompt assets sit below the tool layer; the composition test
is what holds the two spellings together.
"""


@runtime_checkable
class EvidencePolicy(Protocol):
    """Removes from a tool result anything the environment should not have revealed.

    A benchmark can leak its own fault injection into the telemetry it serves — a
    chaos-injector's own labels appearing in the metrics an agent queries, for instance,
    which turns a diagnosis into a lookup. Filtering belongs to the adapter that knows
    its environment, not to the tools, which is why this is a port and not a function.
    """

    def filter_result(self, content: str) -> str:
        """Redact anything the model should not see.

        Args:
            content: A tool result.

        Returns:
            The result to hand to the model.
        """
        ...

    def filter_query(self, query: str) -> str:
        """Rewrite a query so it cannot reach what the environment is hiding.

        Filtering a result is not enough on its own: a model that can *ask* for a
        hidden namespace learns it exists from the shape of the answer. The rewrite
        and the redaction are the same policy seen from two directions, so they belong
        to the same object.

        Args:
            query: The query the model wrote.

        Returns:
            The query to actually run.
        """
        ...


class NullEvidencePolicy:
    """Filters nothing. The default, for any environment with nothing to hide."""

    def filter_result(self, content: str) -> str:
        """Return the content unchanged.

        Args:
            content: A tool result.

        Returns:
            The same content.
        """
        return content

    def filter_query(self, query: str) -> str:
        """Return the query unchanged.

        Args:
            query: The query the model wrote.

        Returns:
            The same query.
        """
        return query


def redact(policy: EvidencePolicy, content: str) -> str:
    """Apply an evidence policy, failing open.

    Deliberately the opposite of how an authority failure is handled. A policy that
    cannot answer has not authorised anything, so a write is refused; but a filter that
    raises has not found anything to hide, and turning a good tool result into an error
    would cost the role a real observation over a bug in an adapter's redaction.

    Args:
        policy: The evidence policy.
        content: A tool result.

    Returns:
        The filtered result, or the original when filtering failed.
    """
    try:
        return policy.filter_result(content)
    except Exception as exc:  # a redaction bug must not destroy the observation
        _log.warning("evidence policy failed and the result was passed through: %s", exc)
        return content


def deliver(ctx: "ToolContext", content: str, label: str, limit: int) -> str:
    """Hand a tool result to the model: filtered, then bounded.

    One function, because the alternative was thirteen call sites each remembering to
    apply the policy and ten of them not doing it. Every telemetry tool — metrics,
    alerts, logs, labels, services, operations, traces, the dependency graph — saved its
    result straight to the workspace, so the adapter's hide list reached kubectl and
    nothing else. The policy's own description names a chaos injector's labels in the
    metrics an agent queries as the thing it exists to remove, and that was the one place
    it was never applied.

    Under the benchmark transport the harness proxy filters these paths as well, so what
    this changes there is a second line of defence. Under the direct transport there is
    no proxy, and this was the only one.

    Args:
        ctx: The run's tool context.
        content: The raw result.
        label: The workspace name to save an oversized result under.
        limit: The size above which the result is saved rather than returned.

    Returns:
        The filtered result, or a pointer to where the whole of it was saved.
    """
    return ctx.workspace.save_tool_output(redact(ctx.evidence, content), label, limit)


def redact_query(policy: EvidencePolicy, query: str) -> str:
    """Apply an evidence policy to a query, failing **closed**.

    The mirror of :func:`redact`, and closed rather than open on purpose: a query that
    was not rewritten reaches somewhere the environment meant to hide, and a model that
    can ask about a hidden namespace learns it exists from the shape of the answer.

    Args:
        policy: The evidence policy.
        query: The query the model wrote.

    Returns:
        The rewritten query.

    Raises:
        Exception: Whatever the policy raised. The caller turns it into a tool error.
    """
    return policy.filter_query(query)


def kubectl_proxy_url(transport: ToolTransport) -> str | None:
    """The credential-free endpoint a local kubectl can reach this cluster through.

    Read with a default rather than as a plain attribute. Transports arrive through an
    entry point, so one built outside this repository need not carry the property at
    all, and the safe reading of a transport that does not answer is that it offers no
    proxy — which leaves the sandbox tools refusing kubectl exactly as they do today.

    Args:
        transport: The run's bound transport.

    Returns:
        The proxy's base URL, or None when this environment has no proxy route.
    """
    url = getattr(transport, "kubectl_proxy_url", None)
    return str(url) if url else None


def kubectl_route(transport: ToolTransport) -> str:
    """How roles in this deployment reach the cluster.

    One question answered in one place, because three of them ask it and a disagreement
    is silent: the sandbox tools decide whether to refuse kubectl, the description
    catalogue decides which text the model is shown, and the prompt loader decides which
    guidance block it composes. Two of the three saying ``tool`` while the third says
    ``proxy_exec`` is a model told the sandbox has no credentials while its kubectl
    works, which costs it turns on refusals it does not believe.

    Args:
        transport: The run's bound transport.

    Returns:
        ``"proxy_exec"`` when the transport offers a proxy, ``"tool"`` otherwise.
    """
    return "proxy_exec" if kubectl_proxy_url(transport) is not None else "tool"


@dataclass(frozen=True)
class ToolContext:
    """One role's run, from a tool's point of view."""

    transport: ToolTransport
    workspace: SessionWorkspace
    role: str
    """The calling role. This is the stage the approval gate keys on."""
    session_id: str
    journal: Journal
    scope: IncidentScope = field(default_factory=IncidentScope)
    """What the alert said about this incident, and what has been discovered about it.

    Trusted context rather than a tool argument: the telemetry wrappers read the
    namespace to keep one application's evidence out of another's incident, and a value
    the model could set would defeat that.
    """
    settings: ToolSettings = field(default_factory=ToolSettings)
    """Tool-layer limits, injected rather than read from the process.

    A tool that reaches for global settings behaves differently depending on process
    state its call site cannot see, which is how the prototype ended up with five
    environment reads hidden inside its toolset builder.
    """
    authority_mode: str = "prompt"
    """What the approval gate does with a write in this deployment: ``auto``, ``prompt``
    or ``deny``.

    Injected for the same reason as the settings above, and needed by exactly one
    caller: the proxy route puts a real kubectl on a role's PATH, and a write typed
    there does not pass through the gate at all. The shim on that PATH refuses one
    unless this says the gate would have approved it unattended, so the mode has to
    reach the subprocess environment. Defaults to the fail-closed mode, matching
    ``authority.mode``, so a context built without it refuses shell writes.
    """
    kubectl_binary: str = "kubectl"
    """The real kubectl the proxy route execs, from ``transport.kubectl_binary``.

    Carried here rather than looked up, because the shim this puts on PATH is itself
    named ``kubectl`` and a lookup from inside it finds the shim.
    """
    widened_write_stages: frozenset[str] = frozenset()
    """Stages this deployment opened writes to, beyond each tool's own declared set.

    Injected for the same reason the tool settings are. A tool screening itself against
    an independent read of the environment would refuse a stage the gate was built to
    allow, which is two knobs disagreeing with nothing to say why.
    """
    evidence: EvidencePolicy = field(default_factory=NullEvidencePolicy)
    memory: Retriever | None = None
    """The experience store, or None when this configuration has none."""
    sessions: SessionStore | None = None
    runs: RunStore | None = None
    """Role runs and their traces, when this deployment records them."""
    memory_exclude: tuple[str, ...] = ()
    """Session ids retrieval must withhold, supplied per run by an evaluation harness."""
    observation: ToolObservation | None = None
    """Audit-only measurements for the current tool attempt."""
    memory_aliases: dict[str, str] = field(default_factory=dict)
    """Maps the short handles shown to the model back to real session ids.

    Per run rather than global: a handle means "the first result of *my* search", and
    two concurrent incidents would otherwise resolve each other's.
    """
