import asyncio
from pathlib import Path

import pytest

from maver.audit.journal import Journal
from maver.settings.groups import ToolSettings
from maver.tools.context import ToolContext
from maver.tools.execution import bash_exec
from maver.tools.host import ssh_exec
from maver.tools.kubernetes import (
    KubernetesPolicy,
    _append_delete_wait_false,
    _bound_blocking_wait,
    _reject,
    _selectors,
    classify,
    exec_kubectl_cmd_safely,
    kubectl_authority,
    tokenize,
)
from maver.tools.workspace import SessionWorkspace

POLICY = KubernetesPolicy()


# --- classification, which the approval gate acts on -------------------------------


@pytest.mark.parametrize(
    ("cmd", "expected"),
    [
        ("kubectl get pods -n shop", "read"),
        ("kubectl delete deploy web -n shop", "write"),
        ("kubectl rollout status deploy/web", "read"),
        ("kubectl rollout undo deploy/web", "write"),
        ("kubectl exec -n shop pod -- cat /etc/hosts", "exec"),
        ("kubectl explain pod", "blocked"),
    ],
)
def test_the_common_shapes_classify_as_measured(cmd: str, expected: str) -> None:
    assert classify(tokenize(cmd), POLICY) == expected


@pytest.mark.parametrize(
    "cmd",
    [
        "kubectl frobnicate deploy web",  # unknown verb
        "kubectl --weird-flag get pods",  # unrecognised leading flag
        "kubectl rollout sideways deploy/web",  # unknown rollout subcommand
    ],
)
def test_an_unrecognised_shape_is_treated_as_a_write(cmd: str) -> None:
    """Everything the parser cannot vouch for is gated.

    The gate can only refuse what reaches it as a write. A shape that classifies as a
    read because nothing matched would skip the gate entirely, which is the one failure
    mode that matters here: it is silent, and it is a cluster mutation.
    """
    assert classify(tokenize(cmd), POLICY) == "write"


def test_a_verb_hidden_behind_global_flags_is_still_found() -> None:
    """Classification runs after the global flags, not on the first token."""
    assert classify(tokenize("kubectl -n shop delete deploy web"), POLICY) == "write"
    assert classify(tokenize("kubectl --request-timeout=5s get pods"), POLICY) == "read"


def test_a_credential_or_context_override_is_blocked_outright() -> None:
    """These re-point kubectl at another cluster or identity.

    Nothing the agent legitimately does needs them, and a command carrying one is not
    describable by the classification that the approval gate acts on.
    """
    for cmd in ("kubectl --context prod get pods", "kubectl --kubeconfig /tmp/k get pods"):
        assert classify(tokenize(cmd), POLICY) == "blocked"


def test_exec_is_its_own_class_and_not_a_read() -> None:
    """An in-container command cannot be proven read-only.

    Classified as a read it would skip the gate, which is what the prototype did. It is
    kept permitted so measured behaviour is unchanged, but naming it lets an operator
    hold or deny it by policy and lets the event log record that its authority was never
    provable.
    """
    assert classify(tokenize("kubectl exec pod -- ls"), POLICY) == "exec"


def test_tokenize_drops_the_binary_exactly_once() -> None:
    """A caller that slices again eats the verb, and everything then misclassifies."""
    assert tokenize("kubectl get pods") == ["get", "pods"]
    assert tokenize("get pods") == ["get", "pods"]


# --- screening, and the exact text the model gets ----------------------------------


def test_a_bare_shell_operator_is_refused_but_a_quoted_one_is_not() -> None:
    """kubectl runs without a shell, so an operator would be passed as an argument.

    Only bare operator tokens count: shlex keeps an operator inside quotes within one
    token, so a go-template's own pipe must survive untouched.
    """
    refusal = _reject(tokenize("kubectl get pods | grep web"), POLICY)
    assert refusal is not None
    assert refusal.startswith("error: shell operator '|' is not supported")

    template = "kubectl get pods -o go-template='{{range .items}}|{{end}}'"
    assert _reject(tokenize(template), POLICY) is None


def test_a_blocked_argument_is_reported_as_an_argument_not_as_the_verb() -> None:
    """Naming the wrong thing teaches the model the wrong lesson.

    The prototype reported the verb whatever the cause, so a refused argument told the
    model that `get` was banned and it stopped using a verb that was never the problem.
    """
    refusal = _reject(tokenize("kubectl get pod chaos-abc -n shop"), POLICY)
    assert refusal == "error: kubectl argument 'chaos-abc' is not permitted for the agent"

    verb_refusal = _reject(tokenize("kubectl explain pod"), POLICY)
    assert verb_refusal == "error: kubectl 'explain' is not permitted for the agent"


def test_interactive_flags_are_refused_including_combined_short_flags() -> None:
    for cmd in ("kubectl exec -it pod -- sh", "kubectl exec -ti pod -- sh", "kubectl run -i x"):
        refusal = _reject(tokenize(cmd), POLICY)
        assert refusal is not None
        assert "with interactive flags" in refusal


# --- the two argv normalisations, each from a real incident ------------------------


def test_a_blocking_wait_is_bounded_just_under_the_transport_timeout() -> None:
    """Unbounded, it blocks until the transport hard-kills the call.

    One wrong fix then turns every retry into a full stall. Capped, kubectl returns its
    own clean timeout first, which is informative and cheap to retry.
    """
    assert _bound_blocking_wait(tokenize("kubectl wait --for=condition=Ready pod/x"), 60)[-1] == (
        "--timeout=50s"
    )
    assert _bound_blocking_wait(tokenize("kubectl rollout status deploy/web"), 60)[-1] == (
        "--timeout=50s"
    )
    # An explicit timeout is respected, and a non-blocking verb is left alone.
    explicit = tokenize("kubectl wait --for=condition=Ready pod/x --timeout=5s")
    assert _bound_blocking_wait(explicit, 60) == explicit
    plain = tokenize("kubectl get pods")
    assert _bound_blocking_wait(plain, 60) == plain


def test_a_delete_never_blocks_on_finalizers() -> None:
    """A blocking delete on a finalizer-protected resource wedged one session's
    server-side processing for roughly three quarters of an hour, after which every
    subsequent kubectl in that session failed, reads included."""
    assert _append_delete_wait_false(tokenize("kubectl delete pvc data-0"))[-1] == "--wait=false"
    explicit = tokenize("kubectl delete pvc data-0 --wait=true")
    assert _append_delete_wait_false(explicit) == explicit
    plain = tokenize("kubectl get pods")
    assert _append_delete_wait_false(plain) == plain


def test_cluster_wide_reads_are_scoping_not_safety_and_can_be_turned_off() -> None:
    """Blocking -A is incident scoping. A real operator needs cluster-wide reads.

    Default on, so behaviour matches every published measurement.
    """
    assert classify(tokenize("kubectl get pods -A"), POLICY) == "blocked"
    open_policy = KubernetesPolicy(deny_all_namespaces=False)
    assert classify(tokenize("kubectl get pods -A"), open_policy) == "read"


def test_the_argument_screen_is_a_knob_and_says_which_kind_it_is() -> None:
    """It refuses reads and writes alike, before the verb is read, so a real cluster
    with a namespace matching it cannot be operated through this agent at all.

    That makes it an evaluation-integrity setting rather than a safety rule, and the
    prototype hard-coded it with no way to clear it. The default is the pattern every
    published measurement was taken under.
    """
    assert KubernetesPolicy.from_settings(ToolSettings()).denies_argument("chaos-mesh")

    lifted = KubernetesPolicy.from_settings(ToolSettings(kubectl_denied_argument_pattern=""))
    assert not lifted.denies_argument("chaos-mesh")

    renamed = KubernetesPolicy.from_settings(
        ToolSettings(kubectl_denied_argument_pattern="fault-injector")
    )
    assert renamed.denies_argument("FAULT-INJECTOR-0") and not renamed.denies_argument("chaos-mesh")


def test_the_gate_classifies_under_the_screens_this_deployment_actually_runs() -> None:
    """A classifier reading the shipped screens while the dispatch reads the
    deployment's disagree the moment an operator changes one — and disagree in the
    dangerous direction. A screen the deployment had lifted still classified as
    `blocked`, which is the one class the gate passes straight through on the grounds
    that the tool will refuse it. The tool no longer would.
    """
    write = {"cmd": "kubectl delete deployment chaos-controller -n chaos-mesh"}
    assert kubectl_authority(write, ToolSettings()) == "blocked"
    assert kubectl_authority(write, ToolSettings(kubectl_denied_argument_pattern="")) == "write"

    read = {"cmd": "kubectl get pods -A"}
    assert kubectl_authority(read, ToolSettings()) == "blocked"
    assert kubectl_authority(read, ToolSettings(kubectl_deny_all_namespaces=False)) == "read"


@pytest.mark.parametrize("arguments", [{}, {"cmd": ""}, {"cmd": "   "}, {"cmd": None}])
def test_a_call_with_no_readable_command_is_a_write(arguments: dict[str, object]) -> None:
    """The gate is deciding whether to ask a human. An argument shape the classifier
    cannot read is not evidence that nothing will be changed, and classifying it as a
    read sends it past the gate with the audit record saying a write was a read."""
    assert kubectl_authority(arguments) == "write"


# --- the facets a policy rule matches on -------------------------------------------


@pytest.mark.parametrize(
    ("cmd", "resource"),
    [
        # A kind whose singular ends in `s`, in both spellings, and its short name.
        ("kubectl delete ingress web -n shop", "ingress"),
        ("kubectl delete ingresses web -n shop", "ingress"),
        ("kubectl delete ing web -n shop", "ingress"),
        ("kubectl delete storageclass fast", "storageclass"),
        # A verb-local value flag written before the resource.
        ("kubectl patch -n shop --type merge -p {} secret db-creds", "secret"),
        ("kubectl delete -o name secret db-creds -n shop", "secret"),
        # The two other spellings of the same target.
        ("kubectl delete secret/db-creds -n shop", "secret"),
        ("kubectl delete -- secret db-creds -n shop", "secret"),
        # Nothing identifiable, which a catch-all rule can still name.
        ("kubectl delete -f manifest.yaml", "?"),
    ],
)
def test_a_resource_facet_names_the_target_not_a_flags_value(cmd: str, resource: str) -> None:
    """A rule is written against a kind, so every spelling of it has to reach one facet.

    Both halves failed open. The resource was read off the token after the verb, so
    `--type merge -p {}` put `merge` there; and `rstrip("s")` stripped every trailing
    `s`, so `ingress` became `ingre` and its plural became a third facet again. Each of
    these is a command kubectl accepts and runs.
    """
    assert f"resource:{resource}" in _selectors({"cmd": cmd})


def test_an_in_container_command_still_names_the_pod_it_runs_in() -> None:
    """`--` separates the container's command once the target has been named."""
    assert _selectors({"cmd": "kubectl exec -it web-0 -n shop -- sh"}) == (
        "verb:exec",
        "namespace:shop",
        "resource:web-0",
    )


# --- the kubectl guard the sandbox and the host tool share -------------------------


class _NoCluster:
    name = "none"
    capabilities = frozenset()
    timeout_seconds = 30


@pytest.mark.parametrize(
    "command",
    [
        "systemctl restart kubelet && kubectl delete ns shop",
        "systemctl restart kubelet\nkubectl delete ns shop",
        "echo $(kubectl get ns chaos-mesh -o yaml)",
        "echo `kubectl get secret db -o yaml`",
        "(k3s kubectl delete ns shop)",
    ],
)
async def test_kubectl_cannot_reach_a_shell_through_either_tool(
    command: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One guard, because the two copies of it disagreed about what a separator is.

    The host tool's copy saw `;`, `&` and `|` and neither a newline nor a subshell, and
    ssh hands the command to the node's login shell, where a second line is a second
    command run against a kubeconfig that is cluster-admin on a k3s host — outside every
    screen the kubectl tool applies and outside the ServiceAccount's RBAC.
    """

    def never_runs(*argv: object, **kwargs: object) -> None:
        raise AssertionError(f"a subprocess was started: {argv}")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", never_runs)
    ctx = ToolContext(
        transport=_NoCluster(),
        workspace=SessionWorkspace(tmp_path),
        role="execution",
        session_id="s1",
        journal=Journal(session_id="s1", stage="execution"),
        settings=ToolSettings(host_recovery_nodes=frozenset({"worker1"})),
    )

    assert (await ssh_exec(ctx, node="worker1", command=command)).startswith(
        "error: do not run kubectl through ssh_exec"
    )
    assert (await bash_exec(ctx, cmd=command)).startswith(
        "error: do not run kubectl through bash_exec"
    )


# --- the second enforcement point, inside the tool ---------------------------------


class _EchoCluster(_NoCluster):
    capabilities = frozenset({"kubernetes"})

    async def kubectl(self, argv: list[str]) -> str:
        return "DISPATCHED " + " ".join(argv)


def _context(tmp_path: Path, role: str, widened: frozenset[str] = frozenset()) -> ToolContext:
    return ToolContext(
        transport=_EchoCluster(),
        workspace=SessionWorkspace(tmp_path),
        role=role,
        session_id="s1",
        journal=Journal(session_id="s1", stage=role),
        widened_write_stages=widened,
    )


async def test_a_cluster_write_is_refused_inside_the_tool_and_not_only_at_the_gate(
    tmp_path: Path,
) -> None:
    """The second of the two enforcement points the documents promise.

    This tool is on every role's allowlist because the diagnostic roles read with it, so
    the allowlist is not the check the documents took it for, and `build_toolset` has no
    read/write logic. With the gate absent or misconfigured nothing else stood between a
    diagnosing role and a mutation: called directly as rca, it dispatched
    `kubectl delete deployment web -n shop`.
    """
    write = "kubectl delete deployment web -n shop"

    assert await exec_kubectl_cmd_safely(_context(tmp_path, "rca"), cmd=write) == (
        "error: kubectl writes are not permitted for the rca agent (read-only)"
    )
    assert "DISPATCHED" in await exec_kubectl_cmd_safely(_context(tmp_path, "execution"), cmd=write)
    # The reads the diagnostic roles hold the tool for are untouched.
    assert "DISPATCHED" in await exec_kubectl_cmd_safely(
        _context(tmp_path, "rca"), cmd="kubectl get pods -n shop"
    )

    # The operator's widening reaches the tool as well as the gate. Two knobs
    # disagreeing would refuse a write the gate had allowed, with nothing to say why.
    assert "DISPATCHED" in await exec_kubectl_cmd_safely(
        _context(tmp_path, "verify", frozenset({"verify"})), cmd=write
    )


# --- what ssh_exec may run at all --------------------------------------------------


def _on_a_worker(tmp_path: Path) -> ToolContext:
    """Execution, with one node reachable: the only shape in which ssh_exec runs."""
    return ToolContext(
        transport=_NoCluster(),
        workspace=SessionWorkspace(tmp_path),
        role="execution",
        session_id="s1",
        journal=Journal(session_id="s1", stage="execution"),
        settings=ToolSettings(host_recovery_nodes=frozenset({"worker1"})),
    )


async def test_ssh_exec_refuses_a_node_outside_the_allowlist_in_the_archive_s_words(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The refusal every archived call to this tool received, byte for byte.

    No measured campaign configured a node, so this string — and the empty allowlist
    rendering as "(none configured)" — is the whole of what the model ever saw back from
    ssh_exec. It is offered on an empty allowlist rather than withheld, so the refusal is
    now the only thing standing between the model and a node, and a reworded one would
    change a measured input.
    """

    def never_runs(*argv: object, **kwargs: object) -> None:
        raise AssertionError(f"a subprocess was started: {argv}")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", never_runs)
    nowhere = ToolContext(
        transport=_NoCluster(),
        workspace=SessionWorkspace(tmp_path),
        role="execution",
        session_id="s1",
        journal=Journal(session_id="s1", stage="execution"),
        settings=ToolSettings(),
    )

    assert await ssh_exec(nowhere, node="worker1", command="systemctl start kubelet") == (
        "error: host 'worker1' is not an allowed recovery node; configured: (none configured)"
    )
    assert await ssh_exec(
        _on_a_worker(tmp_path), node="worker2", command="systemctl start kubelet"
    ) == ("error: host 'worker2' is not an allowed recovery node; configured: worker1")


@pytest.mark.parametrize(
    "command",
    [
        "xargs kubectl get ns",
        "sh -c 'kubectl get ns'",
        "bash -lc 'kubectl get ns'",
        "eval kubectl get ns",
        "timeout 5 kubectl get ns",
        "env kubectl get ns",
        "nohup kubectl get ns",
        "sudo -u root kubectl get ns",
        "python3 -c \"import subprocess; subprocess.run(['kubectl', 'get', 'ns'])\"",
        "/usr/local/bin/collect-node-state.sh",
        "echo hi | xargs -I{} kubectl delete ns {}",
    ],
)
async def test_ssh_exec_refuses_a_command_that_is_not_a_host_binary(
    command: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Screening by position cannot close a shell, so the vocabulary is closed instead.

    Every line here was sent to the node: the guard looks for kubectl at a command
    position, and a wrapper puts it somewhere else. On a k3s host the login shell holds
    a cluster-admin kubeconfig, so each of these is a cluster call that the kubectl
    tool's screen, the approval gate's selectors and the ServiceAccount's RBAC never saw.
    """

    def never_runs(*argv: object, **kwargs: object) -> None:
        raise AssertionError(f"a subprocess was started: {argv}")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", never_runs)

    result = await ssh_exec(_on_a_worker(tmp_path), node="worker1", command=command)

    assert result.startswith("error:")
    assert "host-level commands only" in result


@pytest.mark.parametrize(
    "command",
    [
        "systemctl restart kubelet",
        "sudo systemctl start kubelet",
        "sudo systemctl restart containerd",
        "journalctl -u kubelet -n 50",
        "journalctl -u kubelet -n 200 | grep -i error",
        "df -h",
        "timeout 20 journalctl -u kubelet -n 50",
    ],
)
async def test_ssh_exec_still_runs_the_host_commands_it_exists_for(
    command: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The allowlist is a screen on this tool, not a retirement of it.

    `sudo systemctl start kubelet` is what two role prompts prescribe for a stopped
    kubelet, and a pipeline into `grep` is how its journal gets read.
    """
    sent: list[tuple[object, ...]] = []

    class _Ran:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return (b"ok", b"")

    async def spawn(*argv: object, **kwargs: object) -> _Ran:
        sent.append(argv)
        return _Ran()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)

    result = await ssh_exec(_on_a_worker(tmp_path), node="worker1", command=command)

    assert not result.startswith("error:")
    assert sent and sent[0][-1] == command
