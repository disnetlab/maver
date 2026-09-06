import pytest

from kira.settings.groups import ToolSettings
from kira.tools.kubernetes import (
    KubernetesPolicy,
    _append_delete_wait_false,
    _bound_blocking_wait,
    _reject,
    _selectors,
    classify,
    kubectl_authority,
    tokenize,
)

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
