"""The deploy artifacts, exercised rather than reviewed.

Nothing here needs a cluster. Both overlays are rendered with ``kubectl kustomize``,
which is pure local templating, and everything else is read out of the repository and
compared with the code it is supposed to describe.

The reason this file exists is that a manifest is only checked when someone applies it,
and by then the failure is a crash loop on somebody else's cluster. Three shapes of that
are pinned here, each of which had happened:

* A ``MAVER_``-prefixed key the settings model does not recognise is a startup failure by
  design, so a stale ConfigMap key does not run with a silent default, it refuses to
  start. The overlay is therefore validated by *building the settings*, not by matching
  names: a value the validators refuse — a production overlay that also approves its own
  cluster writes — fails here rather than at the first pod.
* A probe on a path the API does not serve restarts a healthy pod forever. The paths are
  compared with the routes ``maver.api.health`` actually declares.
* RBAC that forbids a verb the kubectl tool can issue surfaces mid-incident as a
  Forbidden the model reads as a fault to work around. Both sides of that comparison are
  derived — the verbs and resource kinds from ``maver.tools.kubernetes``, the grants from
  the YAML — so a verb or kind added to the tool fails this file until somebody decides
  what it costs in RBAC.
"""

import os
import re
import shutil
import subprocess
from collections.abc import Iterator, Mapping
from functools import cache, lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pytest
import yaml

from maver.api.health import router as health_router
from maver.settings import Settings
from maver.settings.root import unknown_prefixed_names
from maver.tools.kubernetes import (
    _KINDS,
    _READ_VERBS,
    _RESOURCE_ALIASES,
    _WRITE_VERBS,
)

REPO = Path(__file__).resolve().parents[2]
DEPLOY = REPO / "deploy"
K8S = DEPLOY / "k8s"
DOCKER = DEPLOY / "docker"
OVERLAYS = ("local", "production")


# --------------------------------------------------------------------------- rendering


def _kubectl() -> str:
    """Locate kubectl, or say why these checks cannot run.

    A skip reports green, so in CI — where the workflow installs kubectl for exactly
    this file — a missing binary is a failure instead. Otherwise a contributor without
    kubectl still gets a usable ``make check``.
    """
    binary = shutil.which("kubectl")
    if binary:
        return binary
    if os.environ.get("CI"):
        pytest.fail(
            "kubectl is not on PATH in CI, so the deploy manifests went unrendered and "
            "this file passed having checked nothing. .github/workflows/ci.yml installs "
            "it; that step is what keeps this from being a skip."
        )
    pytest.skip("kubectl is not on PATH, and these checks render the overlays with it")


@lru_cache(maxsize=len(OVERLAYS))
def _render(overlay: str) -> tuple[dict[str, Any], ...]:
    """Render one overlay. Pure templating: kustomize contacts no API server."""
    out = subprocess.run(
        [_kubectl(), "kustomize", str(K8S / "overlays" / overlay)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return tuple(doc for doc in yaml.safe_load_all(out) if doc)


def _of_kind(docs: tuple[dict[str, Any], ...], kind: str) -> list[dict[str, Any]]:
    return [doc for doc in docs if doc["kind"] == kind]


def _one(docs: tuple[dict[str, Any], ...], kind: str) -> dict[str, Any]:
    found = _of_kind(docs, kind)
    assert len(found) == 1, f"expected exactly one {kind}, found {len(found)}"
    return found[0]


def _container(docs: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    containers = _one(docs, "Deployment")["spec"]["template"]["spec"]["containers"]
    assert len(containers) == 1
    return containers[0]


# ----------------------------------------------------------------- the container's env

_ENV_LINE = re.compile(r"^\s*ENV\s+(?P<rest>.*)$")


def _dockerfile_env() -> dict[str, str]:
    """Read the prefixed settings the runtime image bakes in.

    Derived from the Dockerfile rather than restated, because these are half of what the
    pod actually resolves: ``MAVER_SERVER__ENVIRONMENT=production`` lives here, and it is
    what makes an overlay that also sets ``authority.mode=auto`` refuse to start.
    """
    lines = (DOCKER / "Dockerfile").read_text().splitlines()
    runtime = lines[next(i for i, line in enumerate(lines) if line.endswith("AS runtime")) :]
    joined: list[str] = []
    buffer = ""
    for line in runtime:
        buffer = f"{buffer} {line.rstrip()}" if buffer else line.rstrip()
        if buffer.endswith("\\"):
            buffer = buffer[:-1]
            continue
        joined.append(buffer)
        buffer = ""
    found: dict[str, str] = {}
    for statement in joined:
        match = _ENV_LINE.match(statement)
        if not match:
            continue
        for token in match["rest"].split():
            if "=" not in token:
                continue
            name, _, value = token.partition("=")
            if name.startswith("MAVER_"):
                found[name] = value.strip('"').replace("${REVISION}", "0000000")
    return found


def _pod_env(overlay: str) -> dict[str, str]:
    """Everything the container would resolve: image ENV, then ConfigMap, then Secret.

    In the order Kubernetes applies them, so what this validates is the environment the
    process starts on rather than one file out of the three that produce it.
    """
    docs = _render(overlay)
    sources = {
        doc["metadata"]["name"]: doc.get("data", {}) | doc.get("stringData", {})
        for doc in docs
        if doc["kind"] in {"ConfigMap", "Secret"}
    }
    # The Secret is a template outside kustomization.yaml on purpose, so nothing renders
    # it; its keys are still part of the environment a real deployment has.
    secret = yaml.safe_load((K8S / "base" / "secret.example.yaml").read_text())
    sources[secret["metadata"]["name"]] = secret["stringData"]

    env = dict(_dockerfile_env())
    container = _container(docs)
    for source in container.get("envFrom", []) or []:
        reference = source.get("configMapRef") or source.get("secretRef")
        env.update(sources.get(reference["name"], {}))
    for entry in container.get("env", []) or []:
        if "value" in entry:
            env[entry["name"]] = entry["value"]
    return {name: str(value) for name, value in env.items()}


def _compose_env(filename: str) -> dict[str, str]:
    """The environment a compose service starts on, from the example env file it ships."""
    service = next(iter(yaml.safe_load((DOCKER / filename).read_text())["services"].values()))
    env = dict(_dockerfile_env())
    for name in service.get("env_file", []) or []:
        # maver.env is the operator's own copy and is gitignored; the example beside it is
        # what this repository ships and therefore what can be checked.
        example = DOCKER / f"{name}.example"
        for line in example.read_text().splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and "=" in stripped:
                name_, _, value = stripped.partition("=")
                env[name_.strip()] = value.strip()
    for name, value in (service.get("environment") or {}).items():
        env[name] = str(value)
    return env


@pytest.fixture
def build_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Build Settings from exactly one environment, the way the container would.

    Three paths are redirected to somewhere that exists on this machine: the workspace,
    the model profiles and the arm files all live at image paths. Everything else is the
    manifest's own value, which is the point.
    """

    def build(env: Mapping[str, str]) -> Settings:
        monkeypatch.chdir(tmp_path)  # no developer's .env underneath the assertion
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        monkeypatch.setenv("MAVER_SERVER__WORKDIR", str(tmp_path / "workspace"))
        monkeypatch.setenv("MAVER_MODEL__PROFILE_DIR", str(REPO / "configs" / "models"))
        monkeypatch.setenv("MAVER_ARM_DIR", str(REPO / "configs" / "arms"))
        return Settings()

    return build


# ------------------------------------------------------------------------- the overlays


@pytest.mark.parametrize("overlay", OVERLAYS)
def test_every_rendered_key_is_a_setting(overlay: str) -> None:
    """A prefixed name the model does not know stops the process at startup.

    That is deliberate — a misspelling is otherwise indistinguishable from an unset
    value — and it makes a stale key in a ConfigMap a pod that never starts. The sweep
    below is the one ``load_settings`` runs, against the rendered environment.
    """
    env = _pod_env(overlay)
    assert [name for name in env if name.startswith("MAVER_")], "no prefixed keys rendered"
    assert unknown_prefixed_names(env) == []


@pytest.mark.parametrize("overlay", OVERLAYS)
def test_settings_accept_the_overlay(overlay: str, build_settings) -> None:
    """The validators, not just the names.

    A key can be spelled correctly and still be refused: the settings layer rejects
    ``server.environment=production`` together with ``authority.mode=auto``, and refuses
    to start with no intake enabled. Both are configurations an overlay could ship.
    """
    settings = build_settings(_pod_env(overlay))
    assert settings.intake.http_enabled
    assert settings.model.profile


def test_production_overlay_does_not_approve_its_own_writes() -> None:
    """The one combination the settings layer refuses, stated as its own check.

    Building the settings above already covers it, but a later reader should be able to
    see that this is a property of the shipped overlay rather than an accident of what
    the base happens to set.
    """
    env = _pod_env("production")
    assert env["MAVER_SERVER__ENVIRONMENT"] == "production"
    assert env["MAVER_AUTHORITY__MODE"] != "auto"


def test_production_keeps_every_base_configmap_key() -> None:
    """An overlay is a merge; a `-f` apply of the patch is not.

    The rendered ConfigMap silently losing a base key is the failure the README's first
    trap describes, and it reads as a pod running something other than what was asked.
    """
    base = yaml.safe_load((K8S / "base" / "configmap.yaml").read_text())["data"]
    rendered = _one(_render("production"), "ConfigMap")["data"]
    assert set(base) <= set(rendered)


def test_local_overlay_unsets_the_telemetry_it_has_none_of() -> None:
    """A removed endpoint withdraws its tools; a wrong one fails at the first call."""
    rendered = _one(_render("local"), "ConfigMap")["data"]
    for name in (
        "MAVER_TRANSPORT__PROMETHEUS_URL",
        "MAVER_TRANSPORT__LOKI_URL",
        "MAVER_TRANSPORT__JAEGER_URL",
        "MAVER_OBSERVABILITY__OTLP_ENDPOINT",
    ):
        assert name not in rendered, f"{name} survived the local overlay's null"


@pytest.mark.parametrize("filename", ["docker-compose.yml", "docker-compose.eval.yml"])
def test_compose_services_start_on_their_example_environment(filename, build_settings) -> None:
    """The evaluation service sets authority=auto, and the image bakes in production.

    That pair is refused, so the container exited at startup and the harness saw a
    health-check timeout naming nothing. It is a compose file rather than a manifest and
    is checked here for the same reason: nothing else reads it until a sweep is running.
    """
    env = _compose_env(filename)
    assert unknown_prefixed_names(env) == []
    build_settings(env)


def test_eval_compose_keeps_the_harness_contract() -> None:
    """Three names the evaluation harness hard-codes, and one port it polls."""
    service = yaml.safe_load((DOCKER / "docker-compose.eval.yml").read_text())["services"][
        "maver-eval"
    ]
    assert service["image"] == "k8s-agent:local"
    assert service["container_name"] == "k8s-agent-sregym"
    assert service["ports"] == ["127.0.0.1:8080:8080"]


# ---------------------------------------------------------------------------- the probes

HEALTH_ROUTES = {route.path for route in health_router.routes}  # type: ignore[attr-defined]


@pytest.mark.parametrize("overlay", OVERLAYS)
def test_probes_name_routes_the_api_serves(overlay: str) -> None:
    """A probe on a path nothing serves restarts a pod that is perfectly healthy.

    Liveness and readiness are also not interchangeable: liveness must stay on the
    dependency-free endpoint, or an outage in the model endpoint or the database becomes
    a restart loop, and readiness must be the one that reads the store, or a draining pod
    keeps receiving incidents.
    """
    container = _container(_render(overlay))
    paths = {
        probe: container[probe]["httpGet"]["path"]
        for probe in ("startupProbe", "readinessProbe", "livenessProbe")
    }
    assert set(paths.values()) <= HEALTH_ROUTES, f"{paths} vs {HEALTH_ROUTES}"
    assert paths["livenessProbe"] == "/healthz"
    assert paths["startupProbe"] == "/healthz"
    assert paths["readinessProbe"] == "/readyz"
    for probe in paths:
        assert container[probe]["httpGet"]["port"] == "http"


@pytest.mark.parametrize(
    "path",
    [
        DOCKER / "Dockerfile",
        DOCKER / "docker-compose.yml",
        DOCKER / "docker-compose.eval.yml",
    ],
    ids=lambda path: path.name,
)
def test_container_healthchecks_name_a_route(path: Path) -> None:
    """Docker's own health check answers "can it take work", so it is the readiness one."""
    urls = set(re.findall(r"http://127\.0\.0\.1:8080(/\w+)", path.read_text()))
    assert urls == {"/readyz"}, urls
    assert urls <= HEALTH_ROUTES


# ------------------------------------------------------------------------------ the RBAC

# Kubernetes' own kind to (apiGroup, plural). A statement about the API, not about what
# this repository grants; the decision half is DECIDED_UNGRANTED below.
KIND_API: dict[str, tuple[str, str]] = {
    "clusterrole": ("rbac.authorization.k8s.io", "clusterroles"),
    "clusterrolebinding": ("rbac.authorization.k8s.io", "clusterrolebindings"),
    "configmap": ("", "configmaps"),
    "cronjob": ("batch", "cronjobs"),
    "daemonset": ("apps", "daemonsets"),
    "deployment": ("apps", "deployments"),
    "endpoints": ("", "endpoints"),
    "event": ("", "events"),
    "horizontalpodautoscaler": ("autoscaling", "horizontalpodautoscalers"),
    "ingress": ("networking.k8s.io", "ingresses"),
    "ingressclass": ("networking.k8s.io", "ingressclasses"),
    "job": ("batch", "jobs"),
    "namespace": ("", "namespaces"),
    "networkpolicy": ("networking.k8s.io", "networkpolicies"),
    "node": ("", "nodes"),
    "persistentvolume": ("", "persistentvolumes"),
    "persistentvolumeclaim": ("", "persistentvolumeclaims"),
    "pod": ("", "pods"),
    "poddisruptionbudget": ("policy", "poddisruptionbudgets"),
    "priorityclass": ("scheduling.k8s.io", "priorityclasses"),
    "replicaset": ("apps", "replicasets"),
    "role": ("rbac.authorization.k8s.io", "roles"),
    "rolebinding": ("rbac.authorization.k8s.io", "rolebindings"),
    "secret": ("", "secrets"),
    "service": ("", "services"),
    "serviceaccount": ("", "serviceaccounts"),
    "statefulset": ("apps", "statefulsets"),
    "storageclass": ("storage.k8s.io", "storageclasses"),
}

DECIDED_UNGRANTED: dict[str, str] = {
    "secret": "reading one means reading its values, including ServiceAccount tokens",
    "role": "something that can write a binding can grant itself everything else",
    "rolebinding": "something that can write a binding can grant itself everything else",
    "clusterrole": "cluster-scoped RBAC, the same escalation one level up",
    "clusterrolebinding": "cluster-scoped RBAC, the same escalation one level up",
}
"""Kinds the tool can name and RBAC deliberately refuses, each with what it would cost.

Listed so that the coverage check below can tell a decision from an omission. Adding a
kind here is how somebody says "on purpose"; the API server then refuses it with a
message the model can read.
"""

# One kubectl verb to the RBAC verbs issuing it needs. Keyed off the tool's own verb
# tables, so a verb added to maver.tools.kubernetes without a decision about what it costs
# in RBAC fails the check below rather than surfacing as a Forbidden mid-incident.
VERB_RBAC: dict[str, frozenset[str]] = {
    # reads
    "describe": frozenset({"get", "list"}),
    "events": frozenset({"get", "list"}),
    "get": frozenset({"get", "list", "watch"}),
    "logs": frozenset({"get"}),
    "top": frozenset({"get", "list"}),
    "version": frozenset(),  # /version, which is not an RBAC resource
    "wait": frozenset({"get", "watch"}),
    # writes
    "annotate": frozenset({"get", "patch"}),
    "apply": frozenset({"get", "create", "patch", "update"}),
    "autoscale": frozenset({"get", "create"}),
    "cordon": frozenset({"get", "patch"}),
    "create": frozenset({"create"}),
    "delete": frozenset({"get", "list", "delete"}),
    "drain": frozenset({"get", "list", "patch"}),
    "edit": frozenset({"get", "patch"}),
    "expose": frozenset({"get", "create"}),
    "label": frozenset({"get", "patch"}),
    "patch": frozenset({"get", "patch"}),
    "replace": frozenset({"get", "update"}),
    "rollout": frozenset({"get", "list", "patch"}),
    "scale": frozenset({"get", "patch", "update"}),
    "set": frozenset({"get", "patch"}),
    "taint": frozenset({"get", "patch"}),
    "uncordon": frozenset({"get", "patch"}),
}

NODE_SCOPED_WRITES = frozenset({"cordon", "uncordon", "drain", "taint"})
"""Writes granted nowhere, because they act on a node.

A fault injected as a cordon is one to diagnose rather than undo, and node writes also
grant evicting every pod on every node. Excluded from the namespaced coverage check for
that reason and no other.
"""


@cache
def _grants() -> dict[str, dict[tuple[str, str], frozenset[str]]]:
    """Read every base ClusterRole into role -> (apiGroup, resource) -> verbs."""
    roles: dict[str, dict[tuple[str, str], set[str]]] = {}
    for path in sorted((K8S / "base").glob("rbac-*.yaml")):
        for doc in yaml.safe_load_all(path.read_text()):
            if not doc or doc["kind"] != "ClusterRole":
                continue
            rules = roles.setdefault(doc["metadata"]["name"], {})
            for rule in doc["rules"]:
                for group in rule["apiGroups"]:
                    for resource in rule["resources"]:
                        rules.setdefault((group, resource), set()).update(rule["verbs"])
    return {
        name: {key: frozenset(verbs) for key, verbs in rules.items()}
        for name, rules in roles.items()
    }


def _granted(group: str, resource: str) -> frozenset[str]:
    """Every verb any base role grants on one resource."""
    found: set[str] = set()
    for rules in _grants().values():
        found |= rules.get((group, resource), frozenset())
    return frozenset(found)


def _tool_kinds() -> Iterator[str]:
    """The resource kinds the kubectl tool's own vocabulary names."""
    yield from sorted(set(_KINDS) | set(_RESOURCE_ALIASES.values()))


def test_every_kubectl_verb_has_an_rbac_decision() -> None:
    """A verb added to the classifier must come with what it costs in RBAC."""
    assert set(VERB_RBAC) == set(_READ_VERBS) | set(_WRITE_VERBS)


@pytest.mark.parametrize("kind", list(_tool_kinds()))
def test_every_kind_the_tool_names_is_granted_or_decided(kind: str) -> None:
    """The tool can name it, so RBAC either allows it or refuses it on purpose.

    Both sides are derived: the kinds from the tool's ``_KINDS`` and its short-name
    table, the grants from the rule files. A kind added to one and not the other is
    caught here rather than as a Forbidden the model works around.
    """
    group, plural = KIND_API[kind]
    verbs = _granted(group, plural)
    if kind in DECIDED_UNGRANTED:
        assert not verbs, f"{kind} is listed as ungranted by design but has {sorted(verbs)}"
        return
    assert verbs, f"the kubectl tool can name {kind!r} and no base role grants it"


def test_the_read_roles_carry_every_verb_a_read_needs() -> None:
    """get, list and watch, derived from the read verbs the tool can issue."""
    needed = frozenset().union(*(VERB_RBAC[verb] for verb in _READ_VERBS))
    for role in ("maver-cluster-read", "maver-namespace-read"):
        rules = _grants()[role]
        # The metrics API serves no watch, and kubectl top does not ask for one.
        core = {key: verbs for key, verbs in rules.items() if key[0] != "metrics.k8s.io"}
        for key, verbs in core.items():
            if key == ("", "namespaces"):
                continue  # get by name only, never list: see rbac-cluster-read.yaml
            assert needed <= verbs, f"{role} {key} grants {sorted(verbs)}, needs {sorted(needed)}"


def test_the_write_role_carries_every_verb_a_write_needs() -> None:
    """Derived from the write verbs, minus the node-scoped ones granted nowhere.

    Checked against the rules that already mutate, so a rule deliberately kept read-only
    — the disruption budgets, the revision history — is not read as a missing grant.
    """
    needed = frozenset().union(
        *(VERB_RBAC[verb] for verb in set(_WRITE_VERBS) - NODE_SCOPED_WRITES)
    )
    rules = _grants()["maver-namespace-write"]
    mutating = {key: verbs for key, verbs in rules.items() if "create" in verbs}
    assert mutating, "the write role mutates nothing"
    for key, verbs in mutating.items():
        if key[1].endswith(("/eviction", "/exec")):
            continue  # create-only subresources, covered below
        assert needed <= verbs, f"{key} grants {sorted(verbs)}, needs {sorted(needed)}"


@pytest.mark.parametrize(
    ("group", "resource", "verbs", "issued_by"),
    [
        ("", "pods/log", {"get"}, "kubectl logs"),
        ("", "pods/exec", {"create"}, "kubectl exec"),
        ("", "pods/eviction", {"create"}, "an eviction, preferred over a hard delete"),
        ("apps", "deployments/scale", {"get", "patch", "update"}, "kubectl scale"),
        ("apps", "statefulsets/scale", {"get", "patch", "update"}, "kubectl scale"),
        ("apps", "replicasets/scale", {"get", "patch", "update"}, "kubectl scale"),
        ("apps", "replicasets", {"get", "list"}, "kubectl rollout history/undo, deployments"),
        ("apps", "controllerrevisions", {"get", "list"}, "kubectl rollout history/undo, sts+ds"),
        ("metrics.k8s.io", "pods", {"get", "list"}, "kubectl top pods"),
        ("metrics.k8s.io", "nodes", {"get", "list"}, "kubectl top nodes"),
    ],
    ids=lambda value: value if isinstance(value, str) else "",
)
def test_subresources_the_tool_reaches_are_granted(
    group: str, resource: str, verbs: set[str], issued_by: str
) -> None:
    """A resource kind's own rule does not cover its subresources.

    kubectl logs, exec, scale and rollout each reach one, and each was a Forbidden with
    the resource's own grant fully in place.
    """
    granted = _granted(group, resource)
    assert set(verbs) <= granted, f"{group or 'core'}/{resource} for {issued_by}: {sorted(granted)}"


def test_no_role_can_reach_a_secret() -> None:
    """The one grant the README promises is absent from every role in the base."""
    for role, rules in _grants().items():
        for group, resource in rules:
            assert not (group == "" and resource.startswith("secret")), f"{role} reaches secrets"


# ----------------------------------------------------------------- the network boundary


def test_egress_reaches_every_endpoint_the_configmap_names() -> None:
    """The policy and the endpoints have to move together, so this is what checks that.

    Every address the deployment is configured to talk to has to be a destination the
    policy allows, and a telemetry backend silently blocked by egress reads exactly like
    a backend that is down.
    """
    rendered = _one(_render("production"), "ConfigMap")["data"]
    policy = _one(_render("production"), "NetworkPolicy")
    allowed: set[tuple[str, int]] = set()
    for rule in policy["spec"]["egress"]:
        namespaces = {
            to["namespaceSelector"]["matchLabels"].get("kubernetes.io/metadata.name")
            for to in rule.get("to", [])
            if "namespaceSelector" in to
        }
        for port in rule.get("ports", []):
            for namespace in namespaces:
                allowed.add((namespace, port["port"]))

    checked = 0
    for name, value in rendered.items():
        host, port = _host_port(value)
        if host is None or not host.endswith(".svc.cluster.local"):
            continue
        namespace = host.split(".")[1]
        checked += 1
        assert (namespace, port) in allowed, (
            f"{name}={value} points at {namespace}:{port}, which the NetworkPolicy's "
            f"egress does not allow. Allowed: {sorted(allowed)}"
        )
    assert checked >= 4, "no in-cluster endpoints were checked; the parsing is wrong"


def _host_port(value: str) -> tuple[str | None, int | None]:
    """Read a host and port out of a URL or a bare host:port, as OTLP is spelled."""
    parsed = urlparse(value if "//" in value else f"//{value}")
    if not parsed.hostname or parsed.port is None:
        return None, None
    return parsed.hostname, parsed.port


def test_local_overlay_removes_the_policy_rather_than_relaxing_it() -> None:
    """kind ships no controller for it, and a policy that looks enforced and is not is
    worse than an absent one."""
    assert _of_kind(_render("local"), "NetworkPolicy") == []
    assert _of_kind(_render("production"), "NetworkPolicy") != []


def test_the_service_reaches_the_port_the_policy_admits() -> None:
    """Ingress is written against the pod's port, which is what a policy sees post-DNAT."""
    docs = _render("production")
    target = _one(docs, "Service")["spec"]["ports"][0]["targetPort"]
    ports = {port["containerPort"] for port in _container(docs)["ports"]}
    named = {port["name"]: port["containerPort"] for port in _container(docs)["ports"]}
    admitted = {
        port["port"]
        for rule in _one(docs, "NetworkPolicy")["spec"]["ingress"]
        for port in rule["ports"]
    }
    assert named[target] in ports
    assert admitted == {named[target]}
