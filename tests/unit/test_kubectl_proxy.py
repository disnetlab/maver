"""The proxy_exec kubectl route: what it refuses, in the words it refused with.

The refusals are the part of this that a model was measured against, so they are not
written down here twice. Every expectation below is parsed out of the frozen prototype
shim and formatted, which means a reworded refusal fails as a difference from the archive
rather than passing against a copy of itself that was edited in the same commit.

Nothing here runs the real kubectl. Every execution path is given a stub script that
prints its arguments, and the one test that needs a proxy stands one up on a loopback
port of its own.
"""

import ast
import http.server
import os
import pathlib
import subprocess
import sys
import threading
from collections.abc import Iterator
from typing import Any

import pytest
from prototype_tree import prototype_root

from maver.tools import kubectl_proxy
from maver.tools.kubectl_proxy import (
    ProxyRoute,
    apply_proxy_env,
    check_proxy,
    main,
    run_via_proxy,
    write_proxy_kubeconfig,
)

PROTOTYPE_SHIM = prototype_root() / "agent" / "tools" / "kubectl_proxy_shim.py"
"""The shim as the archive ran it. Read by the parity check below and by nothing else."""

SHIPPED = pathlib.Path(kubectl_proxy.__file__)

_MESSAGE_CALLS = {"_err", "tool_error", "print"}
"""The calls that put a string in front of a model, in either file."""

_MESSAGE_RETURNS = {
    # The prototype's, which build their message inside an `_err` call and are covered
    # by the rule above; named anyway, so that a refusal moved out of one of them is
    # still read.
    "_authorize",
    "_reject_proxy_watch",
    "_reject_interactive_flags",
    # MAVER's, which return the message for one caller to print, so that the `error:`
    # prefix is added in one place rather than in each of them.
    "_authorisation_refusal",
    "_watch_refusal",
    "_interactive_refusal",
}
"""Functions whose return value is a message rather than a command or a path.

Named rather than inferred, because a rule that read every returned string would also
read the argument rewriting — ``{} --wait=false`` is a command, not something anybody is
shown — and a set comparison with those in it would fail over a difference in where a
flag is appended.
"""

AUTHORITY_REFUSAL = (
    "kubectl writes through the shell are not approval-gated; this deployment "
    "requires approval (authority.mode={}), so writes are refused here"
)
"""The one refusal the archive did not have.

Every other cluster write in this system passes the approval gate. A command inside a
sandbox has nothing in front of it, so a deployment that asks a human about its writes
must not be written around from a shell. Under ``auto`` — the mode every measured cell
ran, and the mode the evaluation compose file sets — this never fires and the route
behaves exactly as the archive did.
"""


def _rendered(node: ast.expr) -> str | None:
    """Render a literal or an f-string as a template, with ``{}`` for each expression."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(
            value.value
            if isinstance(value, ast.Constant) and isinstance(value.value, str)
            else "{}"
            for value in node.values
        )
    return None


def _templates(path: pathlib.Path) -> set[str]:
    """Every model-facing string a module builds, as a template.

    The ``error:`` prefix is stripped because the prototype wrote it into each string and
    MAVER adds it in one place, which is a difference in where the prefix is added and not
    in what the model reads.
    """
    found: set[str | None] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Call):
            name = (
                node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", "")
            )
            if name in _MESSAGE_CALLS:
                found.update(_rendered(argument) for argument in node.args)
        elif isinstance(node, ast.FunctionDef) and node.name in _MESSAGE_RETURNS:
            found.update(
                _rendered(inner.value)
                for inner in ast.walk(node)
                if isinstance(inner, ast.Return) and inner.value is not None
            )
    # A template with no words in it is the `error:` helper's own argument, or a number
    # being formatted. Both are how a message is assembled rather than one of them.
    messages = {text.removeprefix("error: ") for text in found if text is not None}
    return {text for text in messages if text.replace("{}", "").strip()}


ARCHIVED = _templates(PROTOTYPE_SHIM)


def archived(fragment: str) -> str:
    """Look one archived refusal up by a fragment of it, so a test names it once."""
    matched = [text for text in ARCHIVED if fragment in text]
    assert len(matched) == 1, f"{fragment!r} matches {len(matched)} archived messages"
    return matched[0]


def stub_kubectl(directory: pathlib.Path, body: str = "printf '%s' \"$KUBECONFIG $*\"") -> str:
    """Install a kubectl that reports what it was given and never reaches a cluster."""
    script = directory / "kubectl-stub"
    script.write_text(f"#!/bin/sh\n{body}\n")
    script.chmod(0o755)
    return str(script)


@pytest.fixture
def shim_env(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> pathlib.Path:
    """The environment `apply_proxy_env` builds, for a stage that may write."""
    monkeypatch.setenv("KUBECTL_MODE", "proxy_exec")
    monkeypatch.setenv("KUBECTL_PROXY_URL", "http://proxy.invalid:16444")
    monkeypatch.setenv("KUBECTL_PROXY_REQUIRED", "false")
    monkeypatch.setenv("KUBECTL_PROXY_KUBECONFIG", str(tmp_path / "config"))
    monkeypatch.setenv("KUBECTL_REAL_BINARY", stub_kubectl(tmp_path))
    monkeypatch.setenv("KUBECTL_EXEC_TIMEOUT_SECONDS", "120")
    monkeypatch.setenv("AGENT_STAGE", "execution")
    monkeypatch.setenv("MAVER_AUTHORITY_MODE", "auto")
    return tmp_path


@pytest.fixture
def usable_stdin(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Give the shim a real stdin to hand its child.

    It passes the parent's own streams through so that ``kubectl apply -f -`` still reads
    the pipe it was given, and pytest's captured stdin has no file descriptor to pass.
    """
    with open(os.devnull) as devnull:
        monkeypatch.setattr(sys, "stdin", devnull)
        yield


@pytest.mark.parity
def test_every_refusal_is_the_one_the_archive_showed_the_model() -> None:
    """The route's refusals are its interface, and this is what pins them.

    A model that has just been told a command is not permitted decides what to try next
    from the words it was told it in — the ``-A`` refusal says the verb is fine and to
    scope the query, and a reworded one that failed to say so cost a whole class of reads
    in the archive. So the set is compared whole rather than sampled: a message that
    disappears is as much a divergence as one that changes.
    """
    shipped = _templates(SHIPPED)

    assert ARCHIVED - shipped == set()
    assert shipped - ARCHIVED == {AUTHORITY_REFUSAL}


@pytest.mark.parity
@pytest.mark.parametrize(
    ("argv", "stage", "fragment", "arguments"),
    [
        (["cluster-info"], "rca", "kubectl '{}'", ("cluster-info",)),
        (["get", "pods", "-A"], "rca", "scope the query to one namespace", ("get", "-A", "get")),
        (
            ["get", "pods", "-n", "chaos-mesh"],
            "rca",
            "credential/context override",
            ("get", "chaos-mesh"),
        ),
        (["delete", "pod/x", "-n", "shop"], "rca", "writes are not permitted", ("rca",)),
        (["delete", "pod/x", "-n", "shop"], "", "writes are not permitted", ("unknown",)),
        (["rollout", "status", "deploy/x"], "execution", "rollout status", ()),
        (["wait", "--for=condition=ready", "pod/x"], "execution", "kubectl wait", ()),
        (["exec", "-it", "pod/x", "--", "sh"], "execution", "interactive flags", ("exec", "-it")),
    ],
)
def test_a_refused_command_says_what_the_archive_said(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    shim_env: pathlib.Path,
    argv: list[str],
    stage: str,
    fragment: str,
    arguments: tuple[str, ...],
) -> None:
    monkeypatch.setenv("AGENT_STAGE", stage)

    assert main(argv) == 1
    assert capsys.readouterr().err == f"error: {archived(fragment).format(*arguments)}\n"


@pytest.mark.parity
def test_a_write_from_the_remediating_stage_runs(
    shim_env: pathlib.Path, usable_stdin: None, capfd: pytest.CaptureFixture[str]
) -> None:
    """The stage check is the archive's, and ``execution`` is one of the two it names."""
    assert main(["delete", "pod/x", "-n", "shop"]) == 0

    assert "delete pod/x -n shop" in capfd.readouterr().out


def test_a_write_is_refused_where_a_human_approves_writes(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], shim_env: pathlib.Path
) -> None:
    """The one screen added to the route, and the reason it is not the archive's.

    A shell is the one place a cluster write reaches the cluster without passing the
    approval gate, because there is no call for the gate to wrap. A deployment that
    answers ``prompt`` has said writes are a human's decision, and a route that ignored
    that would be the whole gate's exception.
    """
    monkeypatch.setenv("MAVER_AUTHORITY_MODE", "prompt")

    assert main(["delete", "pod/x", "-n", "shop"]) == 1
    assert capsys.readouterr().err == f"error: {AUTHORITY_REFUSAL.format('prompt')}\n"


@pytest.mark.parity
def test_kubectl_outside_the_route_is_refused(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Reached by a shim left on PATH by a session whose deployment does not use it."""
    monkeypatch.delenv("KUBECTL_MODE", raising=False)
    monkeypatch.delenv("KUBECTL_PROXY_URL", raising=False)

    assert main(["get", "pods"]) == 1
    assert capsys.readouterr().err == f"error: {archived('only available through shell/code')}\n"


@pytest.mark.parity
def test_a_route_with_no_proxy_is_refused(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], shim_env: pathlib.Path
) -> None:
    monkeypatch.setenv("KUBECTL_PROXY_URL", "")

    assert main(["get", "pods"]) == 1
    assert capsys.readouterr().err == f"error: {archived('is required in KUBECTL_MODE')}\n"


@pytest.mark.parity
def test_a_missing_kubectl_is_refused(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], shim_env: pathlib.Path
) -> None:
    monkeypatch.setenv("KUBECTL_REAL_BINARY", "kubectl-that-is-not-installed")

    assert main(["get", "pods"]) == 1
    assert capsys.readouterr().err == f"error: {archived('real kubectl binary not found')}\n"


@pytest.mark.parity
def test_a_command_that_outlives_its_bound_is_killed_and_says_so(
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
    shim_env: pathlib.Path,
    usable_stdin: None,
) -> None:
    """124, as a shell reports a timeout, so a pipeline reading the status sees one."""
    monkeypatch.setenv("KUBECTL_REAL_BINARY", stub_kubectl(shim_env, "exec sleep 30"))
    monkeypatch.setenv("KUBECTL_EXEC_TIMEOUT_SECONDS", "1")

    assert main(["get", "pods"]) == 124
    assert capfd.readouterr().err == f"error: {archived('command timed out after').format(1)}\n"


@pytest.mark.parity
def test_a_delete_does_not_block_on_a_finalizer(
    shim_env: pathlib.Path, usable_stdin: None, capfd: pytest.CaptureFixture[str]
) -> None:
    """A blocking delete on a finalizer-protected resource wedged one session's cluster
    access for three quarters of an hour, reads included."""
    assert main(["delete", "pvc/data", "-n", "shop"]) == 0

    assert capfd.readouterr().out.endswith("delete pvc/data -n shop --wait=false")


def test_the_generated_kubeconfig_carries_the_proxy_and_no_credential(
    tmp_path: pathlib.Path,
) -> None:
    """The proxy holds the credentials and decides what a request may do. A kubeconfig
    that carried a second answer would be a way around the first."""
    path = write_proxy_kubeconfig("http://127.0.0.1:16444", tmp_path / "deep" / "config")

    written = path.read_text(encoding="utf-8")
    assert '"server": "http://127.0.0.1:16444"' in written
    assert '"user": {}' in written
    assert "token" not in written


async def test_a_read_returns_the_command_output_byte_for_byte(tmp_path: pathlib.Path) -> None:
    """Two internal readers parse this as JSON, and an envelope in front of the payload
    turns both into "unavailable" without a word about why."""
    code, output = await run_via_proxy(
        ["get", "pods", "-n", "shop", "-o", "json"],
        proxy_url="http://127.0.0.1:16444",
        kubeconfig_dir=tmp_path,
        binary=stub_kubectl(tmp_path),
        timeout_s=120,
    )

    assert code == 0
    assert output == f"{tmp_path / 'config'} get pods -n shop -o json"


async def test_a_blocking_command_is_bounded_under_the_call_that_waits_for_it(
    tmp_path: pathlib.Path,
) -> None:
    """Unbounded, ``wait`` blocks until the caller's own timeout hard-kills it, so one
    wrong fix turns every retry into a full stall."""
    _, output = await run_via_proxy(
        ["wait", "--for=condition=ready", "pod/x"],
        proxy_url="http://127.0.0.1:16444",
        kubeconfig_dir=tmp_path,
        binary=stub_kubectl(tmp_path),
        timeout_s=120,
    )

    assert output.endswith("wait --for=condition=ready pod/x --timeout=110s")


async def test_a_failed_command_comes_back_with_its_own_complaint(
    tmp_path: pathlib.Path,
) -> None:
    """The backend's own words are what let a model repair a query rather than reissue
    it, so the exit status does not replace them."""
    code, output = await run_via_proxy(
        ["get", "pods"],
        proxy_url="http://127.0.0.1:16444",
        kubeconfig_dir=tmp_path,
        binary=stub_kubectl(tmp_path, 'echo "no such resource" >&2; exit 1'),
        timeout_s=120,
    )

    assert code == 1
    assert output == "no such resource"


async def test_a_command_that_outlives_its_bound_is_reported_as_a_timeout(
    tmp_path: pathlib.Path,
) -> None:
    code, output = await run_via_proxy(
        ["get", "pods"],
        proxy_url="http://127.0.0.1:16444",
        kubeconfig_dir=tmp_path,
        binary=stub_kubectl(tmp_path, "sleep 30"),
        timeout_s=1,
    )

    assert code == 124
    assert output == "kubectl timed out after 1s"


async def test_a_kubectl_that_cannot_be_run_is_a_result_and_not_an_exception(
    tmp_path: pathlib.Path,
) -> None:
    code, output = await run_via_proxy(
        ["get", "pods"],
        proxy_url="http://127.0.0.1:16444",
        kubeconfig_dir=tmp_path,
        binary="kubectl-that-is-not-installed",
        timeout_s=120,
    )

    assert code != 0
    assert output.startswith("kubectl could not be executed:")


class _Version(http.server.BaseHTTPRequestHandler):
    """A proxy that answers ``/version``, which is what readiness is read from."""

    status = 200

    def do_GET(self) -> None:
        self.send_response(self.status)
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(self, format: str, *args: Any) -> None:
        return None


@pytest.fixture
def proxy() -> Iterator[str]:
    """A proxy on a loopback port of its own, so nothing here needs a cluster."""
    server = http.server.HTTPServer(("127.0.0.1", 0), _Version)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_a_proxy_that_answers_is_not_reported(proxy: str) -> None:
    assert check_proxy(proxy, check_timeout_s=5.0, ready_timeout_s=5.0) is None


@pytest.mark.parity
def test_a_proxy_that_answers_badly_is_reported_with_what_it_said(
    monkeypatch: pytest.MonkeyPatch, proxy: str
) -> None:
    """A proxy that is up and refusing is a different operator problem from one that is
    gone, and the only thing that distinguishes them in the log is what it said."""
    monkeypatch.setattr(_Version, "status", 503)

    unavailable = check_proxy(proxy, check_timeout_s=5.0, ready_timeout_s=0)

    assert unavailable == (
        f"error: {archived('is unavailable after').format(0, proxy, 'HTTP Error 503: Service Unavailable')}"
    )


@pytest.mark.parity
def test_a_proxy_that_never_answers_is_reported_after_the_wait() -> None:
    """The wait is what a restarted port-forward needs. Naming it in the failure is what
    tells an operator the difference between a proxy that is slow and one that is gone."""
    closed = "http://127.0.0.1:1"

    unavailable = check_proxy(closed, check_timeout_s=0.5, ready_timeout_s=0)

    assert unavailable is not None
    head, _ = archived("is unavailable after").split("(", 1)
    assert unavailable.startswith(f"error: {head.format(0, closed)}(")


def test_the_sandbox_environment_puts_the_shim_in_front_of_the_real_kubectl(
    tmp_path: pathlib.Path,
) -> None:
    """PATH, because a shell pipeline, a Python ``subprocess`` call and a script the
    model wrote all resolve the binary the same way. A rule about what may be typed
    would have to recognise all three as kubectl calls first."""
    session = tmp_path / "session"
    real = stub_kubectl(tmp_path)

    env = apply_proxy_env(
        {"PATH": "/usr/bin", "KUBECONFIG": "/root/.kube/config"},
        session_dir=session,
        stage="rca",
        proxy_url="http://127.0.0.1:16444/",
        proxy_required=True,
        check_timeout_s=5.0,
        ready_timeout_s=60.0,
        kubectl_binary=real,
        exec_timeout_s=120.0,
        authority_mode="auto",
    )

    shim = session / ".kubectl-proxy" / "bin"
    assert env["PATH"] == f"{shim}{os.pathsep}/usr/bin"
    assert env["KUBECTL_MODE"] == "proxy_exec"
    assert env["KUBECTL_PROXY_URL"] == "http://127.0.0.1:16444"
    assert env["KUBECTL_PROXY_REQUIRED"] == "true"
    assert env["KUBECTL_PROXY_KUBECONFIG"] == str(session / ".kubectl-proxy" / "config")
    assert env["KUBECTL_PROXY_CHECK_TIMEOUT_SECONDS"] == "5"
    assert env["KUBECTL_PROXY_READY_TIMEOUT_SECONDS"] == "60"
    assert env["KUBECTL_EXEC_TIMEOUT_SECONDS"] == "120"
    assert env["KUBECTL_REAL_BINARY"] == real
    assert env["AGENT_STAGE"] == "rca"
    assert env["MAVER_AUTHORITY_MODE"] == "auto"

    script = (shim / "kubectl").read_text(encoding="utf-8")
    assert script.startswith("#!/bin/sh\n")
    assert script.endswith(f'exec {sys.executable} -m maver.tools.kubectl_proxy "$@"\n')
    assert os.access(shim / "kubectl", os.X_OK)


def test_the_sandbox_never_inherits_a_kubeconfig(tmp_path: pathlib.Path) -> None:
    """An operator's kubeconfig in a sandbox reaches the cluster around the proxy and
    around the gate, with the agent's own credentials and nothing recording it."""
    env = apply_proxy_env(
        {"PATH": "/usr/bin", "KUBECONFIG": "/root/.kube/config"},
        session_dir=tmp_path,
        stage="rca",
        proxy_url="http://127.0.0.1:16444",
        proxy_required=True,
        check_timeout_s=5.0,
        ready_timeout_s=60.0,
        kubectl_binary="kubectl",
        exec_timeout_s=120.0,
        authority_mode="auto",
    )

    assert "KUBECONFIG" not in env


def test_the_real_kubectl_is_resolved_before_the_shim_shadows_the_name(
    tmp_path: pathlib.Path,
) -> None:
    """Left as a bare name, ``kubectl`` would resolve through the PATH the shim was just
    prepended to, so the shim would re-enter itself for as long as the process limit
    allowed."""
    real = tmp_path / "kubectl"
    real.write_text("#!/bin/sh\ntrue\n")
    real.chmod(0o755)

    env = apply_proxy_env(
        {"PATH": str(tmp_path)},
        session_dir=tmp_path / "session",
        stage="execution",
        proxy_url="http://127.0.0.1:16444",
        proxy_required=True,
        check_timeout_s=5.0,
        ready_timeout_s=60.0,
        kubectl_binary="kubectl",
        exec_timeout_s=120.0,
        authority_mode="auto",
    )

    assert env["KUBECTL_REAL_BINARY"] == str(real)


def test_the_shim_runs_even_when_a_command_clears_the_environment(
    tmp_path: pathlib.Path,
) -> None:
    """A ``sudo``, an ``env -i`` or a script sourcing a clean profile would otherwise
    leave the shim with no route and no stage, and a kubectl with neither reads as a
    cluster that is unreachable rather than as a session that lost its wiring.

    Run rather than read, because this is also the only check that the generated script
    is a working entry point: it resolves an interpreter, imports the package and reaches
    :func:`main`, none of which a comparison of its text would notice going wrong.
    """
    session = tmp_path / "session"
    env = apply_proxy_env(
        {"PATH": "/usr/bin"},
        session_dir=session,
        stage="execution",
        proxy_url="http://127.0.0.1:16444",
        proxy_required=False,
        check_timeout_s=5.0,
        ready_timeout_s=60.0,
        kubectl_binary=stub_kubectl(tmp_path),
        exec_timeout_s=120.0,
        authority_mode="auto",
    )
    shim = session / ".kubectl-proxy" / "bin" / "kubectl"

    stripped = subprocess.run(
        ["/usr/bin/env", "-i", str(shim), "delete", "pod/x", "-n", "shop"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert stripped.returncode == 0, stripped.stderr
    assert stripped.stdout.endswith("delete pod/x -n shop --wait=false")
    assert str(session / ".kubectl-proxy" / "config") in stripped.stdout
    assert env["KUBECTL_PROXY_KUBECONFIG"] in stripped.stdout


def test_the_shipped_defaults_are_the_configuration_the_archive_ran() -> None:
    """The route is not a mode to be turned on: these are the values every archived cell
    reached its cluster with, so a deployment that changes nothing reproduces them."""
    assert ProxyRoute() == ProxyRoute(
        url="http://127.0.0.1:16444",
        required=True,
        check_timeout_s=5.0,
        ready_timeout_s=60.0,
        binary="kubectl",
        exec_timeout_s=120.0,
    )
