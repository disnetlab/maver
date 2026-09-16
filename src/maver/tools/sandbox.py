"""Unprivileged filesystem sandbox for shell/code tool subprocesses.

Linux deployments use Landlock so a tool process can read/write its current
incident directory and read runtime dependencies, but cannot inspect the
persistent WORKDIR, sibling incidents, or the research event ledger. Landlock
is inherited by every descendant, so shell and Python indirection cannot bypass
the boundary. Non-Linux development hosts run without Landlock.
"""

from __future__ import annotations

import ctypes
import errno
import logging
import os
import platform
import sys
from pathlib import Path

_log = logging.getLogger(__name__)

SANDBOX_ERROR_PREFIX = "filesystem sandbox unavailable:"

_LANDLOCK_CREATE_RULESET = 444
_LANDLOCK_ADD_RULE = 445
_LANDLOCK_RESTRICT_SELF = 446
_LANDLOCK_CREATE_RULESET_VERSION = 1
_LANDLOCK_RULE_PATH_BENEATH = 1
_PR_SET_NO_NEW_PRIVS = 38

_EXECUTE = 1 << 0
_WRITE_FILE = 1 << 1
_READ_FILE = 1 << 2
_READ_DIR = 1 << 3
_REMOVE_DIR = 1 << 4
_REMOVE_FILE = 1 << 5
_MAKE_CHAR = 1 << 6
_MAKE_DIR = 1 << 7
_MAKE_REG = 1 << 8
_MAKE_SOCK = 1 << 9
_MAKE_FIFO = 1 << 10
_MAKE_BLOCK = 1 << 11
_MAKE_SYM = 1 << 12
_REFER = 1 << 13
_TRUNCATE = 1 << 14

_ABI1_ACCESS = (1 << 13) - 1
_READ_ONLY_ACCESS = _EXECUTE | _READ_FILE | _READ_DIR


class _RulesetAttr(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class _PathBeneathAttr(ctypes.Structure):
    _fields_ = [
        ("allowed_access", ctypes.c_uint64),
        ("parent_fd", ctypes.c_int32),
    ]


def _syscall(libc: ctypes.CDLL, number: int, *args: object) -> int:
    result = int(libc.syscall(number, *args))
    if result < 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))
    return result


def _handled_access(abi: int) -> int:
    access = _ABI1_ACCESS
    if abi >= 2:
        access |= _REFER
    if abi >= 3:
        access |= _TRUNCATE
    return access


def _runtime_read_roots(session: Path) -> list[Path]:
    """Runtime trees needed to launch bash/Python and kubectl proxy helpers."""
    project_root = Path(__file__).resolve().parents[2]
    # Python imports ``agent`` through the project root (``/app`` in the image),
    # so deployed sessions under the separate ``/workspace`` tree need that
    # root readable. If a local developer puts WORKDIR inside the repository,
    # do not grant the project root: that would also expose sibling incidents.
    project_access = (
        project_root if not session.is_relative_to(project_root) else project_root / "agent"
    )
    candidates = [
        project_access,
        Path(sys.executable).resolve().parent.parent,
        # A virtualenv's python is a symlink to the base interpreter, so the resolved
        # path above lands on the base install and not on the environment the
        # subprocess actually starts in. Without the environment root, Python dies in
        # site initialisation reading pyvenv.cfg, which is how every sandboxed
        # code_exec in the image failed while every test on a developer machine passed.
        Path(sys.prefix),
        Path(sys.base_prefix),
        Path("/usr"),
        Path("/bin"),
        Path("/sbin"),
        Path("/lib"),
        Path("/lib64"),
        Path("/etc"),
        Path("/proc"),
        Path("/dev"),
    ]
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        key = str(resolved)
        if candidate.exists() and key not in seen:
            seen.add(key)
            unique.append(resolved)
    return unique


def _add_path_rule(
    libc: ctypes.CDLL,
    ruleset_fd: int,
    path: Path,
    access: int,
) -> None:
    flags = getattr(os, "O_PATH", os.O_RDONLY) | os.O_CLOEXEC
    fd = os.open(path, flags)
    try:
        attr = _PathBeneathAttr(allowed_access=access, parent_fd=fd)
        _syscall(
            libc,
            _LANDLOCK_ADD_RULE,
            ruleset_fd,
            _LANDLOCK_RULE_PATH_BENEATH,
            ctypes.byref(attr),
            0,
        )
    finally:
        os.close(fd)


def enter_session_sandbox(session_dir: str | Path) -> bool:
    """Restrict this process to its session directory on Linux.

    Returns False on non-Linux development hosts. Linux is fail-closed: callers
    receive an exception when Landlock is unavailable instead of silently
    running an unisolated agent tool.
    """
    if platform.system() != "Linux":
        return False
    if os.environ.get("TOOL_FS_SANDBOX_ENABLED", "true").lower() in {"0", "false", "no", "off"}:
        return False

    session = Path(session_dir).resolve(strict=True)
    if not session.is_dir():
        raise NotADirectoryError(str(session))

    libc = ctypes.CDLL(None, use_errno=True)
    try:
        abi = _syscall(
            libc,
            _LANDLOCK_CREATE_RULESET,
            0,
            0,
            _LANDLOCK_CREATE_RULESET_VERSION,
        )
    except OSError as exc:
        if exc.errno in {errno.ENOSYS, errno.EOPNOTSUPP, errno.EINVAL}:
            raise RuntimeError("Linux kernel does not provide usable Landlock") from exc
        raise

    handled = _handled_access(abi)
    attr = _RulesetAttr(handled_access_fs=handled)
    ruleset_fd = _syscall(
        libc,
        _LANDLOCK_CREATE_RULESET,
        ctypes.byref(attr),
        ctypes.sizeof(attr),
        0,
    )
    try:
        for root in _runtime_read_roots(session):
            _add_path_rule(libc, ruleset_fd, root, _READ_ONLY_ACCESS & handled)
        if Path("/dev/null").exists():
            _add_path_rule(
                libc,
                ruleset_fd,
                Path("/dev/null"),
                (_READ_FILE | _WRITE_FILE) & handled,
            )
        _add_path_rule(libc, ruleset_fd, session, handled)
        if libc.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
            code = ctypes.get_errno()
            raise OSError(code, os.strerror(code))
        _syscall(libc, _LANDLOCK_RESTRICT_SELF, ruleset_fd, 0)
    finally:
        os.close(ruleset_fd)
    return True


_available: bool | None = None


def fs_sandbox_available() -> bool:
    """Whether a sandboxed tool subprocess can actually start on this host.

    Asked once and remembered, so that the two tools it gates are dropped before a run
    rather than failing on every call of it. :func:`enter_session_sandbox` is fail-closed
    on Linux by design, and the wrapper turns that refusal into a model-facing tool error
    — which reads as a fault to work around, so the model tries again, differently, and
    again. The documented amd64 image runs emulated on an Apple Silicon host and that
    kernel does not implement the Landlock syscalls; neither does one built without the
    LSM.

    True where the sandbox does not apply rather than fails: on a non-Linux development
    host :func:`enter_session_sandbox` returns False and the command runs unsandboxed,
    which is that platform's documented behaviour and not a broken tool.

    Returns:
        True when a tool subprocess can start.
    """
    global _available
    if _available is None:
        _available = _probe()
        if not _available:
            _log.warning(
                "this kernel provides no usable Landlock, so bash_exec and code_exec "
                "are withheld from every role. Set MAVER_TOOLS__FS_SANDBOX=false to run "
                "them unsandboxed instead."
            )
    return _available


def _probe() -> bool:
    """Create a Landlock ruleset and close it again, without restricting this process."""
    if platform.system() != "Linux":
        return True
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        abi = _syscall(libc, _LANDLOCK_CREATE_RULESET, 0, 0, _LANDLOCK_CREATE_RULESET_VERSION)
        attr = _RulesetAttr(handled_access_fs=_handled_access(abi))
        ruleset_fd = _syscall(
            libc, _LANDLOCK_CREATE_RULESET, ctypes.byref(attr), ctypes.sizeof(attr), 0
        )
    except Exception:
        # Conservative on an answer we could not get: a tool withheld with a reason in
        # the run's record is recoverable, one that errors on every call is not.
        return False
    os.close(ruleset_fd)
    return True


def sandboxed_command(command: list[str], session_dir: str | Path) -> list[str]:
    """Wrap a command so Landlock is installed immediately before exec."""
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        str(Path(session_dir).resolve()),
        "--",
        *command,
    ]


def _main(argv: list[str]) -> int:
    if len(argv) < 4 or argv[2] != "--":
        print(f"{SANDBOX_ERROR_PREFIX} invalid sandbox launcher arguments")
        return 126
    session_dir = argv[1]
    command = argv[3:]
    try:
        os.chdir(session_dir)
        enter_session_sandbox(session_dir)
    except Exception as exc:
        print(f"{SANDBOX_ERROR_PREFIX} {type(exc).__name__}: {exc}")
        return 126
    os.execvpe(command[0], command, os.environ)


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
