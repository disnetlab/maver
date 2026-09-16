"""Every shipped prompt asset either has a reader or does not ship.

The same lie as an unread setting, in package data. A prompt describing a mechanism this
system does not have ships in the wheel, and a reader who opens it learns behaviour that
never runs: seven such assets were shipping, for a counterfactual fork, a playbook layer
and two memory-delivery paths, all of which the rebuild dropped.

The scan reads the call sites rather than the loader, and resolves a module constant
inside an f-string, because one asset really is read under a composed name and a scan
that missed it would delete a live prompt. It is deliberately crude in one direction
only: a name appearing as a quoted string anywhere counts as a reader, which
over-reports and never under-reports.
"""

import ast
import pathlib
import re

SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "maver"
PROMPTS = SRC / "prompts"
LOADER = PROMPTS / "loader.py"

COMPOSED_FROM_A_ROLE = ("roles/", "blocks/")
"""Families whose members are named by a role and a state rather than by a call site.

Every shipped role is composed in both memory states by ``test_prompt_loader``, so a
member of these missing or unreachable fails there, where the composition happens.
"""


def _sources() -> list[tuple[pathlib.Path, ast.Module]]:
    return [
        (path, ast.parse(path.read_text(encoding="utf-8")))
        for path in sorted(SRC.rglob("*.py"))
        if "__pycache__" not in str(path) and path != LOADER
    ]


def _constants(tree: ast.Module) -> dict[str, str]:
    """Module-level ``NAME = "literal"`` bindings, so an f-string over one resolves."""
    found: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Constant):
            continue
        if not isinstance(node.value.value, str):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                found[target.id] = node.value.value
    return found


def _render(node: ast.JoinedStr, constants: dict[str, str]) -> str:
    """Reduce an f-string to a regexp, substituting any constant it names."""
    parts = []
    for value in node.values:
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            parts.append(re.escape(value.value))
        elif (
            isinstance(value, ast.FormattedValue)
            and isinstance(value.value, ast.Name)
            and value.value.id in constants
        ):
            parts.append(re.escape(constants[value.value.id]))
        else:
            parts.append(r"[^/]+")
    return "^" + "".join(parts) + "$"


def _asked_for() -> tuple[set[str], list[re.Pattern[str]]]:
    """Every asset path the call sites ask for, literally or as a pattern."""
    literals: set[str] = set()
    patterns: list[re.Pattern[str]] = []

    for _, tree in _sources():
        constants = _constants(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            if node.func.id == "read_asset" and node.args:
                arg = node.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    literals.add(arg.value)
                elif isinstance(arg, ast.JoinedStr) and "control/" not in ast.unparse(arg):
                    patterns.append(re.compile(_render(arg, constants)))
            elif node.func.id == "control" and node.args:
                # A one-argument helper over `control/<name>.txt`, whose callers name
                # the block. Without this the helper's own f-string is a wildcard over
                # the whole family and every member of it counts as read.
                arg = node.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    literals.add(f"control/{arg.value}.txt")
            elif node.func.id == "read_runtime" and len(node.args) == 2:
                role, name = node.args
                if not isinstance(role, ast.Constant):
                    continue
                if isinstance(name, ast.Constant) and isinstance(name.value, str):
                    literals.add(f"runtime/{role.value}/{name.value}")
                elif isinstance(name, ast.JoinedStr):
                    rendered = _render(name, constants).removeprefix("^")
                    patterns.append(re.compile(f"^runtime/{re.escape(str(role.value))}/{rendered}"))
    return literals, patterns


def _quoted_names() -> str:
    """Every source file, for the crude fallback: a stem named as a string literal."""
    return "\n".join(
        path.read_text(encoding="utf-8") for path, _ in _sources() if not str(path).endswith(".txt")
    )


def test_every_shipped_prompt_asset_has_a_reader() -> None:
    literals, patterns = _asked_for()
    text = _quoted_names()

    unread = []
    for path in sorted(PROMPTS.rglob("*.txt")):
        relative = path.relative_to(PROMPTS).as_posix()
        if relative.startswith(COMPOSED_FROM_A_ROLE):
            continue
        if relative in literals or any(pattern.match(relative) for pattern in patterns):
            continue
        if re.search(rf'["\']{re.escape(path.stem)}["\'.]', text):
            continue
        unread.append(relative)

    assert not unread, (
        f"these prompt assets have no reader: {unread}. Wire it, or delete it — a "
        "prompt nothing reads describes behaviour this system does not have."
    )
