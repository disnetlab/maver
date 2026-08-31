from importlib.resources import files
from typing import Any

import yaml

from kira.tools.allowlists import ROLE_TOOLSETS
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
