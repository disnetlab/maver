"""Command line entry point.

Subcommands are added alongside the features they drive: ``serve`` with the HTTP
API, ``approvals`` with the human approval workflow, ``config`` with the settings
model. Keeping them together means a feature and the way an operator reaches it land
in the same commit.
"""

import typer

from kira import __version__

app = typer.Typer(
    name="kira",
    help="Kubernetes Incident Recovery Agent.",
    no_args_is_help=True,
    add_completion=False,
)


def _print_version(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit


@app.callback()
def main(
    *,
    show_version: bool = typer.Option(
        False,
        "--version",
        callback=_print_version,
        is_eager=True,
        help="Print the version and exit.",
    ),
) -> None:
    """Kubernetes Incident Recovery Agent.

    The callback exists so that the application stays a command group. Without it,
    Typer collapses an application holding a single command into that command, and
    every subcommand added later silently changes the CLI's shape.
    """


@app.command()
def version() -> None:
    """Print the installed version."""
    typer.echo(__version__)


if __name__ == "__main__":  # pragma: no cover
    app()
