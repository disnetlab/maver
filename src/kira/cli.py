"""Command line entry point.

Subcommands are added alongside the features they drive: ``serve`` with the HTTP
API, ``approvals`` with the human approval workflow, ``config`` with the settings
model. Keeping them together means a feature and the way an operator reaches it
land in the same commit.
"""

import typer

from kira import __version__

app = typer.Typer(
    name="kira",
    help="Kubernetes Incident Recovery Agent.",
    no_args_is_help=True,
    add_completion=False,
)


@app.command()
def version() -> None:
    """Print the installed version."""
    typer.echo(__version__)


if __name__ == "__main__":  # pragma: no cover
    app()
