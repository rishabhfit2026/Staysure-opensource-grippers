"""Entry point: python -m grippers <command>"""

import sys
import click

from grippers.scripts.record import main as record_main
from grippers.scripts.visualize import main as visualize_main


@click.group()
def cli():
    """Staysure OpenSource Grippers — humanoid manipulation learning."""


cli.add_command(record_main, name="record")
cli.add_command(visualize_main, name="visualize")


if __name__ == "__main__":
    cli()
