"""Minimal CLI — launches the web UI in the default browser."""

from __future__ import annotations

import click

from preprod.web import run_web


@click.group()
def main() -> None:
    """VidTighten — pre-production workflow for Final Cut Pro."""


main.add_command(run_web)
