#!/usr/bin/env python3
"""Convert LostBench scenarios to SafeShift format."""

import click


@click.command()
@click.option("--source", required=True, type=click.Path(exists=True), help="LostBench dir")
@click.option("--output", required=True, type=str, help="Output directory")
def main(source: str, output: str) -> None:
    """Import LostBench emergency scenarios to SafeShift format."""
    from safeshift.integration.lostbench import import_lostbench_dir

    converted = import_lostbench_dir(source, output)
    click.echo(f"Converted {len(converted)} scenarios to {output}")


if __name__ == "__main__":
    main()
