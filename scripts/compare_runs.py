#!/usr/bin/env python3
"""Compare two SafeShift result sets."""

from pathlib import Path

import click


@click.command()
@click.argument("run_a", type=click.Path(exists=True))
@click.argument("run_b", type=click.Path(exists=True))
@click.option("--threshold", default=0.05, type=float, help="Significance threshold")
def main(run_a: str, run_b: str, threshold: float) -> None:
    """Compare two result sets and report differences."""
    from safeshift.analysis.regression import run_regression

    path_a = Path(run_a) / "grades.jsonl" if Path(run_a).is_dir() else Path(run_a)
    path_b = Path(run_b) / "grades.jsonl" if Path(run_b).is_dir() else Path(run_b)

    result = run_regression(path_a, path_b, threshold)

    click.echo(f"Run A: {path_a}")
    click.echo(f"Run B: {path_b}")
    click.echo()
    click.echo(result.message)
    click.echo(f"  A mean safety: {result.baseline_mean_safety:.4f}")
    click.echo(f"  B mean safety: {result.current_mean_safety:.4f}")
    click.echo(f"  Delta:         {result.delta:+.4f}")

    if result.new_class_a_failures:
        click.echo(f"  New Class A in B: {result.new_class_a_failures}")


if __name__ == "__main__":
    main()
