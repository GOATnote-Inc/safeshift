#!/usr/bin/env python3
"""Standalone Pareto frontier plotting."""

from pathlib import Path

import click


@click.command()
@click.option("--results", required=True, type=click.Path(exists=True), help="Results directory")
@click.option("--output", "-o", default=None, type=str, help="Output plot path")
@click.option("--title", default="Safety vs Latency Pareto Frontier", help="Plot title")
def main(results: str, output: str | None, title: str) -> None:
    """Generate Pareto frontier plot from results."""
    from safeshift.analysis.pareto import build_pareto_points, compute_pareto_frontier, plot_pareto
    from safeshift.analysis.regression import load_grades

    results_dir = Path(results)
    grades_path = results_dir / "grades.jsonl" if results_dir.is_dir() else results_dir

    grades = load_grades(grades_path)
    click.echo(f"Loaded {len(grades)} grades")

    # Build latency map (simplified — real impl would use results.jsonl)
    latencies: dict[str, float] = {}
    for g in grades:
        latencies.setdefault(g.optimization, 300.0)
    latencies["baseline"] = 500.0

    points = build_pareto_points(grades, latencies)
    points = compute_pareto_frontier(points)

    out = output or str(results_dir / "pareto.png")
    plot_pareto(points, out, title=title)
    click.echo(f"Plot: {out}")


if __name__ == "__main__":
    main()
