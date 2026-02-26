"""Report generation — Markdown and JSON output."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from safeshift.analysis.degradation import CliffEdge, DegradationResult
from safeshift.analysis.pareto import ParetoPoint
from safeshift.grader import GradeResult


def generate_markdown_report(
    grades: list[GradeResult],
    degradation_results: list[DegradationResult],
    cliff_edges: list[CliffEdge],
    pareto_points: list[ParetoPoint],
    output_path: str | Path,
    title: str = "SafeShift Degradation Report",
    metadata: dict[str, Any] | None = None,
) -> str:
    """Generate a comprehensive Markdown degradation report."""
    lines = [
        f"# {title}",
        "",
        f"*Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*",
        "",
    ]

    if metadata:
        lines.extend(["## Run Metadata", ""])
        for k, v in metadata.items():
            lines.append(f"- **{k}:** {v}")
        lines.append("")

    # Summary
    lines.extend(["## Summary", ""])
    n_scenarios = len(set(g.scenario_id for g in grades))
    n_optimizations = len(set(g.optimization for g in grades))
    n_class_a = sum(1 for g in grades if g.failure_class.value == "A")
    lines.append(f"- **Scenarios evaluated:** {n_scenarios}")
    lines.append(f"- **Optimizations tested:** {n_optimizations}")
    lines.append(f"- **Total evaluations:** {len(grades)}")
    lines.append(f"- **Class A failures:** {n_class_a}")
    lines.append(f"- **Cliff-edges detected:** {len(cliff_edges)}")
    lines.append("")

    # Degradation table
    if degradation_results:
        lines.extend(
            [
                "## Degradation Analysis",
                "",
                "| Opt | Baseline | Optimized | Delta | Pass Rate CI | Effect Size | Cliff? |",
                "|---|---|---|---|---|---|---|",
            ]
        )
        for dr in sorted(degradation_results, key=lambda d: d.delta):
            cliff_mark = "YES" if dr.is_cliff_edge else ""
            wci = dr.wilson_ci
            ci_str = f"[{wci.lower:.2f}, {wci.upper:.2f}] (n={wci.n})"
            lines.append(
                f"| {dr.optimization} | {dr.baseline_safety:.3f} | "
                f"{dr.optimized_safety:.3f} | {dr.delta:+.3f} | "
                f"{ci_str} | "
                f"{dr.effect_size.d:+.2f} ({dr.effect_size.interpretation}) | {cliff_mark} |"
            )
        lines.append("")

        # Bootstrap CI on mean safety score
        lines.extend(
            [
                "### Mean Safety Score CI (Bootstrap)",
                "",
                "| Optimization | Mean Safety | 95% CI | n |",
                "|---|---|---|---|",
            ]
        )
        for dr in sorted(degradation_results, key=lambda d: d.delta):
            bci = dr.bootstrap_ci_score
            lines.append(
                f"| {dr.optimization} | {bci.mean:.3f} | "
                f"[{bci.lower:.3f}, {bci.upper:.3f}] | {bci.n} |"
            )
        lines.append("")

    # Cliff edges
    if cliff_edges:
        lines.extend(["## Cliff-Edge Warnings", ""])
        for ce in cliff_edges:
            lines.append(f"- **{ce.description}**")
        lines.append("")

    # Pareto frontier
    if pareto_points:
        optimal = [p for p in pareto_points if p.is_pareto_optimal]
        if optimal:
            lines.extend(["## Pareto-Optimal Configurations", ""])
            lines.append("| Optimization | Safety | Latency (ms) |")
            lines.append("|---|---|---|")
            for p in sorted(optimal, key=lambda x: x.latency_ms):
                lines.append(f"| {p.optimization} | {p.safety_score:.3f} | {p.latency_ms:.1f} |")
            lines.append("")

    # Failure class breakdown
    lines.extend(["## Failure Class Breakdown", ""])
    fc_counts: dict[str, int] = {}
    for g in grades:
        fc = g.failure_class.value
        fc_counts[fc] = fc_counts.get(fc, 0) + 1
    lines.append("| Class | Count | Description |")
    lines.append("|---|---|---|")
    class_desc = {
        "A": "Critical safety drop",
        "B": "Drift from baseline",
        "C": "Partial degradation",
        "D": "Latency violation",
        "E": "Hallucination",
        "none": "Pass",
    }
    for fc in ["A", "B", "C", "D", "E", "none"]:
        count = fc_counts.get(fc, 0)
        if count > 0:
            lines.append(f"| {fc} | {count} | {class_desc.get(fc, '')} |")
    lines.append("")

    content = "\n".join(lines)
    Path(output_path).write_text(content)
    return str(output_path)


def generate_json_report(
    grades: list[GradeResult],
    degradation_results: list[DegradationResult],
    cliff_edges: list[CliffEdge],
    pareto_points: list[ParetoPoint],
    output_path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Generate a JSON report for programmatic consumption."""
    report = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata or {},
        "summary": {
            "n_scenarios": len(set(g.scenario_id for g in grades)),
            "n_optimizations": len(set(g.optimization for g in grades)),
            "n_evaluations": len(grades),
            "n_class_a": sum(1 for g in grades if g.failure_class.value == "A"),
            "n_cliff_edges": len(cliff_edges),
        },
        "grades": [g.to_dict() for g in grades],
        "degradation": [
            {
                "optimization": dr.optimization,
                "baseline_safety": dr.baseline_safety,
                "optimized_safety": dr.optimized_safety,
                "delta": dr.delta,
                "effect_size_d": dr.effect_size.d,
                "effect_size_interpretation": dr.effect_size.interpretation,
                "wilson_ci": {
                    "proportion": dr.wilson_ci.proportion,
                    "lower": dr.wilson_ci.lower,
                    "upper": dr.wilson_ci.upper,
                    "n": dr.wilson_ci.n,
                },
                "bootstrap_ci": {
                    "mean": dr.bootstrap_ci_score.mean,
                    "lower": dr.bootstrap_ci_score.lower,
                    "upper": dr.bootstrap_ci_score.upper,
                    "n": dr.bootstrap_ci_score.n,
                },
                "is_cliff_edge": dr.is_cliff_edge,
                "failure_classes": dr.failure_classes,
                "n_scenarios": dr.n_scenarios,
            }
            for dr in degradation_results
        ],
        "cliff_edges": [
            {
                "optimization_a": ce.optimization_a,
                "optimization_b": ce.optimization_b,
                "latency_delta_pct": ce.latency_delta_pct,
                "safety_delta": ce.safety_delta,
                "cliff_ratio": ce.cliff_ratio,
                "description": ce.description,
            }
            for ce in cliff_edges
        ],
        "pareto": [
            {
                "optimization": p.optimization,
                "safety_score": p.safety_score,
                "latency_ms": p.latency_ms,
                "is_pareto_optimal": p.is_pareto_optimal,
            }
            for p in pareto_points
        ],
    }

    Path(output_path).write_text(json.dumps(report, indent=2))
    return str(output_path)
