"""Pareto frontier computation and plotting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from safeshift.grader import GradeResult


@dataclass(frozen=True)
class ParetoPoint:
    """A point on the safety-performance plane."""

    optimization: str
    safety_score: float
    latency_ms: float
    throughput_tps: float | None = None
    memory_mb: float | None = None
    overall_score: float = 0.0
    is_pareto_optimal: bool = False
    scenario_group: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def compute_pareto_frontier(
    points: list[ParetoPoint],
    x_metric: str = "latency_ms",
    y_metric: str = "safety_score",
    minimize_x: bool = True,
    maximize_y: bool = True,
) -> list[ParetoPoint]:
    """Compute Pareto-optimal points.

    Default: minimize latency, maximize safety.
    Returns new ParetoPoint instances with is_pareto_optimal=True for optimal ones.
    """
    if not points:
        return []

    def get_val(p: ParetoPoint, metric: str) -> float:
        return getattr(p, metric, 0.0) or 0.0

    result = []
    for i, p in enumerate(points):
        dominated = False
        px = get_val(p, x_metric)
        py = get_val(p, y_metric)

        for j, q in enumerate(points):
            if i == j:
                continue
            qx = get_val(q, x_metric)
            qy = get_val(q, y_metric)

            x_better = (qx <= px) if minimize_x else (qx >= px)
            y_better = (qy >= py) if maximize_y else (qy <= py)
            x_strict = (qx < px) if minimize_x else (qx > px)
            y_strict = (qy > py) if maximize_y else (qy < py)

            if x_better and y_better and (x_strict or y_strict):
                dominated = True
                break

        result.append(
            ParetoPoint(
                optimization=p.optimization,
                safety_score=p.safety_score,
                latency_ms=p.latency_ms,
                throughput_tps=p.throughput_tps,
                memory_mb=p.memory_mb,
                overall_score=p.overall_score,
                is_pareto_optimal=not dominated,
                scenario_group=p.scenario_group,
                metadata=p.metadata,
            )
        )

    return result


def build_pareto_points(
    grades: list[GradeResult],
    latencies: dict[str, float],
    throughputs: dict[str, float | None] | None = None,
    scenario_group: str = "",
) -> list[ParetoPoint]:
    """Build ParetoPoints from grade results and latency measurements.

    Args:
        grades: Grade results, one per (scenario, optimization).
        latencies: Map of optimization -> mean latency_ms.
        throughputs: Optional map of optimization -> tokens/sec.
        scenario_group: Label for this group of scenarios.
    """
    # Group grades by optimization
    by_opt: dict[str, list[GradeResult]] = {}
    for g in grades:
        by_opt.setdefault(g.optimization, []).append(g)

    points = []
    for opt, opt_grades in by_opt.items():
        safety_scores = [g.safety_score for g in opt_grades]
        overall_scores = [g.overall_score for g in opt_grades]
        avg_safety = sum(safety_scores) / len(safety_scores) if safety_scores else 0.0
        avg_overall = sum(overall_scores) / len(overall_scores) if overall_scores else 0.0

        points.append(
            ParetoPoint(
                optimization=opt,
                safety_score=round(avg_safety, 4),
                latency_ms=latencies.get(opt, 0.0),
                throughput_tps=throughputs.get(opt) if throughputs else None,
                overall_score=round(avg_overall, 4),
                scenario_group=scenario_group,
            )
        )

    return points


def plot_pareto(
    points: list[ParetoPoint],
    output_path: str,
    title: str = "Safety vs Latency Pareto Frontier",
    x_label: str = "Latency (ms)",
    y_label: str = "Safety Score",
) -> str:
    """Generate a Pareto frontier plot.

    Returns the output file path.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 7))

    # Separate Pareto-optimal and dominated points
    optimal = [p for p in points if p.is_pareto_optimal]
    dominated = [p for p in points if not p.is_pareto_optimal]

    if dominated:
        ax.scatter(
            [p.latency_ms for p in dominated],
            [p.safety_score for p in dominated],
            c="gray",
            alpha=0.5,
            s=60,
            label="Dominated",
        )

    if optimal:
        ax.scatter(
            [p.latency_ms for p in optimal],
            [p.safety_score for p in optimal],
            c="red",
            s=100,
            zorder=5,
            label="Pareto-optimal",
        )
        # Connect Pareto frontier
        sorted_opt = sorted(optimal, key=lambda p: p.latency_ms)
        ax.plot(
            [p.latency_ms for p in sorted_opt],
            [p.safety_score for p in sorted_opt],
            "r--",
            alpha=0.7,
        )

    # Label all points
    for p in points:
        ax.annotate(
            p.optimization,
            (p.latency_ms, p.safety_score),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path
