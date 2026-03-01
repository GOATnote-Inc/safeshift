"""Inter-grader agreement metrics for rubric calibration."""

from __future__ import annotations

from dataclasses import dataclass

from safeshift.grader import GradeResult
from safeshift.thresholds import CALIBRATION


@dataclass(frozen=True)
class AgreementMetrics:
    """Inter-grader agreement statistics."""

    cohens_kappa: float
    percent_agreement: float
    n_comparisons: int
    dimension_agreement: dict[str, float]


def cohens_kappa(ratings_a: list[int], ratings_b: list[int], k: int = 2) -> float:
    """Compute Cohen's kappa for two raters.

    Args:
        ratings_a: Integer ratings from rater A (0-based category indices).
        ratings_b: Integer ratings from rater B.
        k: Number of categories.
    """
    n = len(ratings_a)
    if n == 0:
        return 0.0

    # Build confusion matrix
    matrix = [[0] * k for _ in range(k)]
    for a, b in zip(ratings_a, ratings_b):
        matrix[a][b] += 1

    # Observed agreement
    po = sum(matrix[i][i] for i in range(k)) / n

    # Expected agreement
    pe = 0.0
    for i in range(k):
        row_sum = sum(matrix[i])
        col_sum = sum(matrix[j][i] for j in range(k))
        pe += (row_sum * col_sum) / (n * n)

    if pe == 1.0:
        return 1.0
    return (po - pe) / (1.0 - pe)


def compute_agreement(
    grades_a: list[GradeResult],
    grades_b: list[GradeResult],
    threshold: float = CALIBRATION.pass_fail_cutoff,
) -> AgreementMetrics:
    """Compute agreement between two sets of grades (e.g., two judges)."""
    assert len(grades_a) == len(grades_b), "Grade lists must have same length"

    n = len(grades_a)
    if n == 0:
        return AgreementMetrics(0.0, 0.0, 0, {})

    # Binary pass/fail classification
    pass_a = [1 if g.overall_score >= threshold else 0 for g in grades_a]
    pass_b = [1 if g.overall_score >= threshold else 0 for g in grades_b]

    kappa = cohens_kappa(pass_a, pass_b, k=2)
    agree = sum(a == b for a, b in zip(pass_a, pass_b)) / n

    # Per-dimension agreement (within 0.2 tolerance)
    dim_agree: dict[str, float] = {}
    dim_names = set()
    for g in grades_a:
        for d in g.dimensions:
            dim_names.add(d.dimension)

    for dim_name in dim_names:
        matches = 0
        count = 0
        for ga, gb in zip(grades_a, grades_b):
            score_a = next((d.score for d in ga.dimensions if d.dimension == dim_name), None)
            score_b = next((d.score for d in gb.dimensions if d.dimension == dim_name), None)
            if score_a is not None and score_b is not None:
                count += 1
                if abs(score_a - score_b) <= CALIBRATION.dimension_tolerance:
                    matches += 1
        dim_agree[dim_name] = matches / count if count > 0 else 0.0

    return AgreementMetrics(
        cohens_kappa=round(kappa, 4),
        percent_agreement=round(agree, 4),
        n_comparisons=n,
        dimension_agreement=dim_agree,
    )
