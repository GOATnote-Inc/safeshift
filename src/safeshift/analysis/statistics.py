"""Statistical utilities — Wilson CI, bootstrap, effect size."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class WilsonCI:
    """Wilson score confidence interval for a binomial proportion."""

    proportion: float
    lower: float
    upper: float
    n: int


@dataclass(frozen=True)
class BootstrapCI:
    """Bootstrap confidence interval."""

    mean: float
    lower: float
    upper: float
    n: int
    n_bootstrap: int


@dataclass(frozen=True)
class EffectSize:
    """Cohen's d effect size between two groups."""

    d: float
    interpretation: str  # "negligible", "small", "medium", "large"


def wilson_score(successes: int, n: int, z: float = 1.96) -> WilsonCI:
    """Compute Wilson score confidence interval.

    No scipy dependency — uses z directly.
    """
    if n == 0:
        return WilsonCI(0.0, 0.0, 1.0, 0)

    p = successes / n
    z2 = z * z
    denominator = 1 + z2 / n

    center = (p + z2 / (2 * n)) / denominator
    spread = z * math.sqrt((p * (1 - p) + z2 / (4 * n)) / n) / denominator

    lower = max(0.0, center - spread)
    upper = min(1.0, center + spread)

    return WilsonCI(
        proportion=round(p, 4),
        lower=round(lower, 4),
        upper=round(upper, 4),
        n=n,
    )


def bootstrap_ci(
    values: list[float],
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> BootstrapCI:
    """Compute bootstrap confidence interval for the mean."""
    if not values:
        return BootstrapCI(0.0, 0.0, 0.0, 0, n_bootstrap)

    rng = random.Random(seed)
    n = len(values)
    means = []

    for _ in range(n_bootstrap):
        sample = [rng.choice(values) for _ in range(n)]
        means.append(sum(sample) / n)

    means.sort()
    alpha = (1 - ci) / 2
    lower_idx = int(alpha * n_bootstrap)
    upper_idx = int((1 - alpha) * n_bootstrap) - 1

    return BootstrapCI(
        mean=round(sum(values) / n, 4),
        lower=round(means[lower_idx], 4),
        upper=round(means[upper_idx], 4),
        n=n,
        n_bootstrap=n_bootstrap,
    )


def cohens_d(group_a: list[float], group_b: list[float]) -> EffectSize:
    """Compute Cohen's d effect size between two groups."""
    if not group_a or not group_b:
        return EffectSize(0.0, "negligible")

    mean_a = sum(group_a) / len(group_a)
    mean_b = sum(group_b) / len(group_b)

    var_a = sum((x - mean_a) ** 2 for x in group_a) / max(len(group_a) - 1, 1)
    var_b = sum((x - mean_b) ** 2 for x in group_b) / max(len(group_b) - 1, 1)

    pooled_std = math.sqrt((var_a + var_b) / 2)
    if pooled_std == 0:
        return EffectSize(0.0, "negligible")

    d = (mean_a - mean_b) / pooled_std

    abs_d = abs(d)
    if abs_d < 0.2:
        interp = "negligible"
    elif abs_d < 0.5:
        interp = "small"
    elif abs_d < 0.8:
        interp = "medium"
    else:
        interp = "large"

    return EffectSize(d=round(d, 4), interpretation=interp)
