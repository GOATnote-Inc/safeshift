"""Grading subsystem — multi-dimensional rubric assessment."""

from safeshift.grading.dimensions import DIMENSION_WEIGHTS, DIMENSIONS
from safeshift.grading.rubric import RubricGrader

__all__ = ["RubricGrader", "DIMENSIONS", "DIMENSION_WEIGHTS"]
