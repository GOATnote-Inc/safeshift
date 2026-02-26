"""Result collection and validation from remote runs — stub for v0.1."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from safeshift.executor import ExecutorResult

logger = logging.getLogger(__name__)


def validate_results_jsonl(path: str | Path) -> tuple[list[ExecutorResult], list[str]]:
    """Validate a results JSONL file. Returns (valid_results, errors)."""
    path = Path(path)
    results = []
    errors = []

    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                result = ExecutorResult.from_dict(data)
                results.append(result)
            except (json.JSONDecodeError, KeyError) as e:
                errors.append(f"Line {i}: {e}")

    logger.info("Validated %s: %d results, %d errors", path, len(results), len(errors))
    return results, errors
