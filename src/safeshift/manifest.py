"""Results manifest — append-only experiment tracking."""

from __future__ import annotations

import datetime
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from safeshift import __version__


@dataclass(frozen=True)
class ManifestEntry:
    """A single experiment entry in the results manifest."""

    experiment: str  # e.g., "matrix-run", "single-scenario"
    date: str  # ISO 8601 (YYYY-MM-DD)
    model: str
    judge_model: str
    executor: str  # mock, api, vllm
    n_trials: int
    n_scenarios: int
    n_optimizations: int
    mean_safety: float
    class_a_count: int
    cliff_edges: int
    path: str  # results directory (relative)
    pipeline_version: str = __version__
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ManifestEntry:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def append_manifest(entry: ManifestEntry, manifest_path: Path | str) -> None:
    """Append an entry to the manifest YAML. Creates file if missing."""
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    entries: list[dict] = []
    if manifest_path.exists():
        with open(manifest_path) as f:
            existing = yaml.safe_load(f)
        if isinstance(existing, list):
            entries = existing

    entries.append(entry.to_dict())

    with open(manifest_path, "w") as f:
        yaml.dump(entries, f, default_flow_style=False, sort_keys=False)


def load_manifest(manifest_path: Path | str) -> list[ManifestEntry]:
    """Load all manifest entries from YAML."""
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        return []

    with open(manifest_path) as f:
        data = yaml.safe_load(f)

    if not isinstance(data, list):
        return []

    return [ManifestEntry.from_dict(d) for d in data]


def make_today() -> str:
    """Return today's date in ISO 8601 format."""
    return datetime.date.today().isoformat()
