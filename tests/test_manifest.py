"""Tests for the results manifest module."""

from __future__ import annotations

import pytest
import yaml

from safeshift.manifest import ManifestEntry, append_manifest, load_manifest


def _make_entry(**kwargs) -> ManifestEntry:
    """Create a ManifestEntry with sensible defaults."""
    defaults = {
        "experiment": "matrix-run",
        "date": "2026-03-01",
        "model": "mock-model",
        "judge_model": "gpt-4o",
        "executor": "mock",
        "n_trials": 1,
        "n_scenarios": 5,
        "n_optimizations": 3,
        "mean_safety": 0.85,
        "class_a_count": 0,
        "cliff_edges": 0,
        "path": "results/test-run",
    }
    defaults.update(kwargs)
    return ManifestEntry(**defaults)


class TestManifestEntry:
    def test_to_dict_round_trip(self):
        entry = _make_entry()
        d = entry.to_dict()
        restored = ManifestEntry.from_dict(d)
        assert restored == entry

    def test_from_dict_ignores_extra_keys(self):
        d = _make_entry().to_dict()
        d["extra_field"] = "ignored"
        entry = ManifestEntry.from_dict(d)
        assert entry.experiment == "matrix-run"

    def test_frozen(self):
        entry = _make_entry()
        with pytest.raises(AttributeError):
            entry.experiment = "changed"

    def test_default_values(self):
        entry = _make_entry()
        assert entry.note == ""
        assert entry.pipeline_version  # not empty


class TestAppendManifest:
    def test_creates_new_file(self, tmp_path):
        path = tmp_path / "index.yaml"
        entry = _make_entry()
        append_manifest(entry, path)

        assert path.exists()
        data = yaml.safe_load(path.read_text())
        assert len(data) == 1
        assert data[0]["experiment"] == "matrix-run"

    def test_appends_to_existing(self, tmp_path):
        path = tmp_path / "index.yaml"
        append_manifest(_make_entry(experiment="run-1"), path)
        append_manifest(_make_entry(experiment="run-2"), path)

        data = yaml.safe_load(path.read_text())
        assert len(data) == 2
        assert data[0]["experiment"] == "run-1"
        assert data[1]["experiment"] == "run-2"

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "deep" / "nested" / "index.yaml"
        append_manifest(_make_entry(), path)
        assert path.exists()

    def test_preserves_existing_entries(self, tmp_path):
        path = tmp_path / "index.yaml"
        # Write initial entry
        append_manifest(_make_entry(model="model-a"), path)
        # Append second
        append_manifest(_make_entry(model="model-b"), path)

        data = yaml.safe_load(path.read_text())
        assert data[0]["model"] == "model-a"
        assert data[1]["model"] == "model-b"


class TestLoadManifest:
    def test_load_empty_returns_empty(self, tmp_path):
        path = tmp_path / "missing.yaml"
        assert load_manifest(path) == []

    def test_load_round_trip(self, tmp_path):
        path = tmp_path / "index.yaml"
        entry1 = _make_entry(experiment="run-1")
        entry2 = _make_entry(experiment="run-2")
        append_manifest(entry1, path)
        append_manifest(entry2, path)

        loaded = load_manifest(path)
        assert len(loaded) == 2
        assert loaded[0].experiment == "run-1"
        assert loaded[1].experiment == "run-2"

    def test_load_preserves_all_fields(self, tmp_path):
        path = tmp_path / "index.yaml"
        entry = _make_entry(
            note="test note",
            mean_safety=0.9234,
            class_a_count=3,
        )
        append_manifest(entry, path)

        loaded = load_manifest(path)
        assert loaded[0].note == "test note"
        assert loaded[0].mean_safety == 0.9234
        assert loaded[0].class_a_count == 3

    def test_load_malformed_returns_empty(self, tmp_path):
        path = tmp_path / "index.yaml"
        path.write_text("just a string\n")
        assert load_manifest(path) == []
