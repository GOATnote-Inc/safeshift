"""Tests for CLI interface."""

import json
from pathlib import Path

import yaml
from click.testing import CliRunner

from safeshift.cli import main


class TestCLI:
    def test_version(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "SafeShift" in result.output

    def test_run_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["run", "--help"])
        assert result.exit_code == 0
        assert "--matrix" in result.output
        assert "--executor" in result.output

    def test_analyze_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["analyze", "--help"])
        assert result.exit_code == 0
        assert "--results" in result.output

    def test_regression_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["regression", "--help"])
        assert result.exit_code == 0
        assert "--baseline" in result.output
        assert "--threshold" in result.output

    def test_grade_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["grade", "--help"])
        assert result.exit_code == 0
        assert "--results" in result.output

    def test_run_no_args(self):
        runner = CliRunner()
        result = runner.invoke(main, ["run"])
        assert result.exit_code != 0

    def test_import_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["import", "--help"])
        assert result.exit_code == 0
        assert "lostbench" in result.output


REPO_ROOT = Path(__file__).resolve().parents[1]

_JUDGE_JSON = json.dumps(
    {
        "dimensions": {
            "safety": {"score": 0.9, "evidence": "maintains urgency"},
            "accuracy": {"score": 0.9, "evidence": "correct plan"},
            "completeness": {"score": 0.8, "evidence": "covers key actions"},
            "timeliness": {"score": 1.0, "evidence": "within budget"},
            "specificity": {"score": 0.85, "evidence": "actionable"},
        },
        "failure_class": "none",
        "failure_reason": "",
        "invariant_violations": [],
    }
)


class FakeJudgeProvider:
    """In-process judge double — no network, valid structured JSON."""

    def __init__(self):
        self.calls = 0

    async def chat(self, messages, model, temperature=0.0, seed=42):
        self.calls += 1
        return _JUDGE_JSON


class TestJudgeWiring:
    """The CLI must pass a judge provider through when not --pattern-only."""

    def test_run_matrix_passes_judge_provider_through(self, tmp_path, monkeypatch):
        import safeshift.providers as providers

        fake = FakeJudgeProvider()
        monkeypatch.setattr(providers, "build_judge_provider", lambda judge_model: fake)
        monkeypatch.chdir(tmp_path)

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "run",
                "--matrix",
                str(REPO_ROOT / "configs" / "matrices" / "quick_matrix.yaml"),
                "--executor",
                "mock",
                "--output",
                "out",
            ],
        )
        assert result.exit_code == 0, result.output
        assert fake.calls > 0, "judge provider was never invoked"

        grades_path = tmp_path / "out" / "grades.jsonl"
        grades = [json.loads(line) for line in grades_path.read_text().splitlines() if line]
        assert grades
        for g in grades:
            assert any(d["layer"] == 2 for d in g["dimensions"]), (
                "expected judge (layer 2) scores in every grade"
            )

        manifest = yaml.safe_load((tmp_path / "results" / "index.yaml").read_text())
        entry = manifest[-1]
        assert entry["pattern_only"] is False
        assert entry["judged_fraction"] == 1.0
        assert entry["judge_model"] is not None

    def test_run_matrix_without_judge_key_fails_closed(self, tmp_path, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.chdir(tmp_path)

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "run",
                "--matrix",
                str(REPO_ROOT / "configs" / "matrices" / "quick_matrix.yaml"),
                "--executor",
                "mock",
                "--judge-model",
                "gpt-5.5",
                "--output",
                "out",
            ],
        )
        assert result.exit_code == 2, result.output
        assert "Judge preflight failed" in result.output
        assert not (tmp_path / "out").exists(), "no eval work should start without a judge"

    def test_run_matrix_pattern_only_is_explicit_opt_out(self, tmp_path, monkeypatch):
        """README quick start: mock + --pattern-only works with no API keys."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.chdir(tmp_path)

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "run",
                "--matrix",
                str(REPO_ROOT / "configs" / "matrices" / "quick_matrix.yaml"),
                "--executor",
                "mock",
                "--pattern-only",
                "--output",
                "results/smoke",
            ],
        )
        assert result.exit_code == 0, result.output
        assert (tmp_path / "results" / "smoke" / "grades.jsonl").exists()

        manifest = yaml.safe_load((tmp_path / "results" / "index.yaml").read_text())
        entry = manifest[-1]
        assert entry["pattern_only"] is True
        assert entry["judge_model"] is None, "manifest must not claim a judge that never ran"
        assert entry["git_sha"]


class TestOptimizationSupportRefusal:
    """Executors that cannot vary the optimization must be refused loudly."""

    def test_run_matrix_refuses_vllm_multi_optimization(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "run",
                "--matrix",
                str(REPO_ROOT / "configs" / "matrices" / "full_quantization_matrix.yaml"),
                "--output",
                "out",
            ],
        )
        assert result.exit_code == 2, result.output
        assert "cannot apply optimizations" in result.output
        assert not (tmp_path / "out").exists()

    def test_run_single_refuses_vllm_optimization(self, monkeypatch):
        monkeypatch.chdir(REPO_ROOT)
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "run",
                "--scenario",
                "SCN-C-001",
                "--optimization",
                "quantization=int4",
                "--executor",
                "vllm",
            ],
        )
        assert result.exit_code == 2, result.output
        assert "cannot apply optimization" in result.output
