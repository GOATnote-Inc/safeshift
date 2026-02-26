"""Tests for CLI interface."""

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
