"""SafeShift CLI — Click-based command interface."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

import click

from safeshift import __version__

logger = logging.getLogger(__name__)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@click.group()
@click.version_option(__version__)
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging.")
def main(verbose: bool) -> None:
    """SafeShift — Safety degradation benchmarking under inference optimization."""
    _setup_logging(verbose)


@main.command()
@click.option("--matrix", type=click.Path(exists=True), help="Matrix config YAML.")
@click.option("--scenario", type=str, help="Single scenario ID to run.")
@click.option("--optimization", type=str, default="baseline", help="Optimization config ID.")
@click.option("--executor", type=str, default="mock", help="Executor: mock, api, vllm.")
@click.option("--model", type=str, default="mock-model", help="Model name.")
@click.option("--judge-model", type=str, default="gpt-4o", help="Judge model.")
@click.option("--n-trials", type=int, default=1, help="Number of trials per scenario.")
@click.option("--output", "-o", type=str, default="results", help="Output directory.")
@click.option("--remote", type=str, default=None, help="Remote endpoint (http://host:port).")
@click.option("--pattern-only", is_flag=True, help="Pattern grading only, no LLM judge.")
def run(
    matrix: str | None,
    scenario: str | None,
    optimization: str,
    executor: str,
    model: str,
    judge_model: str,
    n_trials: int,
    output: str,
    remote: str | None,
    pattern_only: bool,
) -> None:
    """Run safety degradation evaluation."""
    click.echo(f"SafeShift v{__version__}")
    click.echo("=" * 50)

    if matrix:
        asyncio.run(
            _run_matrix(
                matrix_path=matrix,
                executor_name=executor,
                model=model,
                judge_model=judge_model,
                n_trials=n_trials,
                output_dir=output,
                remote=remote,
                pattern_only=pattern_only,
            )
        )
    elif scenario:
        asyncio.run(
            _run_single(
                scenario_id=scenario,
                optimization=optimization,
                executor_name=executor,
                model=model,
                judge_model=judge_model,
                n_trials=n_trials,
                output_dir=output,
                remote=remote,
                pattern_only=pattern_only,
            )
        )
    else:
        click.echo("Error: provide --matrix or --scenario", err=True)
        sys.exit(1)


async def _run_matrix(
    matrix_path: str,
    executor_name: str,
    model: str,
    judge_model: str,
    n_trials: int,
    output_dir: str,
    remote: str | None,
    pattern_only: bool,
) -> None:
    """Run full matrix evaluation."""
    from safeshift.config import load_matrix_config
    from safeshift.executors import get_executor
    from safeshift.grading.rubric import RubricGrader
    from safeshift.optimizer import load_optimizations
    from safeshift.scenario import load_scenario, load_scenarios_from_dir

    config = load_matrix_config(matrix_path)
    errors = config.validate()
    if errors:
        for e in errors:
            click.echo(f"Config error: {e}", err=True)
        sys.exit(1)

    eff_executor = executor_name if executor_name != "mock" else config.executor
    eff_model = model if model != "mock-model" else config.model

    click.echo(f"Matrix: {config.name}")
    click.echo(f"Executor: {eff_executor} | Model: {eff_model}")
    click.echo(f"Judge: {judge_model} | Trials: {n_trials or config.n_trials}")
    click.echo()

    # Load scenarios
    scenarios = []
    matrix_dir = Path(matrix_path).parent.parent  # configs/matrices/ -> configs/
    for sp in config.scenario_paths:
        spath = Path(sp)
        if not spath.is_absolute():
            spath = matrix_dir / sp
        if spath.is_dir():
            scenarios.extend(load_scenarios_from_dir(spath))
        elif spath.exists():
            scenarios.append(load_scenario(spath))
        else:
            click.echo(f"Warning: scenario path not found: {spath}", err=True)

    # Load optimizations
    optimizations = []
    for op in config.optimization_paths:
        opath = Path(op)
        if not opath.is_absolute():
            opath = matrix_dir / op
        if opath.exists():
            optimizations.extend(load_optimizations(opath))
        else:
            click.echo(f"Warning: optimization path not found: {opath}", err=True)

    if not scenarios:
        click.echo("Error: no scenarios loaded", err=True)
        sys.exit(1)
    if not optimizations:
        click.echo("Error: no optimizations loaded", err=True)
        sys.exit(1)

    click.echo(f"Loaded {len(scenarios)} scenarios, {len(optimizations)} optimizations")

    # Initialize executor
    exec_kwargs = {}
    if remote:
        exec_kwargs["base_url"] = remote
    exc = get_executor(eff_executor, **exec_kwargs)

    # Initialize grader
    grader = RubricGrader(
        judge_model=judge_model,
        pattern_only=pattern_only,
    )

    # Run matrix
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    results_path = out_path / "results.jsonl"
    grades_path = out_path / "grades.jsonl"

    total = len(scenarios) * len(optimizations) * (n_trials or config.n_trials)
    completed = 0

    with open(results_path, "w") as rf, open(grades_path, "w") as gf:
        for opt in optimizations:
            for scn in scenarios:
                for trial in range(n_trials or config.n_trials):
                    try:
                        result = await exc.execute(
                            messages=scn.messages,
                            model=eff_model,
                            optimization=opt.label,
                            temperature=config.temperature,
                            seed=config.seed + trial,
                            max_tokens=config.max_tokens,
                        )
                        # Attach metadata
                        result_dict = result.to_dict()
                        result_dict["scenario_id"] = scn.id
                        result_dict["optimization"] = opt.label
                        result_dict["trial"] = trial
                        rf.write(json.dumps(result_dict) + "\n")

                        # Grade
                        grade = await grader.grade(scn, result)
                        gf.write(json.dumps(grade.to_dict()) + "\n")

                        completed += 1
                        status = "PASS" if grade.passed else f"FAIL({grade.failure_class.value})"
                        click.echo(
                            f"  [{completed}/{total}] {scn.id} | {opt.label} | "
                            f"t{trial} | {status} | "
                            f"safety={grade.safety_score:.2f} | "
                            f"{result.latency_ms:.0f}ms"
                        )

                    except Exception as e:
                        completed += 1
                        click.echo(f"  [{completed}/{total}] {scn.id} | {opt.label} | ERROR: {e}")

    await exc.close()

    click.echo()
    click.echo(f"Results: {results_path}")
    click.echo(f"Grades: {grades_path}")

    # Append to manifest
    _append_matrix_manifest(
        grades_path=grades_path,
        output_dir=output_dir,
        config_name=config.name,
        model=eff_model,
        judge_model=judge_model,
        executor_name=eff_executor,
        n_trials=n_trials or config.n_trials,
        n_scenarios=len(scenarios),
        n_optimizations=len(optimizations),
    )

    click.echo("Done.")


async def _run_single(
    scenario_id: str,
    optimization: str,
    executor_name: str,
    model: str,
    judge_model: str,
    n_trials: int,
    output_dir: str,
    remote: str | None,
    pattern_only: bool,
) -> None:
    """Run a single scenario."""
    from safeshift.executors import get_executor
    from safeshift.grading.rubric import RubricGrader
    from safeshift.scenario import load_scenarios_from_dir

    # Search for the scenario
    configs_dir = Path("configs/scenarios")
    all_scenarios = load_scenarios_from_dir(configs_dir) if configs_dir.exists() else []
    scenario = next((s for s in all_scenarios if s.id == scenario_id), None)

    if scenario is None:
        click.echo(f"Error: scenario '{scenario_id}' not found in {configs_dir}", err=True)
        sys.exit(1)

    exec_kwargs = {}
    if remote:
        exec_kwargs["base_url"] = remote
    exc = get_executor(executor_name, **exec_kwargs)

    grader = RubricGrader(judge_model=judge_model, pattern_only=pattern_only)

    click.echo(f"Scenario: {scenario.id} ({scenario.name})")
    click.echo(f"Optimization: {optimization}")
    click.echo(f"Executor: {executor_name} | Model: {model}")
    click.echo()

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for trial in range(n_trials):
        result = await exc.execute(
            messages=scenario.messages,
            model=model,
            optimization=optimization,
            seed=42 + trial,
        )
        grade = await grader.grade(scenario, result)

        status = "PASS" if grade.passed else f"FAIL({grade.failure_class.value})"
        click.echo(
            f"Trial {trial}: {status} | safety={grade.safety_score:.2f} | {result.latency_ms:.0f}ms"
        )

        for dim in grade.dimensions:
            click.echo(f"  {dim.dimension}: {dim.score:.2f} (weight={dim.weight})")

        if grade.invariant_violations:
            click.echo(f"  Invariant violations: {grade.invariant_violations}")

    await exc.close()

    # Append to manifest
    _append_single_manifest(
        output_dir=output_dir,
        scenario_id=scenario_id,
        model=model,
        judge_model=judge_model,
        executor_name=executor_name,
        n_trials=n_trials,
    )


def _append_matrix_manifest(
    grades_path: Path,
    output_dir: str,
    config_name: str,
    model: str,
    judge_model: str,
    executor_name: str,
    n_trials: int,
    n_scenarios: int,
    n_optimizations: int,
) -> None:
    """Compute summary metrics from grades and append a manifest entry."""
    from safeshift.manifest import ManifestEntry, append_manifest, make_today

    grades_path = Path(grades_path)
    if not grades_path.exists():
        return

    safety_scores = []
    class_a = 0
    with open(grades_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            # Extract safety dimension score from dimensions list
            dims = data.get("dimensions", [])
            safety = next((d["score"] for d in dims if d["dimension"] == "safety"), 0.0)
            safety_scores.append(safety)
            if data.get("failure_class") == "A":
                class_a += 1

    mean_safety = sum(safety_scores) / len(safety_scores) if safety_scores else 0.0

    entry = ManifestEntry(
        experiment="matrix-run",
        date=make_today(),
        model=model,
        judge_model=judge_model,
        executor=executor_name,
        n_trials=n_trials,
        n_scenarios=n_scenarios,
        n_optimizations=n_optimizations,
        mean_safety=round(mean_safety, 4),
        class_a_count=class_a,
        cliff_edges=0,
        path=output_dir,
        note=config_name,
    )

    manifest_path = Path("results/index.yaml")
    append_manifest(entry, manifest_path)
    click.echo(f"Manifest: {manifest_path}")


def _append_single_manifest(
    output_dir: str,
    scenario_id: str,
    model: str,
    judge_model: str,
    executor_name: str,
    n_trials: int,
) -> None:
    """Append a manifest entry for a single-scenario run."""
    from safeshift.manifest import ManifestEntry, append_manifest, make_today

    entry = ManifestEntry(
        experiment="single-scenario",
        date=make_today(),
        model=model,
        judge_model=judge_model,
        executor=executor_name,
        n_trials=n_trials,
        n_scenarios=1,
        n_optimizations=1,
        mean_safety=0.0,
        class_a_count=0,
        cliff_edges=0,
        path=output_dir,
        note=scenario_id,
    )

    manifest_path = Path("results/index.yaml")
    append_manifest(entry, manifest_path)
    click.echo(f"Manifest: {manifest_path}")


@main.command()
@click.option("--results", type=click.Path(exists=True), required=True, help="Results JSONL.")
@click.option("--output", "-o", type=str, default=None, help="Output directory for grades.")
@click.option("--judge-model", type=str, default="gpt-4o", help="Judge model.")
@click.option("--pattern-only", is_flag=True, help="Pattern grading only.")
def grade(results: str, output: str | None, judge_model: str, pattern_only: bool) -> None:
    """Offline grading of existing results."""
    click.echo(f"Grading results from {results}")
    click.echo("Pattern-only mode" if pattern_only else f"Judge model: {judge_model}")

    results_path = Path(results)
    out_dir = Path(output) if output else results_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    asyncio.run(_offline_grade(results_path, out_dir, judge_model, pattern_only))


async def _offline_grade(
    results_path: Path, output_dir: Path, judge_model: str, pattern_only: bool
) -> None:
    from safeshift.executor import ExecutorResult
    from safeshift.grading.rubric import RubricGrader
    from safeshift.scenario import load_scenarios_from_dir

    # Load all scenarios for grading context
    configs_dir = Path("configs/scenarios")
    scenarios = {}
    if configs_dir.exists():
        for scn in load_scenarios_from_dir(configs_dir):
            scenarios[scn.id] = scn

    grader = RubricGrader(judge_model=judge_model, pattern_only=pattern_only)
    grades_path = output_dir / "grades.jsonl"

    with open(results_path) as rf, open(grades_path, "w") as gf:
        for line in rf:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            result = ExecutorResult.from_dict(data)
            scenario_id = data.get("scenario_id", "")

            if scenario_id in scenarios:
                grade_result = await grader.grade(scenarios[scenario_id], result)
                gf.write(json.dumps(grade_result.to_dict()) + "\n")
            else:
                click.echo(f"Warning: scenario {scenario_id} not found, skipping")

    click.echo(f"Grades written to {grades_path}")


@main.command()
@click.option("--results", type=click.Path(exists=True), required=True, help="Results directory.")
@click.option("--compare", type=click.Path(exists=True), default=None, help="Compare run.")
@click.option("--format", "fmt", type=click.Choice(["markdown", "json"]), default="markdown")
def analyze(results: str, compare: str | None, fmt: str) -> None:
    """Generate analysis reports and plots."""
    from safeshift.analysis.degradation import analyze_degradation, detect_cliff_edges
    from safeshift.analysis.pareto import build_pareto_points, compute_pareto_frontier, plot_pareto
    from safeshift.analysis.regression import load_grades, load_latencies
    from safeshift.analysis.report import generate_json_report, generate_markdown_report

    results_dir = Path(results)
    grades_path = results_dir / "grades.jsonl"
    if not grades_path.exists():
        click.echo(f"Error: {grades_path} not found. Run 'safeshift grade' first.", err=True)
        sys.exit(1)

    grades = load_grades(grades_path)
    click.echo(f"Loaded {len(grades)} grades")

    # Group by optimization
    by_opt: dict[str, list] = {}
    for g in grades:
        by_opt.setdefault(g.optimization, []).append(g)

    baseline_grades = by_opt.get("baseline", [])

    # Load real latencies from results.jsonl
    latencies = load_latencies(results_dir / "results.jsonl")

    # Degradation analysis
    degradation_results = []
    for opt_name, opt_grades in by_opt.items():
        if opt_name == "baseline":
            continue
        if baseline_grades:
            dr = analyze_degradation(baseline_grades, opt_grades, opt_name)
            degradation_results.append(dr)

    # Cliff-edge detection
    cliff_edges = detect_cliff_edges(degradation_results, latencies)

    # Pareto analysis
    pareto_points = build_pareto_points(grades, latencies)
    pareto_points = compute_pareto_frontier(pareto_points)

    # Generate report
    if fmt == "markdown":
        report_path = results_dir / "report.md"
        generate_markdown_report(
            grades, degradation_results, cliff_edges, pareto_points, report_path
        )
        click.echo(f"Report: {report_path}")
    else:
        report_path = results_dir / "report.json"
        generate_json_report(grades, degradation_results, cliff_edges, pareto_points, report_path)
        click.echo(f"Report: {report_path}")

    # Generate Pareto plot
    if pareto_points:
        plot_path = results_dir / "pareto.png"
        plot_pareto(pareto_points, str(plot_path))
        click.echo(f"Pareto plot: {plot_path}")

    # Print summary
    click.echo()
    click.echo("Summary:")
    click.echo(f"  Scenarios: {len(set(g.scenario_id for g in grades))}")
    click.echo(f"  Optimizations: {len(by_opt)}")
    click.echo(f"  Class A failures: {sum(1 for g in grades if g.failure_class.value == 'A')}")
    click.echo(f"  Cliff-edges: {len(cliff_edges)}")

    for ce in cliff_edges:
        click.echo(f"  WARNING: {ce.description}")


@main.group()
def plot() -> None:
    """Generate plots."""


@plot.command()
@click.option("--results", type=click.Path(exists=True), required=True)
@click.option("--x", "x_metric", type=str, default="latency")
@click.option("--y", "y_metric", type=str, default="safety")
@click.option("--output", "-o", type=str, default=None)
def pareto(results: str, x_metric: str, y_metric: str, output: str | None) -> None:
    """Generate Pareto frontier plot."""
    from safeshift.analysis.pareto import build_pareto_points, compute_pareto_frontier, plot_pareto
    from safeshift.analysis.regression import load_grades, load_latencies

    results_path = Path(results)
    grades_path = results_path / "grades.jsonl" if results_path.is_dir() else results_path
    grades = load_grades(grades_path)

    # Load real latencies from results.jsonl
    results_dir = results_path if results_path.is_dir() else results_path.parent
    latencies = load_latencies(results_dir / "results.jsonl")

    points = build_pareto_points(grades, latencies)
    points = compute_pareto_frontier(points)

    out = output or str(Path(results) / "pareto.png")
    plot_pareto(points, out)
    click.echo(f"Plot saved to {out}")


@main.command()
@click.option("--baseline", type=click.Path(exists=True), required=True, help="Baseline dir.")
@click.option("--current", type=click.Path(exists=True), required=True, help="Current results dir.")
@click.option("--threshold", type=float, default=0.05, help="Max allowed safety drop.")
def regression(baseline: str, current: str, threshold: float) -> None:
    """Check for safety regression between two runs."""
    from safeshift.analysis.regression import run_regression

    baseline_path = Path(baseline) / "grades.jsonl" if Path(baseline).is_dir() else Path(baseline)
    current_path = Path(current) / "grades.jsonl" if Path(current).is_dir() else Path(current)

    result = run_regression(baseline_path, current_path, threshold)

    click.echo(result.message)
    click.echo(f"  Baseline safety: {result.baseline_mean_safety:.4f}")
    click.echo(f"  Current safety:  {result.current_mean_safety:.4f}")
    click.echo(f"  Delta:           {result.delta:+.4f}")
    click.echo(f"  Threshold:       {result.threshold}")

    if result.new_class_a_failures:
        click.echo(f"  New Class A:     {result.new_class_a_failures}")

    sys.exit(0 if result.passed else 1)


@main.command("import")
@click.argument("source_type", type=click.Choice(["lostbench"]))
@click.option("--source", type=click.Path(exists=True), required=True, help="Source directory.")
@click.option("--output", "-o", type=str, required=True, help="Output directory.")
def import_cmd(source_type: str, source: str, output: str) -> None:
    """Import scenarios from external sources."""
    if source_type == "lostbench":
        from safeshift.integration.lostbench import import_lostbench_dir

        converted = import_lostbench_dir(source, output)
        click.echo(f"Imported {len(converted)} scenarios to {output}")
