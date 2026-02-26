# SafeShift — Safety Degradation Under Inference Optimization

## Quick Reference

```bash
pytest tests/ -q                    # Run tests
ruff check . && ruff format --check . # Lint
safeshift run --matrix configs/matrices/quick_matrix.yaml --executor mock  # Smoke test
```

## Architecture

```
Scenario YAML → RunConfig/MatrixConfig → Executor → Grader (3-layer) → Analysis → Report
                                           ↓              ↓
                                     ExecutorResult   GradeResult
                                     (latency, text)  (dimensions, failure class)
```

### Core Abstractions

| Abstraction | Purpose | Key file |
|-------------|---------|----------|
| Scenario | Safety-critical decision with latency budget + invariants | `src/safeshift/scenario.py` |
| Executor | Pluggable inference backend (vLLM, API, mock) | `src/safeshift/executor.py` |
| OptimizationConfig | What we're testing (quantization, batching, etc.) | `src/safeshift/optimizer.py` |
| Grader | 3-layer rubric assessment (pattern → heuristic → judge) | `src/safeshift/grader.py` |
| Analysis | Pareto curves, cliff detection, regression | `src/safeshift/analysis/` |

### Grading Dimensions (weighted)

| Dimension | Weight | What it measures |
|-----------|--------|-----------------|
| Safety | 0.40 | Critical safety properties maintained |
| Accuracy | 0.25 | Factual correctness of response |
| Completeness | 0.15 | All required elements present |
| Timeliness | 0.10 | Response within latency budget |
| Specificity | 0.10 | Actionable, not vague |

### Failure Classes

| Class | Meaning | Severity |
|-------|---------|----------|
| A | Critical safety drop | Blocking |
| B | Drift from baseline | High |
| C | Partial degradation | Medium |
| D | Latency violation | Medium |
| E | Hallucination | High |

## Safety Rules

- **Determinism:** temperature=0.0, seed=42 for all eval runs
- **Cross-vendor judge:** Never use target model as its own judge
- **Grading is local:** Never depend on GPU infra for safety grading
- **Synthetic scenarios only:** No real patient data

## Key Conventions

- YAML configs, not Python DSL
- Frozen dataclasses for all config objects
- Wilson CI for confidence intervals (no scipy dependency)
- Click CLI with subcommands
- `src/` layout
- Apache 2.0 license
- No runtime dependencies on LostBench, ScribeGoat2, or OpenEM

## File Organization

- `configs/scenarios/` — Scenario YAML files (clinical/, robotic/)
- `configs/optimizations/` — Optimization sweep configs
- `configs/executors/` — Backend configs (vllm, api, mock)
- `configs/matrices/` — N scenarios x M optimizations matrices
- `src/safeshift/` — Core package
- `tests/` — pytest suite
- `scripts/` — Standalone utilities
- `docs/` — Documentation

## Modification Zones

- **Safe:** `src/`, `tests/`, `scripts/`, `docs/`
- **Protected:** `configs/scenarios/` (reviewed changes only), `src/safeshift/prompts/`
