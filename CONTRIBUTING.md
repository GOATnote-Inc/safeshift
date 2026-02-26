# Contributing to SafeShift

Thanks for your interest in SafeShift. This guide covers the main ways to contribute.

## Adding Scenarios (Easiest Contribution)

Scenarios are standalone YAML files in `configs/scenarios/{clinical,robotic}/`. Adding one requires no code changes.

### 1. Pick a domain

- `clinical/` — Emergency medicine cases (ESI-1/2) where delayed response risks patient harm
- `robotic/` — Industrial/autonomous robot events where degraded response risks physical harm
- New domains welcome — open an issue first to discuss

### 2. Create the YAML file

Use an existing scenario as a template. Required fields:

```yaml
id: SCN-C-016              # SCN-{C|R}-{NNN}
name: Your Scenario Name
domain: clinical            # or "robotic"
description: |
  Brief clinical/situational description.
messages:
  - role: system
    content: |
      System prompt establishing the model's role.
  - role: user
    content: |
      The actual scenario prompt with vitals, context, etc.
latency_budget:
  target_ms: 500            # Ideal response time
  acceptable_ms: 1000       # Tolerable
  critical_ms: 2000         # Safety concern if exceeded
safety_invariants:
  - name: invariant_name
    description: What this checks
    check_type: regex       # regex | contains | not_contains | judge
    pattern: "\\b(keyword1|keyword2)\\b"
    severity: 1.0           # 0.0–1.0
constraint:
  description: Why time matters
  time_pressure: immediate  # immediate | minutes | hours
  consequence_of_delay: What happens if the model is slow
rubric: rubric_clinical     # or rubric_robotic
expected_action: What the correct response should recommend
severity: 1.0
tags: [relevant, tags]
```

### 3. Test it

```bash
safeshift run --scenario SCN-C-016 --executor mock
pytest tests/ -q
```

### 4. Submit a PR

Scenario PRs need:
- The YAML file
- A brief description of why this scenario is safety-critical
- Source/reference for the clinical or engineering basis (if applicable)

## Adding Executor Backends

Executors live in `src/safeshift/executors/`. To add one:

1. Create `src/safeshift/executors/your_backend.py`
2. Subclass `Executor` from `src/safeshift/executor.py`
3. Implement `async execute(scenario, optimization_config) -> ExecutorResult`
4. Add a YAML config in `configs/executors/`
5. Register it in the CLI (`src/safeshift/cli.py`, the executor dispatch dict)
6. Add tests in `tests/test_executor.py`

The `MockExecutor` is a good reference implementation — it shows the full interface including timing, token counting, and degradation simulation.

## Adding Grading Dimensions

Grading dimensions are defined in `src/safeshift/grading/dimensions.py`. The current five (Safety, Accuracy, Completeness, Timeliness, Specificity) cover most cases, but domain-specific dimensions may be needed.

To add one:
1. Add the constant to `dimensions.py`
2. Add grading logic in `rubric.py`
3. Update prompts in `src/safeshift/prompts/` if the LLM judge needs to evaluate it
4. Update tests in `tests/test_grader.py`
5. Open an issue first — dimension weights affect all existing results

## Code Style

- **Formatter/linter:** ruff (`make lint` to check, `make format` to fix)
- **Line length:** 100 characters
- **Dataclasses:** Use `@dataclass(frozen=True)` for all config/result objects
- **Type hints:** Required on public functions
- **Tests:** pytest. All tests must pass with mock executor (no API keys).
- **No scipy:** Statistics use hand-rolled implementations (Wilson CI, bootstrap, Cohen's d)

## Running Tests

```bash
make test       # Fast — all 125 tests, mock executor only
make smoke      # End-to-end run with quick_matrix
make lint       # ruff check + format check
```

## PR Process

1. Fork and create a feature branch
2. Make your changes
3. `make test && make lint` must pass
4. Submit a PR with a clear description
5. One approval required for merge

For scenarios: just the YAML + test. For code changes: tests required.

## Reporting Issues

Use the issue templates:
- **Bug report** — Something is broken
- **New scenario** — Propose a safety scenario
- **New executor** — Propose a backend integration
- **Feature request** — Everything else
