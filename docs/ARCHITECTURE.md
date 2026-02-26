# SafeShift Architecture

## Overview

SafeShift benchmarks whether inference optimizations (quantization, batching,
speculative decoding, attention variants) degrade safety properties of LLM
responses to safety-critical scenarios.

## Data Flow

```
Scenario YAML ─→ MatrixConfig ─→ Executor ─→ ExecutorResult
                                    │              │
                                    ↓              ↓
                              (remote GPU)    Grader (local)
                                                   │
                                                   ↓
                                              GradeResult
                                                   │
                                    ┌──────────────┼──────────────┐
                                    ↓              ↓              ↓
                                 Pareto       Degradation     Regression
                                 Curves       Detection        Testing
                                    │              │              │
                                    ↓              ↓              ↓
                                 Plots        Report.md       Exit Code
```

## Core Abstractions

### Scenario
A safety-critical decision under time constraint. Includes:
- Input messages (system + user)
- Latency budget (target / acceptable / critical ms)
- Safety invariants (must-hold properties)
- Constraint spec (what makes it time-critical)
- Rubric reference

### Executor
Pluggable inference backend. Implementations:
- **MockExecutor** — Deterministic, simulates latency and degradation
- **APIExecutor** — Cloud APIs (OpenAI, Anthropic)
- **VLLMExecutor** — vLLM OpenAI-compatible API
- **RemoteExecutor** — SSH dispatch (stub in v0.1)

### Grader
Three-layer assessment:
- **Layer 0:** Deterministic regex patterns (urgency, escalation, minimization)
- **Layer 1:** Domain heuristics (reserved)
- **Layer 2:** Cross-vendor LLM judge with structured rubric

Five dimensions: Safety (0.40), Accuracy (0.25), Completeness (0.15),
Timeliness (0.10), Specificity (0.10).

### Analysis
- **Pareto frontiers:** Safety vs latency/throughput/memory
- **Cliff-edge detection:** Disproportionate safety drops
- **Degradation reports:** Per-optimization delta with Wilson CI
- **Regression testing:** CI-friendly exit codes

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| Grading is always local | Safety assessment must not depend on GPU infra |
| Cross-vendor LLM judge | Never self-judge (safety rule) |
| Frozen dataclasses | Immutable configs prevent mutation bugs |
| Wilson CI without scipy | Minimal dependencies |
| YAML configs | Ecosystem convention |
| Domain-agnostic constraints | Works for clinical, robotic, financial |
