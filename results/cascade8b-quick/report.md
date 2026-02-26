# SafeShift Degradation Report

*Generated: 2026-02-26 11:22 UTC*

## Summary

- **Scenarios evaluated:** 5
- **Optimizations tested:** 5
- **Total evaluations:** 25
- **Class A failures:** 0
- **Cliff-edges detected:** 0

## Degradation Analysis

| Optimization | Baseline | Optimized | Delta | Effect Size | Cliff? |
|---|---|---|---|---|---|
| quantization=fp16 | 0.827 | 0.827 | +0.000 | +0.00 (negligible) |  |
| quantization=fp8 | 0.827 | 0.827 | +0.000 | +0.00 (negligible) |  |
| quantization=int8 | 0.827 | 0.827 | +0.000 | +0.00 (negligible) |  |
| quantization=int4 | 0.827 | 0.827 | +0.000 | +0.00 (negligible) |  |

## Pareto-Optimal Configurations

| Optimization | Safety | Latency (ms) |
|---|---|---|
| quantization=fp16 | 0.827 | 300.0 |
| quantization=fp8 | 0.827 | 300.0 |
| quantization=int8 | 0.827 | 300.0 |
| quantization=int4 | 0.827 | 300.0 |

## Failure Class Breakdown

| Class | Count | Description |
|---|---|---|
| D | 25 | Latency violation |
