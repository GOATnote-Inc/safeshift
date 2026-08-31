# nemotron30b-full — null result (pipeline shakeout, not a finding)

This run does not measure the effect of quantization. Read `report.md` with
that in mind:

- **The independent variable was never varied.** The vLLM executor at the time
  of this run sent the identical request to a single fp16 server for every
  optimization label. All 23 scenarios have byte-identical `response_text`
  across baseline/fp16/fp8/int8/int4 in `results.jsonl`, so the `+0.000`
  deltas in `report.md` are true by construction, not evidence that
  quantization is safe.
- **Grading was pattern-only (layer 0).** No LLM judge ran; all 115 grades in
  `grades.jsonl` carry `layer: 0` on every dimension.

The current CLI refuses this configuration (`safeshift run` exits 2 when a
matrix requests optimizations the executor cannot apply). A valid quantization
sweep requires one vLLM endpoint per quantization level, run as separate
cells.

This directory is kept unchanged for provenance as a pipeline shakeout
artifact. It contains no findings about quantization and safety.
