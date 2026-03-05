# AAR: SafeShift Grading Pipeline Development

## Campaign ID
safeshift-grading-pipeline

## Date Range
2026-02-20 to 2026-02-27

## Objective
Design and implement a multi-dimensional grading pipeline for evaluating safety degradation under inference optimization techniques (quantization, batching, speculative decoding). The pipeline must distinguish genuine safety degradation from accuracy loss, with safety weighted at 40% of the composite score.

## Experimental Setup
- **Target models:** Frontier multimodal LLMs under simulated inference optimization
- **Scenarios:** 23 clinical (15 clinical, 8 robotic) across emergency, triage, and monitoring contexts
- **Grading dimensions:** Safety (0.40), Accuracy (0.25), Completeness (0.15), Timeliness (0.10), Specificity (0.10)
- **Judge protocol:** Cross-vendor (GPT-5.2 judges Anthropic models, Opus 4.6 judges OpenAI)
- **Determinism:** temperature=0.0, seed=42 for all evaluations

## Key Findings

1. **Safety dimension requires independent weighting.** Early prototypes used equal weighting across 5 dimensions. A scenario where a model correctly identified a diagnosis but recommended against emergency escalation scored 80% — unacceptable for a safety evaluation. The 0.40 safety weight ensures that safety failures dominate the composite score regardless of clinical accuracy.

2. **Cross-vendor judging is essential for credibility.** Initial development used same-vendor judges for convenience. When cross-vendor judging was implemented, 3 of 23 scenarios flipped pass/fail status, confirming that judge bias is a measurable confound in safety evaluations.

3. **OpenEM integration required careful boundary design.** SafeShift references OpenEM conditions for clinical grounding but must remain independently installable. The `safeshift[openem]` optional dependency pattern preserves this boundary while enabling condition-level analysis when OpenEM is available.

4. **Robotic scenarios expose a different failure mode.** The 8 robotic scenarios (surgical robot, medication dispensing, patient monitoring) revealed that models are more likely to maintain safety escalation in clinical dialogue than in systems-integration contexts. This suggests safety persistence is partially domain-dependent.

## Root Causes

**RC-1: Weighting convergence.** The final 5-dimension weighting (0.40/0.25/0.15/0.10/0.10) emerged from iterative calibration against known-bad scenarios. No single weighting scheme was correct from the start — it required testing against edge cases where safety and accuracy diverge.

**RC-2: Judge construct alignment.** Cross-vendor judges sometimes disagreed on whether a response constituted "safety maintenance" vs "over-escalation." This is the same construct-divergence problem documented in LostBench AAR-MSTS-CROSSVENDOR. The rubric was tightened to define safety in terms of specific action recommendations rather than tonal framing.

## Corrective Actions

| Action | Status | Owner |
|--------|--------|-------|
| Implement 5-dimension weighted grading | Complete | eval-lead |
| Add cross-vendor judge enforcement at CLI level | Complete | eval-lead |
| Create `safeshift[openem]` optional dependency | Complete | eval-lead |
| Add 8 robotic scenarios to complement 15 clinical | Complete | eval-lead |
| Document grading dimension rationale in README | Complete | eval-lead |

## Lessons Learned

1. **Safety weight must be explicit and justified.** Equal weighting across dimensions masks safety failures. The 0.40 weight is a design decision, not an empirical finding — it encodes the principle that safety failures are categorically worse than accuracy gaps.

2. **Cross-vendor judging should be default from day one.** Retrofitting cross-vendor judging after initial results are published creates a credibility gap. SafeShift implemented it from the grading pipeline's first release.

3. **Optional dependencies preserve installability.** The `[openem]` extra pattern allows SafeShift to benefit from OpenEM's clinical knowledge without creating a hard dependency that would complicate installation for external users (SafeShift is Apache 2.0 public).

4. **Domain-specific scenario types reveal distinct failure modes.** Clinical dialogue and systems-integration contexts exercise different safety muscles. Future scenario expansion should maintain balance across both domains.
