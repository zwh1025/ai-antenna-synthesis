# Stage 4A — Robustness Benchmark Freeze & Degradation Audit

## 1. Scope and gate
This document records a Track P physics-only degradation audit. It is not robust re-synthesis and does not change any Stage 2/Stage 3C1 artifact.
Gate: `STAGE_4A_GO`. Robustness model version: `1.0.0`; metric version: `1.0.0`.

## 2. Frozen baseline
The baseline is the Stage 2 strict-closure 32×32 planar artifact at `results\stage2_strict_closure\baseline` with SHA-256 `38b8dcc7349ddbc7ada65437072426bfd9710d08da1c36382dd8e1f2ff262c7f`.
Taylor/LCMV sum weights and Bayliss×Taylor difference weights are reused exactly in every perturbation.

## 3. Official evaluator
All formal calls use `mylib.official_evaluator.evaluate_official_case`, evaluator version `1.0.0`, uv grid `201×201`.
The output uses the visible-domain maximum field normalization and retains the evaluator's stable -300 dBc floor.

## 4. Representative cases
Sixteen cases were frozen before any perturbed official evaluation: eight regular scan cases and eight random cases from the Stage 2 manifest.
Case manifest SHA-256: `ea066bb3f120933817d28b70ca9a61c9726c27ce5f49d115ca3424a6396f7a6f`.

## 5. Position-error model
The formal assumption is model B: independent planar Δx and Δy, each Uniform[-0.05,+0.05]λ; Δz=0. This is explicitly recorded as an assumption because the wording permits alternative A/B/C readings.

## 6. Amplitude and phase quantization
Amplitude uses the nearest 0.5 dB grid relative to unit maximum. Phase uses the nearest 6-bit state (5.625°), wrapped modulo 2π. This deterministic quantizer is frozen separately from stochastic failure tests.

## 7. Element failures
Failures set excitation amplitude to exact zero and use floor(1024×rate): 51, 102, and 204 failed elements at 5%, 10%, and 20%. One mask is applied to both sum and difference excitations.
All 16 cases × 3 levels × 20 seeds are present in `failure_cases.json`; failed indices and mask hashes are saved for Stage 4B.

## 8. Frequency offset
The scan is ratio 0.90 through 1.10 in 0.01 increments. Physical coordinates remain fixed and the evaluator uses λ=1/ratio; weights are not retuned.

## 9. Degradation definition
Every perturbed row is compared with the same case's ideal reference. Positive SLL, pointing, or null deltas indicate a worse measured value. Per-realization records retain mean, median, standard deviation, P5, P95, worst, and compliance rates in their aggregates.

## 10. Joint compliance
The comparable `common_joint` field combines sum SLL, difference SLL, difference pointing, and the intrinsic difference null. `track_p_joint` additionally requires measured sum null compliance and is reported only for regular LCMV cases. Difference adaptive-null compliance is unavailable in the frozen Stage 2 baseline and is never imputed.

## 11. Ideal reference
Ideal common-joint pass rate: 1.0; regular adaptive-null joint availability is explicit in the artifact.

## 12. Position results
Across 320 position-error realizations, mean Δsum-SLL is 2.186146 dB and worst Δsum-SLL is 5.398300 dB.

## 13. Quantization results
Across 16 deterministic quantized cases, mean Δsum-SLL is 0.860208 dB and worst Δsum-SLL is 3.700556 dB.

## 14. Failure results
The machine-readable summary reports each rate separately, including common-joint and regular Track P joint pass rates. Interpretation must use both degradation distributions and pass-rate changes; no threshold is changed by this audit.

## 15. Frequency results
Each case has 21 frequency rows, a common-joint compliance vector, and the contiguous compliant band containing ratio 1.0 when it exists.

## 16. Runtime protocol
Physics synthesis and AI inference were measured independently with 10 warmups and 100 timed repetitions. Teacher generation and dense official evaluation are excluded from the runtime samples.

## 17. Runtime stability
Physics N=100; AI inference N=200; AI end-to-end N=200. Mean/std/P50/P95/min/max/CV are in `runtime_stability.json`.

## 18. Legacy implementation audit
`legacy_robustness_audit.json` inventories the old planar, failure, and curved-AI scripts. Their values are not reused because they use legacy evaluators, different case protocols, or forbidden re-optimization.

## 19. AI scope boundary
The Stage 3C1 AI robustness diagnostic is not run in the formal Stage 4A gate. The formal result is physics-only; this prevents curved-geometry AI diagnostics from being mixed with planar Track P evidence.

## 20. Stage 4B handoff and limitations
The data-driven first candidate failure level is `0.20` by mean sum-SLL degradation. Stage 4B may reuse the saved masks exactly, but no Stage 4B action was taken here. No commit or push was performed.
