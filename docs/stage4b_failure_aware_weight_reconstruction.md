# Stage 4B — Failure-Aware Weight Reconstruction Closure

## 1. Scope
Known element failures to active-element weight reconstruction only. No AI, geometry, frequency, or calibration reconstruction is included.

## 2. Stage 4A baseline
The exact Stage 4A failure artifact is `results/stage4a_robustness_degradation/failure_cases.json` (SHA-256 `c3845aac56a7972c60d2590dfb5d2097f76c90370897302ec36a044c88f50ece`); 16 targets × 20 seeds × 3 levels are consumed without regenerating masks.

## 3. Requirement mapping
The implementation addresses the requirement to correct excitation weights when the failed-element state is known.

## 4. Failure model
The 5%, 10%, and 20% levels use the Stage 4A failed indices and masks exactly: 51, 102, and 204 failed elements.

## 5. Reconstruction formulation
B2 solves a closed-form active-coordinate constrained least-squares correction. Sum preserves the ideal complex response at the target while fitting the ideal visible-domain field; the four frozen Sum nulls remain in that field objective and are measured after reconstruction. Difference constrains the intrinsic target null and four frozen nulls to zero.

## 6. Power normalization
Policy A: B1/B2 are scaled to the ideal beam's original l2 norm. This does not increase total excitation power; failed entries remain exact zero.

## 7. Pilot
The pilot was fixed to all 16 representative cases at 20% and seed 0 before formal evaluation. It is recorded separately and is not the gate.

## 8. 20% formal benchmark
All 320 masks were completed. B2 mean Sum-SLL recovery is 2.094337 dB; common-joint pass rate is 0.0.

## 9. 10% generalization
The frozen B2 algorithm is applied to all 320 10% masks without parameter changes.

## 10. 5% generalization
The frozen B2 algorithm is applied to all 320 5% masks without parameter changes.

## 11. Sum recovery
B0, B1, and B2 are compared per mask. Headline recovery is degraded Sum SLL minus reconstructed Sum SLL; positive is improvement.

## 12. Difference recovery
Azimuth Difference uses the exact Stage 4A Bayliss×Taylor reference and reports SLL, pointing, intrinsic null, and B1/B2 measured four-null diagnostics.

## 13. Main-beam preservation
Official pointing and beamwidth are retained. Because the evaluator is peak-normalized, no absolute gain-loss claim is made; l2 power and failed-zero validity are checked.

## 14. Adaptive-null status
Sum nulls are retained in the B2 field-fit objective and measured, but are not added as extra hard constraints in this Sum-first closure. Difference four-null measurements are run for B1/B2; Stage 4A B0 remains unavailable because its frozen baseline did not implement Difference adaptive nulls.

## 15. Runtime
Reconstruction and official pattern-evaluation runtimes are separated in `runtime.json`, with mean/median/P95/min/max statistics.

## 16. Failure cases
Every row stores case, target, rate, seed, exact failed indices, mask hash, ideal/degraded/reconstructed metrics, solver status, runtime, and recovery.

## 17. Recovery classification
Classification: `C — LIMITED RECOVERY`. It is based on the pooled frozen 960-mask benchmark, with 20% as the primary severity level; it is not based on the pilot.

## 18. Provenance
Metric version `1.0.0`, robustness version `1.0.0`, reconstruction version `1.0.0`; target manifest SHA-256 `4d813ee07fa7cc88ac1fae5647ab39ad050843ee5b3c3ecfac8e6859d8f283c4`.

## 19. Limitations
Elevation Difference is not mixed into this exact Stage 4A source comparison; no AI reconstruction, position/frequency compensation, arbitrary geometry, or universal recovery claim is made.

## 20. Gate
`STAGE_4B_CONDITIONAL`. The result is deliberately conditional if an axis or capability remains outside the frozen formal source.

