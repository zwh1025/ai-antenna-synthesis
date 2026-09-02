# Stage 5A: Portable Sample-Efficiency and Convergence Evidence

## Scope

This PR adds a read-only analysis layer and a compact, machine-readable summary
of the frozen Stage 5A experiment. It does not retrain DeepSets, generate new
teacher labels, run an antenna simulation, or include checkpoints and NPZ
weights.

The experiment changes only the number of independent training parents:

| Independent parents | Correlated D4 rows | Validation parents | Test parents |
|---:|---:|---:|---:|
| 2 | 16 | 2 | 4 |
| 4 | 32 | 2 | 4 |
| 6 | 48 | 2 | 4 |
| 8 | 64 | 2 | 4 |

The subsets are nested. Each run uses seeds 0, 1, and 2. D4 rows are correlated
augmentation and are not additional independent teacher parents.

## Frozen task and evidence boundary

The study concerns one fixed reconstructed 32x32, 1024-element curved-array
task. Normalization and residual scale are computed from the corresponding
training subset only. Checkpoint and seed selection are validation-only; the four
independent test parents are not used for selection.

The local Stage 3C1/5A evaluator is the dense `eval_dense_3d` path on an 81x81
visible-uv domain. Upstream v3 acceptance uses a different legacy
`mylib.evaluation.evaluate_uv` definition and a different mixed planar/curved
dataset. Consequently, this evidence is **NOT_DIRECTLY_COMPARABLE_TO_UPSTREAM_V3**.

## Results

| Parents | Test improvement | Mean gain vs Taylor | Mean regret vs teacher | 90%-gain epoch |
|---:|---:|---:|---:|---:|
| 2 | 3/4 nominal | approximately 0 dB | 3.2719 dB | not reached |
| 4 | 4/4 | 0.8368 dB | 2.4351 dB | 70 |
| 6 | 3/4 | 1.3509 dB | 1.9210 dB | 55 |
| 8 | 4/4 | 1.3258 dB | 1.9461 dB | 55 |

The 2-parent change is numerical-level drift and is not a meaningful sample-
efficiency claim. The conservative scientific statement is:

> Nontrivial held-out improvement is demonstrated from four independent teacher
> parents; six to eight parents provide approximately 1.3 dB mean improvement
> on the fixed 1024-element curved-array task.

The strict all-validation-row teacher-proximity condition of 0.5 dB was not
reached. Best epoch, early stopping, 90%-gain epoch, and training runtime are
retained in the compact JSON summary.

## Complexity and cost

All sample-size models use the same architecture:

```text
Parameters = 597,250
MACs       = 405,733,376
FLOPs      = 811,466,752  (2 FLOPs per MAC convention)
```

These numbers describe the neural forward only. The compact summary keeps
offline teacher-generation and AI-training cost separate from online inference.
The reported CPU online latency is reused provenance from Stage 3C1 and is not
required to load a checkpoint in this PR:

```text
Inference-only mean = 9.157 ms
End-to-end mean     = 9.756 ms
```

## Reproduction entry

From the repository root:

```bash
python project/run_stage5a_sample_efficiency.py
python project/run_stage5a_sample_efficiency.py --json
pytest -q project/tests/test_stage5a_sample_efficiency.py
```

The reader validates the compact summary and prints the frozen table. It does
not reproduce the original training; that training requires the external
Stage 3C1 teacher labels, archive, checkpoints, and history tree. If the compact
summary is unavailable, the reader exits with an explicit error instead of
silently training or downloading artifacts.

## Relationship to upstream v3

This is complementary evidence, not a replacement for the upstream v3
mixed-geometry/NPU workflow. It does not claim arbitrary geometry, unseen-
geometry generalization, universal SOCP acceleration, or superiority over the
upstream v3 model.
