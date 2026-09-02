"""Read-only analysis for the frozen Stage 5A sample-efficiency evidence.

This module deliberately does not train a model, solve an optimization problem,
load a checkpoint, or run an antenna simulation.  It reads the compact evidence
summary committed with this PR and prints the frozen parent-count study.  The
original training pipeline and its large artifacts remain outside this PR.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable


PORTABLE_ANALYSIS_ONLY = True
DEFAULT_SUMMARY = Path(__file__).resolve().parent / "outputs" / "stage5a_sample_efficiency_summary.json"
REQUIRED_COUNTS = (2, 4, 6, 8)
TEACHER_PROXIMITY_THRESHOLD_DB = 0.5
GAIN_FRACTION = 0.9


def _walk_numbers(value: Any) -> Iterable[float]:
    if isinstance(value, dict):
        for child in value.values():
            yield from _walk_numbers(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_numbers(child)
    elif isinstance(value, bool) or value is None or isinstance(value, str):
        return
    elif isinstance(value, (int, float)):
        yield float(value)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def first_epoch_within_teacher_gap(
    history: list[dict[str, Any]],
    threshold_db: float = TEACHER_PROXIMITY_THRESHOLD_DB,
    require_all: bool = True,
) -> int | str:
    """Return the first evaluated epoch within the fixed teacher-gap threshold."""

    for item in history:
        if not item.get("validation_sll_evaluated", False):
            continue
        gaps = [float(value) for value in item.get("validation_teacher_gap_db_by_row", {}).values()]
        if not gaps:
            continue
        within = [abs(value) <= threshold_db for value in gaps]
        if all(within) if require_all else any(within):
            return int(item["epoch"])
    return "NOT_REACHED"


def first_epoch_to_gain(
    history: list[dict[str, Any]],
    final_gain_db: float,
    fraction: float = GAIN_FRACTION,
) -> int | str:
    """Return the first evaluated epoch reaching ``fraction`` of final gain."""

    target = fraction * float(final_gain_db)
    if target <= 0:
        return "NOT_REACHED"
    for item in history:
        if not item.get("validation_sll_evaluated", False):
            continue
        gain = item.get("validation_gain_vs_taylor_db")
        if gain is not None and float(gain) >= target:
            return int(item["epoch"])
    return "NOT_REACHED"


def validate_summary(payload: dict[str, Any]) -> None:
    """Validate the portable schema and the frozen sample-efficiency contract."""

    _require(payload.get("schema_version") == "stage5a-portable-evidence-v1", "unsupported summary schema")
    _require(payload.get("stage5a_version") == "1.0.0", "unexpected Stage 5A version")
    _require(payload.get("directly_comparable_to_upstream_v3") is False, "upstream comparison boundary missing")
    study = payload.get("study", {})
    _require(study.get("train_parent_counts") == list(REQUIRED_COUNTS), "parent-count sweep is not 2/4/6/8")
    _require(study.get("validation_independent_parents") == 2, "validation parent count changed")
    _require(study.get("test_independent_parents") == 4, "test parent count changed")
    _require(study.get("seeds") == [0, 1, 2], "seed set changed")
    _require(study.get("d4_rows_are_correlated") is True, "D4 correlation boundary missing")
    _require(study.get("selection") == "validation_only", "selection boundary missing")

    levels = payload.get("levels", {})
    _require(set(levels) == {str(count) for count in REQUIRED_COUNTS}, "incomplete parent-count levels")
    for count in REQUIRED_COUNTS:
        item = levels[str(count)]
        _require(item.get("independent_train_parents") == count, f"invalid level {count}")
        _require(item.get("correlated_d4_rows") == 8 * count, f"invalid D4 row count for {count}")
        _require(item.get("test_used_for_selection") is False, f"test leakage at level {count}")
        _require("test" in item and "convergence" in item, f"missing results at level {count}")

    complexity = payload.get("complexity", {})
    _require(complexity.get("N") == 1024, "complexity N is not 1024")
    _require(complexity.get("parameter_count") == 597250, "parameter provenance mismatch")
    _require(complexity.get("MACs_per_sample") == 405733376, "MAC provenance mismatch")
    _require(complexity.get("FLOPs_at_2_per_MAC") == 811466752, "FLOP provenance mismatch")

    for number in _walk_numbers(payload):
        _require(math.isfinite(number), "non-finite number in portable summary")


def load_summary(path: Path = DEFAULT_SUMMARY) -> dict[str, Any]:
    """Load and validate a compact summary; no external artifact is read."""

    if not path.exists():
        raise FileNotFoundError(
            f"Stage 5A compact summary not found: {path}. "
            "The full training/evidence artifacts are intentionally external to this PR."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Stage 5A summary must be a JSON object")
    validate_summary(payload)
    return payload


def format_report(payload: dict[str, Any]) -> str:
    rows = [
        "Stage 5A portable evidence (independent teacher parents)",
        "N | D4 rows | gain dB | improves | mean regret dB | best epoch | 90%-gain epoch | train mean s",
        "--|---------:|---------:|---------|----------------:|-----------:|----------------:|------------:",
    ]
    for count in REQUIRED_COUNTS:
        item = payload["levels"][str(count)]
        test = item["test"]
        convergence = item["convergence"]
        runtime = item["training_runtime_s"]
        rows.append(
            f"{count} | {item['correlated_d4_rows']} | {test['mean_gain_db']:.6f} | "
            f"{test['improves_count']}/4 | {test['mean_regret_vs_teacher_db']:.6f} | "
            f"{convergence['best_epoch']} | {convergence['epoch_to_90pct_final_gain']} | "
            f"{runtime['mean']:.3f}"
        )
    rows.extend(
        [
            "",
            "Claim boundary: one fixed reconstructed 1024-element curved-array task; "
            "D4 rows are correlated augmentation.",
            "Teacher proximity: strict all-validation-row 0.5 dB condition was not reached.",
        ]
    )
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    assert PORTABLE_ANALYSIS_ONLY
    parser = argparse.ArgumentParser(
        description="Read and summarize frozen Stage 5A evidence; does not train or simulate."
    )
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY, help="compact JSON summary path")
    parser.add_argument("--json", action="store_true", help="print the validated JSON summary")
    args = parser.parse_args(argv)
    try:
        payload = load_summary(args.summary)
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(format_report(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
