"""Stage 2 closure artifact producer.

This runner is intentionally separate from the frozen baseline producer.  It
evaluates only the permitted post-baseline difference-beam closure fix on the
73 regular cases and writes a new atomic ``closure`` artifact beside the
immutable ``baseline`` artifact.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mylib.antenna_calc import (
    beam_steering_phase_2d,
    combine_2d_excitation,
    taylor_2d_separable,
    taylor_excitation,
    uniform_linear_array_pos,
)
from mylib.official_evaluator import (
    ADAPTIVE_NULL_THRESHOLD_DB,
    DEFAULT_GRID_SIZE,
    DIFFERENCE_SLL_THRESHOLD_DB,
    OFFICIAL_EVALUATOR_VERSION,
    STRICT_DIFFERENCE_NULL_THRESHOLD_DB,
    SUM_SLL_THRESHOLD_DB,
    evaluate_official_case,
)
from mylib.sum_diff import (
    bayliss_excitation,
    capon_nulling_difference_2d,
)
from run_stage2_strict_closure import (
    NX,
    NY,
    SLL_DESIGN,
    _git_state,
    _null_metadata,
    _regular_cases,
)


ROOT = Path(__file__).resolve().parent.parent
RESULT_PARENT = ROOT / "results" / "stage2_strict_closure"
BASELINE_DIR = RESULT_PARENT / "baseline"
CLOSURE_DIR = RESULT_PARENT / "closure"
STAGING_DIR = RESULT_PARENT / "closure_staging"


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot serialize {type(value)!r}")


def _write_json(path, payload):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False,
                  default=_json_default, allow_nan=False)


def _diff_row(case, official):
    difference = official["difference"]
    adaptive = official["adaptive_null"]["difference"]
    centers = adaptive["center_db"]
    windows = adaptive["window_worst_db"]
    return {
        "set": case["set"],
        "case_id": case["case_id"],
        "theta0_deg": case["theta0_deg"],
        "phi0_deg": case["phi0_deg"],
        "metric_version": official["metric_version"],
        "method": "bayliss_x_taylor_plus_difference_lcmv",
        "difference_axis_phi_deg": difference["difference_axis_phi_deg"],
        "difference_sll_db": difference["sll_db"],
        "difference_sll_threshold_db": DIFFERENCE_SLL_THRESHOLD_DB,
        "difference_sll_pass": bool(
            difference["sll_db"] <= DIFFERENCE_SLL_THRESHOLD_DB),
        "intrinsic_null_center_db": adaptive["center_db"][0],
        "intrinsic_null_pass": bool(
            adaptive["center_db"][0] <= ADAPTIVE_NULL_THRESHOLD_DB),
        "center_db": centers[1:],
        "window_worst_db": windows[1:],
        "all_center_pass_minus30": bool(
            len(centers) == 5 and
            all(value <= ADAPTIVE_NULL_THRESHOLD_DB for value in centers)),
        "all_center_pass_strict_minus50": bool(
            len(centers) == 5 and
            all(value <= STRICT_DIFFERENCE_NULL_THRESHOLD_DB
                for value in centers)),
        "null_count": len(centers) - 1,
        "nulls": _null_metadata(
            case, difference["main_lobe"], kind="difference"),
        "main_lobe": difference["main_lobe"],
        "window_role": adaptive["window_role"],
    }


def _aggregate(rows, value_key, pass_key):
    values = np.asarray([row[value_key] for row in rows], dtype=float)
    passes = np.asarray([bool(row[pass_key]) for row in rows])
    return {
        "cases": len(rows),
        "pass": int(np.sum(passes)),
        "fail": int(np.sum(~passes)),
        "not_tested": 0,
        "worst": float(np.max(values)),
        "best": float(np.min(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p5": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
    }


def _center_aggregate(rows, key, threshold):
    passes = [all(value <= threshold for value in row[key]) for row in rows]
    values = [value for row in rows for value in row[key]]
    return {
        "cases": len(rows),
        "nulls": len(values),
        "pass": int(sum(passes)),
        "fail": int(len(passes) - sum(passes)),
        "not_tested": 0,
        "worst": float(max(values)),
        "best": float(min(values)),
        "threshold_db": threshold,
    }


def main():
    if not BASELINE_DIR.exists():
        raise SystemExit(f"Missing frozen baseline: {BASELINE_DIR}")
    if CLOSURE_DIR.exists() or STAGING_DIR.exists():
        raise SystemExit(
            f"Refusing to overwrite existing closure or staging: {CLOSURE_DIR}")

    RESULT_PARENT.mkdir(parents=True, exist_ok=True)
    STAGING_DIR.mkdir()
    regular = _regular_cases()
    posx = uniform_linear_array_pos(NX)
    posy = uniform_linear_array_pos(NY)
    amp_x_sum, amp_y_sum = taylor_2d_separable(NX, NY, SLL_DESIGN)
    amp_x_diff, _ = bayliss_excitation(NX, SLL_DESIGN)
    amp_y_diff = taylor_excitation(NY * 0.5, posy, SLL_DESIGN)

    metadata = {
        "stage": "STAGE_2_CLOSURE",
        "status": "POST_BASELINE_MINIMAL_CLOSURE",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git": _git_state(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "baseline_artifact": str(BASELINE_DIR),
        "baseline_status": "Diff adaptive = BASELINE_NOT_IMPLEMENTED",
        "array": {"nx": NX, "ny": NY, "elements": NX * NY,
                  "spacing_wavelengths": 0.5, "wavelength": 1.0},
        "evaluator": {"metric_version": OFFICIAL_EVALUATOR_VERSION,
                      "grid_size": DEFAULT_GRID_SIZE,
                      "null_window": "3 degree spherical cap, diagnostic only"},
        "scan_set": {"cases": len(regular),
                      "definition": "theta=0 once; theta=10..60 by 10 and phi=0..330 by 30"},
        "closure": {
            "function": "mylib.sum_diff.capon_nulling_difference_2d",
            "objective": "minimum-norm correction around Bayliss x Taylor reference",
            "constraints": "zero response at intrinsic target plus four requested null directions",
            "normalization": "common positive amplitude scale only; no target-response normalization",
        },
        "scope_note": "Only the permitted post-baseline difference adaptive closure was evaluated.",
    }
    _write_json(STAGING_DIR / "metadata.json", metadata)

    rows = []
    diff_amp_rows, diff_phase_rows = [], []
    t_start = time.perf_counter()
    for index, case in enumerate(regular):
        theta0, phi0 = case["theta0_deg"], case["phi0_deg"]
        phase_x, phase_y = beam_steering_phase_2d(posx, posy, theta0, phi0)
        sum_amp, sum_phase = combine_2d_excitation(
            amp_x_sum, amp_y_sum, phase_x, phase_y)
        diff_amp, diff_phase = combine_2d_excitation(
            amp_x_diff, amp_y_diff, phase_x, phase_y)
        adaptive_amp, adaptive_phase = capon_nulling_difference_2d(
            posx, posy, diff_amp, diff_phase, theta0, phi0,
            case["null_dirs"])
        official = evaluate_official_case(
            sum_amp, sum_phase, posx, posy, theta0, phi0,
            amp_difference=adaptive_amp, phase_difference=adaptive_phase,
            difference_null_dirs=[(theta0, phi0)] + case["null_dirs"],
            difference_axis_phi_deg=0.0,
            n_uv=DEFAULT_GRID_SIZE,
        )
        rows.append(_diff_row(case, official))
        diff_amp_rows.append(adaptive_amp)
        diff_phase_rows.append(adaptive_phase)
        if (index + 1) % 10 == 0 or index + 1 == len(regular):
            elapsed = time.perf_counter() - t_start
            print(f"closure {index + 1}/{len(regular)} cases, elapsed={elapsed:.1f}s")

    summary = {
        "stage": "STAGE_2_CLOSURE",
        "metric_version": OFFICIAL_EVALUATOR_VERSION,
        "baseline_comparison": {
            "difference_adaptive_null": {
                "before": "BASELINE_NOT_IMPLEMENTED",
                "before_pass": 0,
                "before_fail": 0,
                "before_not_tested": len(rows),
            },
            "after_method": "capon_nulling_difference_2d",
        },
        "closure_table": [
            {"metric": "Diff adaptive null center <= -30 dBc",
             "aggregate": _center_aggregate(
                 rows, "center_db", ADAPTIVE_NULL_THRESHOLD_DB),
             "requirement": "4 nulls in each of 73 regular cases"},
            {"metric": "Diff strict null center <= -50 dBc",
             "aggregate": _center_aggregate(
                 rows, "center_db", STRICT_DIFFERENCE_NULL_THRESHOLD_DB),
             "requirement": "4 nulls in each of 73 regular cases"},
            {"metric": "Diff SLL after closure",
             "aggregate": _aggregate(
                 rows, "difference_sll_db", "difference_sll_pass"),
             "requirement": "<= -20 dBc"},
        ],
        "joint_pass": {
            "minus30": bool(all(
                row["difference_sll_pass"] and
                row["all_center_pass_minus30"] for row in rows)),
            "strict_minus50": bool(all(
                row["difference_sll_pass"] and
                row["all_center_pass_strict_minus50"] for row in rows)),
        },
        "worst_10": {
            "difference_sll": sorted(
                rows, key=lambda row: row["difference_sll_db"], reverse=True)[:10],
            "difference_adaptive_center": sorted(
                rows, key=lambda row: max(row["center_db"]), reverse=True)[:10],
        },
        "counts": {"regular_cases": len(regular),
                   "difference_adaptive_rows": len(rows)},
        "diagnostic_only": True,
        "formal_headline_status": "closure comparison artifact; not a competition headline",
    }
    _write_json(STAGING_DIR / "difference_adaptive_null_cases.json", rows)
    _write_json(STAGING_DIR / "summary.json", summary)
    _write_json(STAGING_DIR / "stage2_closure_summary.json", summary)

    weights_dir = STAGING_DIR / "weights"
    weights_dir.mkdir()
    np.savez_compressed(
        weights_dir / "regular_difference_adaptive_weights.npz",
        amp=np.asarray(diff_amp_rows), phase=np.asarray(diff_phase_rows),
        theta0_deg=np.asarray([case["theta0_deg"] for case in regular]),
        phi0_deg=np.asarray([case["phi0_deg"] for case in regular]),
    )
    os.rename(STAGING_DIR, CLOSURE_DIR)
    print(f"closure artifact written: {CLOSURE_DIR}")


if __name__ == "__main__":
    main()
