"""Stage 2 baseline artifact producer.

This runner deliberately contains no closure algorithm.  It freezes the
current Taylor/Bayliss/LCMV baseline with evaluator v1.0.0 before any future
algorithm change.  The output is written to ``results/stage2_strict_closure``
and the baseline directory is never overwritten.
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
    angular_distance_deg,
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
    STRICT_SUM_NULL_THRESHOLD_DB,
    SUM_SLL_THRESHOLD_DB,
    direction_to_uv,
    evaluate_official_case,
    pointing_threshold_deg,
)
from mylib.sum_diff import bayliss_excitation, capon_nulling_2d
from run_acceptance_v2 import get_null_dirs, get_scan_directions


NX = NY = 32
N_RANDOM = 200
RANDOM_SEED = 42
SLL_DESIGN = 35
RESULT_PARENT = (
    Path(__file__).resolve().parent.parent / "results" / "stage2_strict_closure"
)
BASELINE_DIR = RESULT_PARENT / "baseline"
STAGING_DIR = RESULT_PARENT / "baseline_staging"


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


def _git_state():
    root = Path(__file__).resolve().parent.parent
    def run(*args):
        return subprocess.check_output(
            args, cwd=root, text=True, stderr=subprocess.STDOUT).strip()
    status = run("git", "status", "--short")
    return {
        "branch": run("git", "branch", "--show-current"),
        "commit": run("git", "rev-parse", "HEAD"),
        "dirty": bool(status),
        "dirty_files": status.splitlines() if status else [],
    }


def _random_cases():
    """Reproduce run_random_validation's fixed target distribution exactly."""
    rng = np.random.RandomState(RANDOM_SEED)
    cases = []
    for index in range(N_RANDOM):
        theta0 = float(rng.uniform(0, 60))
        phi0 = float(rng.uniform(0, 360))
        nulls = []
        attempts = 0
        while len(nulls) < 4 and attempts < 100:
            tn = float(rng.uniform(10, 85))
            pn = float(rng.uniform(0, 360))
            if angular_distance_deg(tn, pn, theta0, phi0) >= 15.0:
                nulls.append((tn, pn))
            attempts += 1
        while len(nulls) < 4:
            nulls.append((85.0, float(rng.uniform(0, 360))))
        cases.append({
            "set": "random",
            "case_id": f"random_{index:03d}",
            "theta0_deg": theta0,
            "phi0_deg": phi0,
            "null_dirs": nulls,
        })
    return cases


def _regular_cases():
    cases = []
    for index, (theta0, phi0) in enumerate(get_scan_directions()):
        cases.append({
            "set": "regular",
            "case_id": f"regular_{index:03d}",
            "theta0_deg": float(theta0),
            "phi0_deg": float(phi0),
            "null_dirs": get_null_dirs(theta0, phi0),
        })
    return cases


def _null_metadata(case, main_lobe=None, kind="sum"):
    theta0, phi0 = case["theta0_deg"], case["phi0_deg"]
    details = []
    for theta_null, phi_null in case["null_dirs"]:
        un, vn = direction_to_uv(theta_null, phi_null)
        inside = None
        if main_lobe is not None and kind == "sum":
            peak_u, peak_v = main_lobe["peak_uv"]
            u_left, u_right = main_lobe["u_bounds_relative"]
            v_left, v_right = main_lobe["v_bounds_relative"]
            inside = (
                peak_u + u_left <= float(un) <= peak_u + u_right and
                peak_v + v_left <= float(vn) <= peak_v + v_right
            )
        elif main_lobe is not None and kind == "difference":
            target_u, target_v = main_lobe["target_uv"]
            axis = np.deg2rad(main_lobe["difference_axis_phi_deg"])
            du, dv = np.cos(axis), np.sin(axis)
            tu, tv = -dv, du
            q = (float(un) - target_u) * du + (float(vn) - target_v) * dv
            r = (float(un) - target_u) * tu + (float(vn) - target_v) * tv
            neg = (main_lobe["negative_lobe_q_bounds"][0] <= q <= 0.0 and
                   main_lobe["negative_lobe_r_bounds"][0] <= r <=
                   main_lobe["negative_lobe_r_bounds"][1])
            pos = (0.0 <= q <= main_lobe["positive_lobe_q_bounds"][1] and
                   main_lobe["positive_lobe_r_bounds"][0] <= r <=
                   main_lobe["positive_lobe_r_bounds"][1])
            inside = neg or pos
        details.append({
            "theta_deg": float(theta_null),
            "phi_deg": float(phi_null) % 360.0,
            "distance_from_target_deg": float(
                angular_distance_deg(theta_null, phi_null, theta0, phi0)),
            "outside_main_lobe": (None if inside is None else not inside),
            "outside_generation_guard_15deg": bool(
                angular_distance_deg(theta_null, phi_null, theta0, phi0) >= 15.0),
        })
    return details


def _sum_row(case, method, official):
    result = official["sum"]
    return {
        "set": case["set"],
        "case_id": case["case_id"],
        "method": method,
        "theta0_deg": case["theta0_deg"],
        "phi0_deg": case["phi0_deg"],
        "metric_version": official["metric_version"],
        "sll_db": result["sll_db"],
        "sll_threshold_db": SUM_SLL_THRESHOLD_DB,
        "sll_pass": bool(result["sll_db"] <= SUM_SLL_THRESHOLD_DB),
        "beamwidth_3db_deg": result["beamwidth_3db_deg"],
        "pointing_error_deg": result["pointing_error_deg"],
        "pointing_threshold_deg": result["pointing_threshold_deg"],
        "peak_direction": result["peak_direction"],
        "diagnostic": result["diagnostic"],
        "main_lobe": result["main_lobe"],
    }


def _difference_row(case, official):
    result = official["difference"]
    intrinsic = official["adaptive_null"]["difference"]
    center_db = intrinsic["center_db"][0]
    sum_bw = official["sum"]["beamwidth_3db_deg"]
    threshold = pointing_threshold_deg(sum_bw)
    return {
        "set": case["set"],
        "case_id": case["case_id"],
        "theta0_deg": case["theta0_deg"],
        "phi0_deg": case["phi0_deg"],
        "difference_axis_phi_deg": result["difference_axis_phi_deg"],
        "metric_version": official["metric_version"],
        "sll_db": result["sll_db"],
        "sll_threshold_db": DIFFERENCE_SLL_THRESHOLD_DB,
        "sll_pass": bool(result["sll_db"] <= DIFFERENCE_SLL_THRESHOLD_DB),
        "intrinsic_null_center_db": center_db,
        "intrinsic_null_threshold_db": ADAPTIVE_NULL_THRESHOLD_DB,
        "intrinsic_null_pass": bool(center_db <= ADAPTIVE_NULL_THRESHOLD_DB),
        "pointing_error_deg": result["pointing_error_deg"],
        "beamwidth_3db_sum_deg": sum_bw,
        "pointing_threshold_deg": threshold,
        "pointing_pass": bool(result["pointing_error_deg"] <= threshold),
        "zero_crossing_direction": result["zero_crossing_direction"],
        "main_lobe": result["main_lobe"],
    }


def _adaptive_row(case, official):
    nulls = official["adaptive_null"]["sum"]
    centers = nulls["center_db"]
    windows = nulls["window_worst_db"]
    return {
        "set": case["set"],
        "case_id": case["case_id"],
        "theta0_deg": case["theta0_deg"],
        "phi0_deg": case["phi0_deg"],
        "metric_version": official["metric_version"],
        "null_count": len(centers),
        "nulls": _null_metadata(case, official["sum"]["main_lobe"], kind="sum"),
        "sum_sll_db": official["sum"]["sll_db"],
        "sum_sll_pass": bool(official["sum"]["sll_db"] <= SUM_SLL_THRESHOLD_DB),
        "center_db": centers,
        "window_worst_db": windows,
        "all_center_pass_minus30": bool(
            len(centers) >= 4 and all(v <= ADAPTIVE_NULL_THRESHOLD_DB for v in centers)),
        "all_center_pass_strict_minus65": bool(
            len(centers) >= 4 and all(v <= STRICT_SUM_NULL_THRESHOLD_DB for v in centers)),
        "joint_pass_minus30": bool(
            official["sum"]["sll_db"] <= SUM_SLL_THRESHOLD_DB and
            len(centers) >= 4 and all(v <= ADAPTIVE_NULL_THRESHOLD_DB for v in centers)),
        "joint_pass_strict_minus65": bool(
            official["sum"]["sll_db"] <= SUM_SLL_THRESHOLD_DB and
            len(centers) >= 4 and all(v <= STRICT_SUM_NULL_THRESHOLD_DB for v in centers)),
        "window_role": nulls["window_role"],
    }


def _not_implemented_difference_adaptive_row(case, official_difference):
    return {
        "set": case["set"],
        "case_id": case["case_id"],
        "theta0_deg": case["theta0_deg"],
        "phi0_deg": case["phi0_deg"],
        "metric_version": OFFICIAL_EVALUATOR_VERSION,
        "status": "BASELINE_NOT_IMPLEMENTED",
        "null_count": 4,
        "nulls": _null_metadata(case, official_difference["main_lobe"], kind="difference"),
        "center_db": None,
        "window_worst_db": None,
        "threshold_minus30_db": ADAPTIVE_NULL_THRESHOLD_DB,
        "threshold_minus50_db": STRICT_DIFFERENCE_NULL_THRESHOLD_DB,
    }


def _aggregate(rows, value_key, pass_key):
    values = np.asarray([r[value_key] for r in rows if r.get(value_key) is not None], dtype=float)
    if values.size == 0:
        return {
            "cases": len(rows), "pass": 0, "fail": 0, "not_tested": len(rows),
            "worst": None, "best": None, "mean": None, "median": None,
            "p5": None, "p95": None,
        }
    passes = np.asarray([bool(r[pass_key]) for r in rows if r.get(value_key) is not None])
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


def _null_aggregate(rows, strict=False):
    key = "all_center_pass_strict_minus65" if strict else "all_center_pass_minus30"
    return {
        "cases": len(rows),
        "pass": int(sum(bool(row[key]) for row in rows)),
        "fail": int(sum(not bool(row[key]) for row in rows)),
        "not_tested": 0,
        "worst": float(max(max(row["center_db"]) for row in rows)),
        "best": float(min(min(row["center_db"]) for row in rows)),
        "requirement": STRICT_SUM_NULL_THRESHOLD_DB if strict else ADAPTIVE_NULL_THRESHOLD_DB,
    }


def _top_worst(rows, value_key, count=10):
    return sorted(rows, key=lambda row: row[value_key], reverse=True)[:count]


def main():
    if BASELINE_DIR.exists() or STAGING_DIR.exists():
        raise SystemExit(
            f"Refusing to overwrite existing baseline or staging: {BASELINE_DIR}"
        )
    RESULT_PARENT.mkdir(parents=True, exist_ok=True)
    STAGING_DIR.mkdir()

    git = _git_state()
    regular = _regular_cases()
    random = _random_cases()
    all_cases = regular + random
    posx = uniform_linear_array_pos(NX)
    posy = uniform_linear_array_pos(NY)
    amp_x_sum, amp_y_sum = taylor_2d_separable(NX, NY, SLL_DESIGN)
    amp_x_diff, _ = bayliss_excitation(NX, SLL_DESIGN)
    amp_y_diff = taylor_excitation(NY * 0.5, posy, SLL_DESIGN)

    metadata = {
        "stage": "STAGE_2_BASELINE",
        "status": "BASELINE_FORMAL_EVALUATION",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git": git,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "array": {
            "nx": NX, "ny": NY, "elements": NX * NY,
            "spacing_wavelengths": 0.5, "wavelength": 1.0,
            "coordinate_definition": "uniform_linear_array_pos in x/y, centered",
        },
        "design": {"sum": "Taylor", "difference": "Bayliss x Taylor", "sll_design_db": SLL_DESIGN},
        "evaluator": {
            "metric_version": OFFICIAL_EVALUATOR_VERSION,
            "grid_size": DEFAULT_GRID_SIZE,
            "sum_sll": "official first-null uv envelope",
            "difference_sll": "official two-lobe first-null envelopes",
            "null_center": "continuous target response",
            "null_window": "3 degree spherical cap, diagnostic only",
        },
        "scan_sets": {
            "regular": {
                "cases": len(regular),
                "definition": "theta=0 once; theta=10..60 by 10 and phi=0..330 by 30",
                "theta_range_deg": [0, 60], "phi_range_deg": [0, 330],
            },
            "random": {
                "cases": len(random), "seed": RANDOM_SEED,
                "distribution": "theta uniform [0,60), phi uniform [0,360)",
                "duplicate_check": "target pairs checked in artifact",
            },
        },
        "methods": {
            "sum_taylor": "current taylor_2d_separable + steering phase",
            "sum_lcmv": "current capon_nulling_2d, regular set only for adaptive null",
            "difference": "current bayliss_2d_separable equivalent: Bayliss x Taylor y",
            "difference_adaptive": "BASELINE_NOT_IMPLEMENTED",
        },
        "scope_note": "Formal baseline only; no closure fix, training, robustness, or large benchmark beyond this artifact.",
    }
    _write_json(STAGING_DIR / "metadata.json", metadata)

    sum_rows = []
    difference_rows = []
    adaptive_rows = []
    difference_adaptive_rows = []
    regular_taylor_amp, regular_taylor_phase = [], []
    regular_lcmv_amp, regular_lcmv_phase = [], []
    regular_diff_amp, regular_diff_phase = [], []
    random_taylor_amp, random_taylor_phase = [], []
    random_diff_amp, random_diff_phase = [], []

    t_start = time.perf_counter()
    for index, case in enumerate(all_cases):
        theta0, phi0 = case["theta0_deg"], case["phi0_deg"]
        px, py = beam_steering_phase_2d(posx, posy, theta0, phi0)
        sum_amp, sum_phase = combine_2d_excitation(amp_x_sum, amp_y_sum, px, py)
        diff_amp, diff_phase = combine_2d_excitation(amp_x_diff, amp_y_diff, px, py)

        # One official call captures Taylor sum + difference SLL + intrinsic null.
        r_taylor = evaluate_official_case(
            sum_amp, sum_phase, posx, posy, theta0, phi0,
            amp_difference=diff_amp, phase_difference=diff_phase,
            difference_null_dirs=[(theta0, phi0)],
            difference_axis_phi_deg=0.0,
            n_uv=DEFAULT_GRID_SIZE,
        )
        sum_rows.append(_sum_row(case, "taylor", r_taylor))
        difference_rows.append(_difference_row(case, r_taylor))

        if case["set"] == "regular":
            regular_taylor_amp.append(sum_amp)
            regular_taylor_phase.append(sum_phase)
            regular_diff_amp.append(diff_amp)
            regular_diff_phase.append(diff_phase)

            # Baseline adaptive-null uses the current sum LCMV implementation.
            lcmv_amp, lcmv_phase = capon_nulling_2d(
                posx, posy, sum_amp, sum_phase, theta0, phi0, case["null_dirs"])
            r_lcmv = evaluate_official_case(
                lcmv_amp, lcmv_phase, posx, posy, theta0, phi0,
                null_dirs=case["null_dirs"], n_uv=DEFAULT_GRID_SIZE)
            sum_rows.append(_sum_row(case, "lcmv", r_lcmv))
            adaptive_rows.append(_adaptive_row(case, r_lcmv))
            difference_adaptive_rows.append(
                _not_implemented_difference_adaptive_row(case, r_taylor["difference"])
            )
            regular_lcmv_amp.append(lcmv_amp)
            regular_lcmv_phase.append(lcmv_phase)
        else:
            random_taylor_amp.append(sum_amp)
            random_taylor_phase.append(sum_phase)
            random_diff_amp.append(diff_amp)
            random_diff_phase.append(diff_phase)

        if (index + 1) % 10 == 0 or index + 1 == len(all_cases):
            elapsed = time.perf_counter() - t_start
            print(f"baseline {index + 1}/{len(all_cases)} cases, elapsed={elapsed:.1f}s")

    sum_taylor_rows = [row for row in sum_rows if row["method"] == "taylor"]
    sum_lcmv_rows = [row for row in sum_rows if row["method"] == "lcmv"]
    baseline_table = [
        {"metric": "Sum SLL (Taylor)", "aggregate": _aggregate(sum_taylor_rows, "sll_db", "sll_pass"), "requirement": "<= -35 dBc"},
        {"metric": "Sum SLL (LCMV, regular)", "aggregate": _aggregate(sum_lcmv_rows, "sll_db", "sll_pass"), "requirement": "<= -35 dBc"},
        {"metric": "Diff SLL", "aggregate": _aggregate(difference_rows, "sll_db", "sll_pass"), "requirement": "<= -20 dBc"},
        {"metric": "Diff intrinsic null center", "aggregate": _aggregate(difference_rows, "intrinsic_null_center_db", "intrinsic_null_pass"), "requirement": "<= -30 dBc"},
        {"metric": "Diff pointing", "aggregate": _aggregate(difference_rows, "pointing_error_deg", "pointing_pass"), "requirement": "<= BW_sum/30"},
        {"metric": "Sum adaptive null center", "aggregate": _null_aggregate(adaptive_rows, strict=False), "requirement": "4 nulls, each <= -30 dBc and Sum SLL <= -35 for joint gate"},
        {"metric": "Sum strict null center", "aggregate": _null_aggregate(adaptive_rows, strict=True), "requirement": "4 nulls, each <= -65 dBc and Sum SLL <= -35 for joint gate"},
        {"metric": "Diff adaptive null", "aggregate": {"cases": len(difference_adaptive_rows), "pass": 0, "fail": 0, "not_tested": len(difference_adaptive_rows)}, "requirement": "4 nulls, each <= -30 dBc; BASELINE_NOT_IMPLEMENTED"},
        {"metric": "Diff strict null", "aggregate": {"cases": len(difference_adaptive_rows), "pass": 0, "fail": 0, "not_tested": len(difference_adaptive_rows)}, "requirement": "4 nulls, each <= -50 dBc; BASELINE_NOT_IMPLEMENTED"},
    ]
    summary = {
        "stage": "STAGE_2_BASELINE",
        "metric_version": OFFICIAL_EVALUATOR_VERSION,
        "baseline_table": baseline_table,
        "worst_10": {
            "sum_taylor_sll": _top_worst(sum_taylor_rows, "sll_db"),
            "sum_lcmv_sll": _top_worst(sum_lcmv_rows, "sll_db"),
            "difference_sll": _top_worst(difference_rows, "sll_db"),
            "difference_pointing": _top_worst(difference_rows, "pointing_error_deg"),
            "sum_adaptive_joint_minus30": [
                row for row in sorted(adaptive_rows, key=lambda r: not r["joint_pass_minus30"])[:10]
            ],
        },
        "counts": {
            "regular_cases": len(regular), "random_cases": len(random),
            "sum_taylor_rows": len(sum_taylor_rows), "sum_lcmv_rows": len(sum_lcmv_rows),
            "difference_rows": len(difference_rows), "sum_adaptive_rows": len(adaptive_rows),
            "difference_adaptive_rows": len(difference_adaptive_rows),
        },
        "diagnostic_only": True,
        "formal_headline_status": "not yet a final competition headline; baseline artifact for Stage 2 comparison",
    }

    _write_json(STAGING_DIR / "sum_cases.json", sum_rows)
    _write_json(STAGING_DIR / "difference_cases.json", difference_rows)
    _write_json(STAGING_DIR / "adaptive_null_cases.json", adaptive_rows)
    _write_json(STAGING_DIR / "difference_adaptive_null_cases.json", difference_adaptive_rows)
    _write_json(STAGING_DIR / "summary.json", summary)
    _write_json(STAGING_DIR / "stage2_baseline_summary.json", summary)

    weights_dir = STAGING_DIR / "weights"
    weights_dir.mkdir()
    np.savez_compressed(
        weights_dir / "regular_weights.npz",
        taylor_amp=np.asarray(regular_taylor_amp),
        taylor_phase=np.asarray(regular_taylor_phase),
        lcmv_amp=np.asarray(regular_lcmv_amp),
        lcmv_phase=np.asarray(regular_lcmv_phase),
        difference_amp=np.asarray(regular_diff_amp),
        difference_phase=np.asarray(regular_diff_phase),
        theta0_deg=np.asarray([c["theta0_deg"] for c in regular]),
        phi0_deg=np.asarray([c["phi0_deg"] for c in regular]),
    )
    np.savez_compressed(
        weights_dir / "random_weights.npz",
        taylor_amp=np.asarray(random_taylor_amp),
        taylor_phase=np.asarray(random_taylor_phase),
        difference_amp=np.asarray(random_diff_amp),
        difference_phase=np.asarray(random_diff_phase),
        theta0_deg=np.asarray([c["theta0_deg"] for c in random]),
        phi0_deg=np.asarray([c["phi0_deg"] for c in random]),
    )

    # Add a compact case manifest after all arrays have been written.
    _write_json(STAGING_DIR / "case_manifest.json", {
        "regular": regular,
        "random": random,
        "random_target_duplicates": len(random) - len({(c["theta0_deg"], c["phi0_deg"]) for c in random}),
    })
    os.rename(STAGING_DIR, BASELINE_DIR)
    print(f"baseline artifact written: {BASELINE_DIR}")


if __name__ == "__main__":
    main()
