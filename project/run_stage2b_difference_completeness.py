"""Stage 2B formal Difference-axis completeness benchmark.

The case manifest is read from the frozen Stage 2 artifact.  No random case
is regenerated here.  Both Difference axes use the same Bayliss/LCMV code
path and the same official evaluator primitives.
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
    uniform_linear_array_pos,
)
from mylib.official_evaluator import (
    ADAPTIVE_NULL_THRESHOLD_DB,
    DEFAULT_GRID_SIZE,
    DIFFERENCE_SLL_THRESHOLD_DB,
    OFFICIAL_EVALUATOR_VERSION,
    STRICT_DIFFERENCE_NULL_THRESHOLD_DB,
    SUM_SLL_THRESHOLD_DB,
    evaluate_difference_beam,
    evaluate_nulls,
    evaluate_sum_beam,
    pointing_threshold_deg,
    pointing_rmse_deg,
)
from mylib.sum_diff import (
    bayliss_2d_separable,
    capon_nulling_difference_2d,
    difference_axis_to_phi_deg,
    difference_null_is_legal,
)


ROOT = Path(__file__).resolve().parent.parent
STAGE2_BASELINE = ROOT / "results" / "stage2_strict_closure" / "baseline"
RESULT_PARENT = ROOT / "results" / "stage2b_difference_completeness"
FINAL_DIR = RESULT_PARENT / "formal"
STAGING_DIR = RESULT_PARENT / "formal_staging"

NX = NY = 32
SLL_DESIGN = 35
AXES = ("azimuth", "elevation")
REPORTING_FLOOR_DB = -100.0


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot serialize {type(value)!r}")


def _write_json(path, payload):
    path.write_text(json.dumps(
        payload, indent=2, ensure_ascii=False,
        default=_json_default, allow_nan=False), encoding="utf-8")


def _git_state():
    def run(*args):
        return subprocess.check_output(
            args, cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()

    status = run("git", "status", "--short")
    return {
        "branch": run("git", "branch", "--show-current"),
        "commit": run("git", "rev-parse", "HEAD"),
        "dirty": bool(status),
        "dirty_files": status.splitlines() if status else [],
    }


def _load_manifest():
    manifest_path = STAGE2_BASELINE / "case_manifest.json"
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (len(source_manifest["regular"]) != 73 or
            len(source_manifest["random"]) != 200):
        raise RuntimeError("Stage 2 manifest does not contain 73 + 200 cases")
    if source_manifest.get("random_target_duplicates") != 0:
        raise RuntimeError("Stage 2 random manifest contains duplicate targets")
    # Keep the Stage 2 random target pairs exactly.  Its original random null
    # coordinates are retained below for provenance, but the 15-degree guard
    # alone allowed random_140 to enter an official Difference main lobe.
    # Use one fixed, result-independent rule for all random adaptive cases.
    effective_manifest = {
        "regular": source_manifest["regular"],
        "random": [],
        "random_target_duplicates": source_manifest["random_target_duplicates"],
        "source": str(manifest_path),
        "random_adaptive_null_rule": (
            "theta=90 deg; phi=target_phi+[45,135,225,315] deg modulo 360"
        ),
    }
    for source_case in source_manifest["random"]:
        case = dict(source_case)
        case["source_null_dirs"] = source_case["null_dirs"]
        phi0 = float(source_case["phi0_deg"])
        case["null_dirs"] = [
            [90.0, (phi0 + offset) % 360.0]
            for offset in (45.0, 135.0, 225.0, 315.0)
        ]
        effective_manifest["random"].append(case)
    cases = effective_manifest["regular"] + effective_manifest["random"]
    for case in cases:
        if len(case["null_dirs"]) != 4:
            raise RuntimeError(f"{case['case_id']} does not have four nulls")
    return source_manifest, effective_manifest, cases


def _display_db(value):
    if float(value) <= REPORTING_FLOOR_DB:
        return f"< {abs(REPORTING_FLOOR_DB):g} dBc"
    return f"{float(value):.6f} dBc"


def _legality(case, main_lobe):
    details = []
    seen = set()
    for theta_null, phi_null in case["null_dirs"]:
        key = (float(theta_null), float(phi_null) % 360.0)
        duplicate = key in seen
        seen.add(key)
        distance = float(angular_distance_deg(
            theta_null, phi_null, case["theta0_deg"], case["phi0_deg"]))
        outside = bool(difference_null_is_legal(
            main_lobe, theta_null, phi_null))
        details.append({
            "theta_deg": float(theta_null),
            "phi_deg": float(phi_null) % 360.0,
            "distance_from_target_deg": distance,
            "outside_main_lobe": outside,
            "outside_generation_guard_15deg": bool(distance >= 15.0),
            "duplicate": duplicate,
        })
    return details


def _make_reference(case, axis, posx, posy, amp_x_sum, amp_y_sum):
    amp_x_diff, amp_y_diff, _ = bayliss_2d_separable(
        NX, NY, SLL_DESIGN, difference_axis=axis)
    phase_x, phase_y = beam_steering_phase_2d(
        posx, posy, case["theta0_deg"], case["phi0_deg"])
    sum_amp, sum_phase = combine_2d_excitation(
        amp_x_sum, amp_y_sum, phase_x, phase_y)
    diff_amp, diff_phase = combine_2d_excitation(
        amp_x_diff, amp_y_diff, phase_x, phase_y)
    return sum_amp, sum_phase, diff_amp, diff_phase


def _case_row(case, axis, index, sum_result, reference, adaptive,
              reference_nulls, adaptive_nulls, legality_reference,
              legality_adaptive):
    target = adaptive_nulls["center_db"][0]
    adaptive_centers = adaptive_nulls["center_db"][1:]
    adaptive_windows = adaptive_nulls["window_worst_db"][1:]
    sum_bw = sum_result["beamwidth_3db_deg"]
    threshold = pointing_threshold_deg(sum_bw)
    legal = all(item["outside_main_lobe"] and
                item["outside_generation_guard_15deg"] and
                not item["duplicate"] for item in legality_adaptive)
    pass_sll = adaptive["sll_db"] <= DIFFERENCE_SLL_THRESHOLD_DB
    pass_intrinsic = target <= ADAPTIVE_NULL_THRESHOLD_DB
    pass_adaptive = (len(adaptive_centers) == 4 and
                     all(value <= ADAPTIVE_NULL_THRESHOLD_DB
                         for value in adaptive_centers))
    pass_strict = (len(adaptive_centers) == 4 and
                   all(value <= STRICT_DIFFERENCE_NULL_THRESHOLD_DB
                       for value in adaptive_centers))
    pass_pointing = adaptive["pointing_error_deg"] <= threshold
    return {
        "case_id": case["case_id"],
        "set": case["set"],
        "difference_axis": axis,
        "difference_axis_phi_deg": difference_axis_to_phi_deg(axis),
        "target_theta_deg": case["theta0_deg"],
        "target_phi_deg": case["phi0_deg"],
        "nulls": legality_adaptive,
        "null_legality": {
            "reference_main_lobe": legality_reference,
            "adaptive_main_lobe": legality_adaptive,
            "pass": legal,
        },
        "sum_sll_db": sum_result["sll_db"],
        "sum_sll_pass": bool(sum_result["sll_db"] <= SUM_SLL_THRESHOLD_DB),
        "beamwidth_3db_sum": sum_bw,
        "reference_difference_sll_db": reference["sll_db"],
        "reference_intrinsic_null_center_db": reference_nulls["center_db"][0],
        "reference_pointing_error_deg": reference["pointing_error_deg"],
        "reference_main_lobe": reference["main_lobe"],
        "difference_sll_db": adaptive["sll_db"],
        "difference_sll_pass": bool(pass_sll),
        "pass_sll": bool(pass_sll),
        "intrinsic_null_center_db": target,
        "intrinsic_null_reporting": _display_db(target),
        "pass_intrinsic_null": bool(pass_intrinsic),
        "adaptive_null_center_db": adaptive_centers,
        "adaptive_null_center_reporting": [
            _display_db(value) for value in adaptive_centers
        ],
        "adaptive_null_window_worst_db": adaptive_windows,
        "adaptive_null_window_median_db": float(np.median(adaptive_windows)),
        "adaptive_null_window_worst_case_db": float(max(adaptive_windows)),
        "pass_adaptive_-30": bool(pass_adaptive),
        "pass_strict_-50": bool(pass_strict),
        "pointing_error_deg": adaptive["pointing_error_deg"],
        "pointing_threshold_deg": threshold,
        "pass_pointing": bool(pass_pointing),
        "zero_crossing_direction": adaptive["zero_crossing_direction"],
        "adaptive_main_lobe": adaptive["main_lobe"],
        "pass_null_legality": legal,
        "joint_pass": bool(
            legal and pass_sll and pass_intrinsic and pass_adaptive and
            pass_pointing),
        "joint_pass_strict": bool(
            legal and pass_sll and pass_intrinsic and pass_adaptive and
            pass_strict and pass_pointing),
        "metric_version": OFFICIAL_EVALUATOR_VERSION,
        "weight_reference": {
            "file": f"weights/{axis}_weights.npz",
            "index": index,
        },
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


def _null_aggregate(rows, key, threshold):
    all_values = [value for row in rows for value in row[key]]
    case_pass = [all(value <= threshold for value in row[key]) for row in rows]
    return {
        "cases": len(rows),
        "nulls": len(all_values),
        "pass": int(sum(case_pass)),
        "fail": int(len(case_pass) - sum(case_pass)),
        "not_tested": 0,
        "worst": float(max(all_values)),
        "best": float(min(all_values)),
        "median": float(np.median(all_values)),
        "threshold_db": threshold,
    }


def _pointing_aggregate(rows):
    errors = [row["pointing_error_deg"] for row in rows]
    thresholds = [row["pointing_threshold_deg"] for row in rows]
    return {
        **_aggregate(rows, "pointing_error_deg", "pass_pointing"),
        "rmse_deg": pointing_rmse_deg(errors),
        "threshold_definition": "per-case sum BW / 30",
        "threshold_min_deg": float(min(thresholds)),
        "threshold_median_deg": float(np.median(thresholds)),
        "threshold_max_deg": float(max(thresholds)),
    }


def _bucket_summary(rows):
    return {
        "cases": len(rows),
        "reference": {
            "difference_sll": _aggregate(
                rows, "reference_difference_sll_db",
                "reference_difference_sll_pass"),
            "intrinsic_null": _aggregate(
                rows, "reference_intrinsic_null_center_db",
                "reference_intrinsic_null_pass"),
            "pointing": _aggregate(
                rows, "reference_pointing_error_deg",
                "reference_pointing_pass"),
        },
        "adaptive": {
            "difference_sll": _aggregate(
                rows, "difference_sll_db", "difference_sll_pass"),
            "intrinsic_null": _aggregate(
                rows, "intrinsic_null_center_db", "pass_intrinsic_null"),
            "null_minus30": _null_aggregate(
                rows, "adaptive_null_center_db", ADAPTIVE_NULL_THRESHOLD_DB),
            "strict_null_minus50": _null_aggregate(
                rows, "adaptive_null_center_db",
                STRICT_DIFFERENCE_NULL_THRESHOLD_DB),
            "pointing": _pointing_aggregate(rows),
            "window_worst": {
                "median_db": float(np.median([
                    value for row in rows
                    for value in row["adaptive_null_window_worst_db"]
                ])),
                "worst_db": float(max(
                    value for row in rows
                    for value in row["adaptive_null_window_worst_db"])),
            },
            "joint": {
                "standard_pass": int(sum(row["joint_pass"] for row in rows)),
                "strict_pass": int(sum(row["joint_pass_strict"] for row in rows)),
                "standard_fail": int(sum(not row["joint_pass"] for row in rows)),
                "strict_fail": int(sum(not row["joint_pass_strict"] for row in rows)),
            },
        },
        "legality": {
            "cases": len(rows),
            "pass": int(sum(row["pass_null_legality"] for row in rows)),
            "fail": int(sum(not row["pass_null_legality"] for row in rows)),
            "nulls": 4 * len(rows),
        },
    }


def main():
    if not STAGE2_BASELINE.exists():
        raise SystemExit(f"Missing Stage 2 baseline: {STAGE2_BASELINE}")
    if FINAL_DIR.exists() or STAGING_DIR.exists():
        raise SystemExit(f"Refusing to overwrite Stage 2B artifact: {FINAL_DIR}")
    RESULT_PARENT.mkdir(parents=True, exist_ok=True)
    STAGING_DIR.mkdir()

    source_manifest, manifest, cases = _load_manifest()
    posx = uniform_linear_array_pos(NX)
    posy = uniform_linear_array_pos(NY)
    amp_x_sum, amp_y_sum = taylor_2d_separable(NX, NY, SLL_DESIGN)
    baseline_metadata = json.loads(
        (STAGE2_BASELINE / "metadata.json").read_text(encoding="utf-8"))
    metadata = {
        "stage": "STAGE_2B",
        "status": "FORMAL_DIFFERENCE_COMPLETENESS_FREEZE",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git": _git_state(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "evaluator_version": OFFICIAL_EVALUATOR_VERSION,
        "array": {"nx": NX, "ny": NY, "elements": NX * NY,
                  "frequency_hz": None,
                  "frequency_note": "repository uses normalized wavelength domain; no Hz configuration exists",
                  "wavelength": 1.0, "spacing_wavelengths": 0.5,
                  "coordinates": "centered uniform_linear_array_pos in x/y"},
        "scan_sets": {
            "regular": {"cases": 73,
                        "definition": "copied from Stage 2 case_manifest.json"},
            "random": {"cases": 200,
                        "source": str(STAGE2_BASELINE / "case_manifest.json"),
                        "seed": baseline_metadata["scan_sets"]["random"]["seed"],
                        "definition": baseline_metadata["scan_sets"]["random"]["distribution"],
                        "regenerated": False,
                        "target_duplicate_check": manifest["random_target_duplicates"],
                        "adaptive_null_rule": manifest["random_adaptive_null_rule"],
                        "source_nulls_retained_for_provenance": True},
        },
        "difference_axes": {
            "azimuth": {"local_array_axis": "+x / +u", "evaluator_phi_deg": 0.0,
                        "reference": "Bayliss on x, Taylor on y"},
            "elevation": {"local_array_axis": "+y / +v", "evaluator_phi_deg": 90.0,
                          "reference": "Taylor on x, Bayliss on y"},
        },
        "adaptive_null_generation": {
            "rule": "regular reuses Stage 2 null_dirs; random uses one fixed rule after the Stage 2 null audit",
            "regular": "existing deterministic get_null_dirs(target)",
            "random": "fixed theta=90 deg ring at target_phi+[45,135,225,315] deg; target cases unchanged and no new sampling",
            "guard": "spherical distance >= 15 degrees plus official Difference main-lobe exclusion",
            "duplicate_rule": "exact (theta, phi modulo 360) pairs rejected",
        },
        "reporting": {
            "raw_center_floor": "preserve evaluator values such as -300 dBc and -289 dBc",
            "formal_display_floor_db": REPORTING_FLOOR_DB,
            "formal_display_rule": "values at or below -100 dBc are shown as < -100 dBc",
            "interpretation": "numerical precision floor under ideal array model, not physical hardware dynamic range",
            "null_window": "diagnostic only; never promoted to strict gate",
        },
        "baseline_reference": str(STAGE2_BASELINE),
        "design": {"sll_design_db": SLL_DESIGN, "sum": "current Taylor", "difference": "parameterized Bayliss x Taylor"},
    }
    _write_json(STAGING_DIR / "metadata.json", metadata)
    _write_json(STAGING_DIR / "case_manifest_used.json", {
        "source_manifest": str(STAGE2_BASELINE / "case_manifest.json"),
        "source": source_manifest,
        "effective": manifest,
    })

    axis_rows = {axis: [] for axis in AXES}
    axis_weights = {}
    t_start = time.perf_counter()
    for axis in AXES:
        axis_phi = difference_axis_to_phi_deg(axis)
        ref_amp_rows, ref_phase_rows = [], []
        adaptive_amp_rows, adaptive_phase_rows = [], []
        for index, case in enumerate(cases):
            sum_amp, sum_phase, ref_amp, ref_phase = _make_reference(
                case, axis, posx, posy, amp_x_sum, amp_y_sum)
            sum_result = evaluate_sum_beam(
                sum_amp, sum_phase, posx, posy, case["theta0_deg"],
                case["phi0_deg"], n_uv=DEFAULT_GRID_SIZE)
            reference = evaluate_difference_beam(
                ref_amp, ref_phase, posx, posy, case["theta0_deg"],
                case["phi0_deg"], difference_axis_phi_deg=axis_phi,
                n_uv=DEFAULT_GRID_SIZE)
            reference_nulls = evaluate_nulls(
                ref_amp, ref_phase, posx, posy,
                [(case["theta0_deg"], case["phi0_deg"])],
                reference["_peak_abs"])
            legality_reference = _legality(case, reference["main_lobe"])
            if not all(item["outside_main_lobe"] and
                       item["outside_generation_guard_15deg"] and
                       not item["duplicate"] for item in legality_reference):
                raise RuntimeError(
                    f"illegal {axis} null scenario before adaptive closure: {case['case_id']}")

            adaptive_amp, adaptive_phase = capon_nulling_difference_2d(
                posx, posy, ref_amp, ref_phase, case["theta0_deg"],
                case["phi0_deg"], case["null_dirs"],
                difference_axis=axis)
            adaptive = evaluate_difference_beam(
                adaptive_amp, adaptive_phase, posx, posy,
                case["theta0_deg"], case["phi0_deg"],
                difference_axis_phi_deg=axis_phi, n_uv=DEFAULT_GRID_SIZE)
            adaptive_nulls = evaluate_nulls(
                adaptive_amp, adaptive_phase, posx, posy,
                [(case["theta0_deg"], case["phi0_deg"])] + case["null_dirs"],
                adaptive["_peak_abs"])
            legality_adaptive = _legality(case, adaptive["main_lobe"])
            row = _case_row(
                case, axis, index, sum_result, reference, adaptive,
                reference_nulls, adaptive_nulls, legality_reference,
                legality_adaptive)
            # Remove private evaluator peak fields from persisted data only;
            # all official public fields remain intact in the row.
            row["reference_difference_pointing"] = reference["zero_crossing_direction"]
            row["pointing_threshold_deg"] = pointing_threshold_deg(
                sum_result["beamwidth_3db_deg"])
            row["reference_difference_sll_pass"] = bool(
                reference["sll_db"] <= DIFFERENCE_SLL_THRESHOLD_DB)
            row["reference_intrinsic_null_pass"] = bool(
                reference_nulls["center_db"][0] <= ADAPTIVE_NULL_THRESHOLD_DB)
            row["reference_pointing_pass"] = bool(
                reference["pointing_error_deg"] <= row["pointing_threshold_deg"])
            axis_rows[axis].append(row)
            ref_amp_rows.append(ref_amp)
            ref_phase_rows.append(ref_phase)
            adaptive_amp_rows.append(adaptive_amp)
            adaptive_phase_rows.append(adaptive_phase)
            if (index + 1) % 25 == 0:
                elapsed = time.perf_counter() - t_start
                print(f"{axis} {index + 1}/{len(cases)} cases, elapsed={elapsed:.1f}s")
        axis_weights[axis] = {
            "reference_amp": np.asarray(ref_amp_rows),
            "reference_phase": np.asarray(ref_phase_rows),
            "adaptive_amp": np.asarray(adaptive_amp_rows),
            "adaptive_phase": np.asarray(adaptive_phase_rows),
            "theta0_deg": np.asarray([case["theta0_deg"] for case in cases]),
            "phi0_deg": np.asarray([case["phi0_deg"] for case in cases]),
            "set_code": np.asarray([0 if case["set"] == "regular" else 1 for case in cases]),
        }

    summary = {
        "stage": "STAGE_2B",
        "metric_version": OFFICIAL_EVALUATOR_VERSION,
        "case_counts": {"regular": 73, "random": 200, "overall": 273,
                         "axes": 2, "adaptive_null_targets": 2 * 273 * 4},
        "axes": {},
        "reporting_rule": metadata["reporting"],
        "random_source": str(STAGE2_BASELINE / "case_manifest.json"),
        "official_thresholds": {
            "difference_sll_db": DIFFERENCE_SLL_THRESHOLD_DB,
            "intrinsic_null_db": ADAPTIVE_NULL_THRESHOLD_DB,
            "adaptive_null_db": ADAPTIVE_NULL_THRESHOLD_DB,
            "strict_difference_null_db": STRICT_DIFFERENCE_NULL_THRESHOLD_DB,
        },
    }
    representatives = {}
    for axis in AXES:
        rows = axis_rows[axis]
        summary["axes"][axis] = {
            "regular": _bucket_summary([row for row in rows if row["set"] == "regular"]),
            "random": _bucket_summary([row for row in rows if row["set"] == "random"]),
            "overall": _bucket_summary(rows),
        }
        ordered = sorted(enumerate(rows), key=lambda item: item[1]["difference_sll_db"])
        reps = {"best": ordered[0], "median": ordered[len(ordered) // 2], "worst": ordered[-1]}
        representatives[axis] = {
            label: {"case_id": rows[item[0]]["case_id"], "index": item[0],
                    "difference_sll_db": item[1]["difference_sll_db"]}
            for label, item in reps.items()
        }
        for set_name, filename in (("regular", f"{axis}_regular_cases.json"),
                                   ("random", f"{axis}_random_cases.json")):
            _write_json(
                STAGING_DIR / filename,
                [row for row in rows if row["set"] == set_name])
    _write_json(STAGING_DIR / "representatives.json", representatives)
    _write_json(STAGING_DIR / "summary.json", summary)
    _write_json(STAGING_DIR / "stage2b_difference_completeness_summary.json", summary)

    weights_dir = STAGING_DIR / "weights"
    weights_dir.mkdir()
    for axis in AXES:
        np.savez_compressed(weights_dir / f"{axis}_weights.npz",
                            **axis_weights[axis])

    os.rename(STAGING_DIR, FINAL_DIR)
    print(f"Stage 2B artifact written: {FINAL_DIR}")


if __name__ == "__main__":
    main()
