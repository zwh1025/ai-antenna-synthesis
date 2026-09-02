"""Stage 4B failure-aware weight reconstruction closure.

The runner consumes the Stage 4A failure artifact byte-for-byte at the
logical mask level.  It never regenerates masks, changes the frozen targets,
or changes the official evaluator.  Stage 4B's formal proposal is a small
active-coordinate minimum-norm LCMV correction; B0 and B1 are retained as
comparison baselines.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "project"
STAGE4A = ROOT / "results" / "stage4a_robustness_degradation"
OUT = ROOT / "results" / "stage4b_failure_reconstruction"
BASELINE = ROOT / "results" / "stage2_strict_closure" / "baseline"
STAGE2B = ROOT / "results" / "stage2b_difference_completeness" / "formal"

NX = NY = 32
N_ELEMENTS = NX * NY
METRIC_VERSION = "1.0.0"
ROBUSTNESS_MODEL_VERSION = "1.0.0"
FAILURE_RECONSTRUCTION_VERSION = "1.0.0"
GRID_SIZE = 201
LEVELS = (("20pct", 0.20), ("10pct", 0.10), ("5pct", 0.05))
MAX_WORKERS = min(8, max(1, os.cpu_count() or 1))
PILOT_SEED = 0

sys.path.insert(0, str(PROJECT))
import run_stage4a_robustness_degradation as stage4a  # noqa: E402
from mylib.antenna_calc import uniform_linear_array_pos  # noqa: E402
from mylib.failure_reconstruction import (  # noqa: E402
    FAILURE_RECONSTRUCTION_VERSION,
    _field_matrix,
    active_renormalization,
    complex_weights,
    minimum_norm_active_lcmv,
    no_reconstruction,
)
from mylib.official_evaluator import (  # noqa: E402
    ADAPTIVE_NULL_THRESHOLD_DB,
    DIFFERENCE_SLL_THRESHOLD_DB,
    OFFICIAL_EVALUATOR_VERSION,
    SUM_SLL_THRESHOLD_DB,
    evaluate_official_case,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_hash(value) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        value, ensure_ascii=False, indent=2, allow_nan=False
    ) + "\n", encoding="utf-8")


def git_state() -> dict:
    def run(*args):
        return subprocess.run(
            ["git", *args], cwd=ROOT, check=True, capture_output=True,
            text=True,
        ).stdout.strip()
    status = run("status", "--short")
    return {
        "branch": run("branch", "--show-current"),
        "head": run("rev-parse", "HEAD"),
        "dirty": bool(status),
        "dirty_files": status.splitlines() if status else [],
        "commit_or_push_performed": False,
    }


def _metric_snapshot(row: dict) -> dict:
    keys = (
        "sum_sll_db", "sum_sll_pass", "sum_beamwidth_3db_deg",
        "sum_peak_direction", "sum_pointing_error_deg",
        "sum_pointing_threshold_deg", "difference_sll_db",
        "difference_sll_pass", "difference_zero_crossing_direction",
        "difference_pointing_error_deg", "difference_pointing_threshold_deg",
        "difference_intrinsic_null_db", "difference_intrinsic_null_pass",
        "sum_null_center_db", "sum_null_window_worst_db", "sum_null_worst_db",
        "sum_null_pass", "sum_null_available", "sum_null_status",
        "common_joint_pass", "track_p_joint_pass",
        "difference_adaptive_null_status",
        "difference_adaptive_null_center_db", "difference_adaptive_null_pass",
        "difference_adaptive_joint_pass",
    )
    return {key: row[key] for key in keys if key in row}


def _stats(values) -> dict:
    values = [float(value) for value in values if value is not None]
    if not values:
        return {"n": 0, "mean": None, "median": None, "std": None,
                "P5": None, "P95": None, "min": None, "max": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size), "mean": float(np.mean(arr)),
        "median": float(np.median(arr)), "std": float(np.std(arr)),
        "P5": float(np.percentile(arr, 5)),
        "P95": float(np.percentile(arr, 95)),
        "min": float(np.min(arr)), "max": float(np.max(arr)),
    }


def _pass_stats(values) -> dict:
    values = [bool(value) for value in values if value is not None]
    return {
        "n": len(values), "pass": int(sum(values)),
        "pass_rate": float(np.mean(values)) if values else None,
        "status": "MEASURED" if values else "NOT_AVAILABLE",
    }


def _aggregate(rows: list[dict], metric_key: str = "b2") -> dict:
    values = [row[metric_key]["metrics"].get("sum_sll_db") for row in rows]
    delta_keys = {
        "sum_sll": "sum_sll_recovery_db",
        "difference_sll": "difference_sll_recovery_db",
        "sum_pointing": "sum_pointing_recovery_deg",
        "difference_pointing": "difference_pointing_recovery_deg",
        "sum_null": "sum_null_recovery_db",
        "difference_intrinsic_null": "difference_intrinsic_null_recovery_db",
    }
    out = {
        "n_realizations": len(rows),
        "sum_sll_db": _stats(values),
        "difference_sll_db": _stats([
            row[metric_key]["metrics"].get("difference_sll_db") for row in rows
        ]),
        "sum_pointing_error_deg": _stats([
            row[metric_key]["metrics"].get("sum_pointing_error_deg") for row in rows
        ]),
        "difference_pointing_error_deg": _stats([
            row[metric_key]["metrics"].get("difference_pointing_error_deg") for row in rows
        ]),
        "sum_null_worst_db": _stats([
            row[metric_key]["metrics"].get("sum_null_worst_db") for row in rows
        ]),
        "difference_intrinsic_null_db": _stats([
            row[metric_key]["metrics"].get("difference_intrinsic_null_db") for row in rows
        ]),
        "compliance": {
            "sum_sll": _pass_stats([row[metric_key]["metrics"].get("sum_sll_pass") for row in rows]),
            "difference_sll": _pass_stats([row[metric_key]["metrics"].get("difference_sll_pass") for row in rows]),
            "difference_pointing": _pass_stats([row[metric_key]["metrics"].get("difference_pointing_pass") for row in rows]),
            "difference_intrinsic_null": _pass_stats([row[metric_key]["metrics"].get("difference_intrinsic_null_pass") for row in rows]),
            "common_joint": _pass_stats([row[metric_key]["metrics"].get("common_joint_pass") for row in rows]),
            "track_p_joint": _pass_stats([row[metric_key]["metrics"].get("track_p_joint_pass") for row in rows]),
            "difference_adaptive_joint": _pass_stats([row[metric_key]["metrics"].get("difference_adaptive_joint_pass") for row in rows]),
        },
        "recovery": {},
    }
    for name, key in delta_keys.items():
        out["recovery"][name] = _stats([
            row[metric_key].get("recovery", {}).get(key) for row in rows
        ])
    return out


def _load_inputs():
    required = [
        "failure_cases.json", "summary.json", "decision.json",
        "robustness_case_manifest.json", "seed_manifest.json",
        "ideal_reference.json",
    ]
    missing = [name for name in required if not (STAGE4A / name).exists()]
    if missing:
        raise FileNotFoundError(f"missing Stage 4A artifacts: {missing}")
    stage4a_decision = read_json(STAGE4A / "decision.json")
    if stage4a_decision.get("gate") != "STAGE_4A_GO":
        raise RuntimeError("Stage 4B requires STAGE_4A_GO")
    manifest = read_json(STAGE4A / "robustness_case_manifest.json")
    cases = manifest["cases"]
    failure = read_json(STAGE4A / "failure_cases.json")
    if len(cases) != 16 or len(failure["realizations"]) != 960:
        raise RuntimeError("Stage 4A case/mask counts do not match the frozen contract")
    ideal = {row["case_id"]: row for row in read_json(
        STAGE4A / "ideal_reference.json"
    )["cases"]}
    failure_rows = {}
    for row in failure["realizations"]:
        key = (row["case_id"], f"{float(row['failure_rate']):.2f}", int(row["seed"]))
        if key in failure_rows:
            raise RuntimeError(f"duplicate Stage 4A failure row {key}")
        indices = np.asarray(row["failed_indices"], dtype=np.int64)
        if len(indices) != int(row["failed_count"]):
            raise RuntimeError(f"failure count mismatch in Stage 4A row {key}")
        mask = np.zeros(N_ELEMENTS, dtype=bool)
        mask[indices] = True
        if stage4a._mask_hash(mask) != row["mask_sha256"]:
            raise RuntimeError(f"mask hash mismatch in Stage 4A row {key}")
        failure_rows[key] = row
    return cases, ideal, failure_rows


def _load_weights(cases):
    regular = np.load(BASELINE / "weights" / "regular_weights.npz")
    random = np.load(BASELINE / "weights" / "random_weights.npz")
    data = {}
    for case in cases:
        z = regular if case["set"] == "regular" else random
        index = int(case["source_index"])
        sum_key = "lcmv" if case["set"] == "regular" else "taylor"
        data[case["case_id"]] = {
            "sum_amp": z[f"{sum_key}_amp"][index].astype(np.float64, copy=True),
            "sum_phase": z[f"{sum_key}_phase"][index].astype(np.float64, copy=True),
            "difference_amp": z["difference_amp"][index].astype(np.float64, copy=True),
            "difference_phase": z["difference_phase"][index].astype(np.float64, copy=True),
        }
    return data


def _make_mask(indices):
    mask = np.zeros((NX, NY), dtype=bool)
    mask.reshape(-1)[np.asarray(indices, dtype=np.int64)] = True
    return mask


def _stage4b_metrics(official: dict, case: dict, ideal: dict) -> dict:
    metrics = stage4a.compact_metrics(official, case, ideal)
    diff_null = official["adaptive_null"]["difference"]
    if diff_null is not None and len(diff_null["center_db"]) >= 5:
        adaptive_centers = [float(value) for value in diff_null["center_db"][1:]]
        metrics["difference_adaptive_null_center_db"] = adaptive_centers
        metrics["difference_adaptive_null_pass"] = bool(
            all(value <= ADAPTIVE_NULL_THRESHOLD_DB for value in adaptive_centers)
        )
        metrics["difference_adaptive_joint_pass"] = bool(
            metrics["difference_sll_pass"] and
            metrics["difference_pointing_pass"] and
            metrics["difference_intrinsic_null_pass"] and
            metrics["difference_adaptive_null_pass"]
        )
        metrics["difference_adaptive_null_status"] = "MEASURED_BY_STAGE4B_RECONSTRUCTION"
    else:
        metrics["difference_adaptive_null_center_db"] = None
        metrics["difference_adaptive_null_pass"] = None
        metrics["difference_adaptive_joint_pass"] = None
        metrics["difference_adaptive_null_status"] = "NOT_AVAILABLE"
    return metrics


def _solver_snapshot(solver: dict) -> dict:
    """Remove internal ndarray weight tuples before JSON serialization."""
    return {
        beam: metadata
        for beam, (_, _, metadata) in solver.items()
    }


def _evaluate_job(job):
    started = time.perf_counter()
    official = evaluate_official_case(
        job["amp_sum"], job["phase_sum"], job["posx"], job["posy"],
        job["case"]["theta0_deg"], job["case"]["phi0_deg"],
        null_dirs=job["case"]["null_dirs"],
        amp_difference=job["amp_difference"],
        phase_difference=job["phase_difference"],
        difference_null_dirs=[
            (job["case"]["theta0_deg"], job["case"]["phi0_deg"]),
            *job["case"]["null_dirs"],
        ],
        difference_axis_phi_deg=0.0,
        n_uv=GRID_SIZE,
    )
    return official, time.perf_counter() - started


def _recovery(degraded: dict, reconstructed: dict, ideal: dict) -> dict:
    def gain(key):
        a, b = degraded.get(key), reconstructed.get(key)
        return float(a - b) if a is not None and b is not None else None
    result = {
        "sum_sll_recovery_db": gain("sum_sll_db"),
        "difference_sll_recovery_db": gain("difference_sll_db"),
        "sum_pointing_recovery_deg": gain("sum_pointing_error_deg"),
        "difference_pointing_recovery_deg": gain("difference_pointing_error_deg"),
        "sum_null_recovery_db": gain("sum_null_worst_db"),
        "difference_intrinsic_null_recovery_db": gain("difference_intrinsic_null_db"),
        "sum_beamwidth_change_deg": (
            reconstructed.get("sum_beamwidth_3db_deg") - degraded.get("sum_beamwidth_3db_deg")
            if reconstructed.get("sum_beamwidth_3db_deg") is not None and degraded.get("sum_beamwidth_3db_deg") is not None else None
        ),
        "before_common_joint_pass": degraded.get("common_joint_pass"),
        "after_common_joint_pass": reconstructed.get("common_joint_pass"),
        "before_track_p_joint_pass": degraded.get("track_p_joint_pass"),
        "after_track_p_joint_pass": reconstructed.get("track_p_joint_pass"),
        "after_difference_adaptive_joint_pass": reconstructed.get("difference_adaptive_joint_pass"),
        "normalized_peak_loss_claim": "not applicable: official patterns are peak-normalized; pointing/beamwidth are retained instead",
    }
    denom = degraded.get("sum_sll_db") - ideal.get("sum_sll_db")
    result["sum_sll_recovery_fraction"] = (
        result["sum_sll_recovery_db"] / denom
        if result["sum_sll_recovery_db"] is not None and abs(denom) > 1e-9 else None
    )
    return result


def _build_definition():
    return {
        "failure_reconstruction_version": FAILURE_RECONSTRUCTION_VERSION,
        "primary_method": "B2_field_fit_active_lcmv",
        "method_selection_reason": "existing active-set LCMV is legacy-evaluator code; this is the smallest arbitrary-active-coordinate constrained least-squares correction that preserves the frozen target/null formulation",
        "common_hyperparameters": {
            "solver": "Cholesky solve plus numpy.linalg.lstsq for the small constraint system",
            "rcond": 1e-12,
            "regularization": "reference ridge with fixed ratio 0.05",
            "stopping_criteria": "closed-form solve; no iterative stopping or mask-specific tuning",
            "level_dependent_parameters": False,
        },
        "B0": "No Reconstruction: exact Stage 4A masked reference; no normalization or redesign.",
        "B1": "Masked / Active Re-normalization: failed entries zero, active complex weights unchanged up to one common scale.",
        "B2_formulation": {
            "active_set": "A = complement of the exact Stage 4A frozen failure mask",
            "sum": "min ||A_active w_A-A_full w0||_2^2 + rho||w_A-w0_A||_2^2 subject only to preserving a(theta0,phi0)^H w_A=(a(theta0,phi0)^H w0); the four frozen Sum nulls remain in the ideal-field objective and are measured after reconstruction",
            "difference": "min ||A_active w_A-A_full w0||_2^2 + rho||w_A-w0_A||_2^2 subject to a(theta0,phi0)^H w_A=0 and a(null_j)^H w_A=0 for the four frozen nulls",
            "coordinates": "arbitrary active x/y coordinates from the original planar 32x32 array; failed coordinates are excluded from variables",
            "complex_solution": "R=A_active^H A_active+rho I; w=R^-1(A_active^H A_full w0+rho w0)+R^-1 C(C^H R^-1 C)^+(f-C^H R^-1 rhs)",
            "field_grid": "61 theta samples × 121 azimuth samples, visible domain; A_full is the ideal reference field and A_active is the same grid on active coordinates",
            "reference_ridge_ratio": 0.05,
        },
        "reconstruction_policy": "Every case and every level uses the same B0/B1/B2 definitions; no performance-selected parameters.",
        "formal_primary": "Sum B2 on all 320 masks at each of 20%, 10%, and 5%; Difference azimuth is evaluated in the same official call.",
        "elevation_status": "not part of Stage 4A formal weight source; Stage 2B elevation remains a separate capability and is not silently mixed into this exact baseline comparison",
        "frozen_before_pilot": True,
    }


def _build_power_policy():
    return {
        "policy": "A",
        "definition": "For B1 and B2 separately, ||w_reconstructed||_2 is scaled to ||w0||_2 of the corresponding ideal beam. B0 retains the degraded masked norm.",
        "failed_entries": "exact zero after reconstruction",
        "power_increase_for_B1_B2": False,
        "normalization_is_common_complex_scale": True,
        "pattern_invariance": "common positive scaling does not change official peak-normalized dBc metrics",
        "main_beam_response_loss": "not reported as an absolute gain loss because the frozen official evaluator peak-normalizes each pattern; pointing and beamwidth are reported",
        "amplitude_limit_policy": "no additional per-element amplitude limit was invented; finite weights and exact failed zeros are checked",
    }


def _prepare_freeze(cases, ideal, failure_rows):
    OUT.mkdir(parents=True, exist_ok=True)
    definition = _build_definition()
    path = OUT / "algorithm_definition.json"
    if path.exists():
        old = read_json(path)
        if old.get("definition_sha256") != json_hash(definition):
            if any((OUT / f"failure_{key}_cases.json").exists() for key, _ in LEVELS):
                raise RuntimeError("algorithm definition changed after Stage 4B formal freeze")
            write_json(path, {**definition, "definition_sha256": json_hash(definition)})
            definition = read_json(path)
        else:
            definition = old
    else:
        write_json(path, {**definition, "definition_sha256": json_hash(definition)})
        definition = read_json(path)
    write_json(OUT / "power_normalization.json", _build_power_policy())
    write_json(OUT / "seed_manifest.json", {
        "source": str(STAGE4A / "seed_manifest.json"),
        "source_sha256": sha256_file(STAGE4A / "seed_manifest.json"),
        "rule": "Stage 4B uses exactly the Stage 4A case_id, rate, seed, and failed_indices; no RNG call creates a new mask.",
        "pilot": {"failure_rate": 0.20, "seed": PILOT_SEED, "cases": len(cases)},
        "formal": {"rates": [0.20, 0.10, 0.05], "seeds": list(range(20)), "cases_per_rate": len(cases)},
    })
    metadata = {
        "stage": "Stage 4B — Failure-Aware Weight Reconstruction Closure",
        "run_status": "PILOT_READY",
        "generated_utc": utc_now(),
        "git_start": git_state(),
        "metric_version": METRIC_VERSION,
        "robustness_model_version": ROBUSTNESS_MODEL_VERSION,
        "failure_reconstruction_version": FAILURE_RECONSTRUCTION_VERSION,
        "official_evaluator_version": OFFICIAL_EVALUATOR_VERSION,
        "official_grid_size": GRID_SIZE,
        "array": {"nx": NX, "ny": NY, "elements": N_ELEMENTS,
                  "spacing_wavelengths": 0.5, "geometry": "Stage 4A frozen planar geometry"},
        "stage4a_artifacts": {
            name: sha256_file(STAGE4A / name) for name in (
                "failure_cases.json", "summary.json", "decision.json",
                "robustness_case_manifest.json", "seed_manifest.json",
                "ideal_reference.json",
            )
        },
        "target_manifest_sha256": sha256_file(STAGE4A / "robustness_case_manifest.json"),
        "target_case_list_sha256": read_json(STAGE4A / "robustness_case_manifest.json")["case_list_sha256"],
        "seed_manifest_sha256": sha256_file(STAGE4A / "seed_manifest.json"),
        "failure_mask_count": len(failure_rows),
        "source_weights": rel(BASELINE),
        "forbidden_actions": [
            "no AI failure reconstruction", "no DeepSets retraining/fine-tuning",
            "no position/frequency/amplitude-phase reconstruction", "no NPU",
            "no Stage 2/Stage 3C1/evaluator modification", "no target/mask deletion",
            "no README/final report modification", "no commit", "no push",
        ],
    }
    pilot_path = OUT / "pilot_results.json"
    if pilot_path.exists():
        pilot = read_json(pilot_path)
        metadata["pilot_status"] = {
            "complete": pilot.get("failure_count") == 0,
            "case_count": pilot.get("case_count"),
            "failure_rate": pilot.get("failure_rate"),
            "seed": pilot.get("seed"),
        }
    write_json(OUT / "metadata.json", metadata)
    return definition, metadata


def _build_field_cache(cases, case_data, posx_2d, posy_2d):
    """Build the fixed-geometry Gram matrix once; this is an exact algebraic cache."""
    started = time.perf_counter()
    field = _field_matrix(
        posx_2d.ravel(), posy_2d.ravel(), 0.0, 0.0, 1.0
    )
    gram = field.conj().T @ field
    rhs = {}
    for case in cases:
        source = case_data[case["case_id"]]
        sum_w0 = complex_weights(source["sum_amp"], source["sum_phase"]).ravel()
        diff_w0 = complex_weights(
            source["difference_amp"], source["difference_phase"]
        ).ravel()
        rhs[case["case_id"]] = {
            "sum": gram @ sum_w0,
            "difference": gram @ diff_w0,
        }
    return {
        "field_gram": gram,
        "field_sample_count": int(field.shape[0]),
        "rhs": rhs,
        "build_runtime_s": float(time.perf_counter() - started),
    }


def _make_reconstruction(case, source, mask, posx_2d, posy_2d,
                         field_cache=None):
    started = time.perf_counter()
    b0_sum = no_reconstruction(source["sum_amp"], source["sum_phase"], mask)
    b0_diff = no_reconstruction(source["difference_amp"], source["difference_phase"], mask)
    b1_sum_started = time.perf_counter()
    b1_sum = active_renormalization(source["sum_amp"], source["sum_phase"], mask)
    b1_sum_time = time.perf_counter() - b1_sum_started
    b1_diff_started = time.perf_counter()
    b1_diff = active_renormalization(source["difference_amp"], source["difference_phase"], mask)
    b1_diff_time = time.perf_counter() - b1_diff_started
    shared_system = {} if field_cache is not None else None
    field_kwargs = {}
    field_rhs = {"sum": None, "difference": None}
    if field_cache is not None:
        field_rhs = field_cache["rhs"][case["case_id"]]
        field_kwargs = {
            "field_gram": field_cache["field_gram"],
        }
    b2_sum_started = time.perf_counter()
    b2_sum = minimum_norm_active_lcmv(
        source["sum_amp"], source["sum_phase"], mask, posx_2d, posy_2d,
        case["theta0_deg"], case["phi0_deg"], case["null_dirs"],
        beam_kind="sum",
        field_rhs=field_rhs["sum"],
        shared_system=shared_system,
        **field_kwargs,
    )
    b2_sum_time = time.perf_counter() - b2_sum_started
    b2_diff_started = time.perf_counter()
    b2_diff = minimum_norm_active_lcmv(
        source["difference_amp"], source["difference_phase"], mask,
        posx_2d, posy_2d, case["theta0_deg"], case["phi0_deg"],
        case["null_dirs"], beam_kind="difference",
        field_rhs=field_rhs["difference"],
        shared_system=shared_system,
        **field_kwargs,
    )
    b2_diff_time = time.perf_counter() - b2_diff_started
    b1_sum[2]["runtime_s"] = float(b1_sum_time)
    b1_diff[2]["runtime_s"] = float(b1_diff_time)
    b2_sum[2]["runtime_s"] = float(b2_sum_time)
    b2_diff[2]["runtime_s"] = float(b2_diff_time)
    return {
        "B0": {"sum": b0_sum, "difference": b0_diff},
        "B1": {"sum": b1_sum, "difference": b1_diff},
        "B2": {"sum": b2_sum, "difference": b2_diff},
        "total_runtime_s": float(time.perf_counter() - started),
    }


def _run_pilot(cases, ideal, failure_rows, case_data, posx, posy,
               field_cache):
    posx_2d = np.broadcast_to(posx[:, None], (NX, NY))
    posy_2d = np.broadcast_to(posy[None, :], (NX, NY))
    rows = []
    for case in cases:
        key = (case["case_id"], "0.20", PILOT_SEED)
        source = case_data[case["case_id"]]
        stage4a_row = failure_rows[key]
        mask = _make_mask(stage4a_row["failed_indices"])
        reconstructed = _make_reconstruction(
            case, source, mask, posx_2d, posy_2d, field_cache
        )
        jobs = []
        for method in ("B1", "B2"):
            sum_amp, sum_phase, _ = reconstructed[method]["sum"]
            diff_amp, diff_phase, _ = reconstructed[method]["difference"]
            jobs.append({"method": method, "case": case, "amp_sum": sum_amp,
                         "phase_sum": sum_phase, "amp_difference": diff_amp,
                         "phase_difference": diff_phase, "posx": posx, "posy": posy})
        method_results = {}
        for job in jobs:
            official, eval_s = _evaluate_job(job)
            metrics = _stage4b_metrics(official, case, ideal[case["case_id"]])
            method_results[job["method"]] = {
                "metrics": metrics, "evaluation_runtime_s": eval_s,
                "solver": _solver_snapshot(reconstructed[job["method"]]),
                "recovery": _recovery(
                    _stage4a_degraded_snapshot(stage4a_row), metrics,
                    ideal[case["case_id"]]
                ),
            }
        degraded = _stage4a_degraded_snapshot(stage4a_row)
        _assert_stage4a_metric_consistency(stage4a_row, degraded)
        rows.append({
            "case_id": case["case_id"], "failure_rate": 0.20,
            "seed": PILOT_SEED, "failed_indices": stage4a_row["failed_indices"],
            "degraded_metrics_stage4a": degraded,
            "B1": method_results["B1"], "B2": method_results["B2"],
            "B2_recovery": method_results["B2"]["recovery"],
            "pilot_solver_status": "all B1/B2 solves completed",
        })
    pilot = {
        "stage": "Stage 4B pilot",
        "frozen_rule": "all selected cases, seed 0, 20%; selection was fixed before measurement",
        "failure_rate": 0.20, "seed": PILOT_SEED, "case_count": len(rows),
        "success_count": len(rows), "failure_count": 0,
        "rows": rows,
        "aggregate_B1": _aggregate(rows, "B1"),
        "aggregate_B2": _aggregate(rows, "B2"),
        "pilot_is_not_the_formal_gate": True,
    }
    write_json(OUT / "pilot_results.json", pilot)
    return pilot


def _stage4a_degraded_snapshot(row):
    result = _metric_snapshot(row)
    result["stage4a_source"] = "failure_cases.json exact row; not re-evaluated or regenerated"
    result["stage4a_row_key"] = {
        "case_id": row["case_id"], "failure_rate": row["failure_rate"],
        "seed": row["seed"], "mask_sha256": row["mask_sha256"],
    }
    return result


def _assert_stage4a_metric_consistency(stage4a_row, degraded):
    """Assert that every loaded degraded metric is the frozen Stage 4A value."""
    numeric_keys = (
        "sum_sll_db", "difference_sll_db", "sum_pointing_error_deg",
        "difference_pointing_error_deg", "difference_intrinsic_null_db",
        "sum_null_worst_db", "sum_beamwidth_3db_deg",
    )
    for key in numeric_keys:
        original = stage4a_row.get(key)
        loaded = degraded.get(key)
        if original is None or loaded is None:
            if original != loaded:
                raise RuntimeError(f"Stage 4A metric mismatch for {key}")
        elif abs(float(original) - float(loaded)) >= 1e-12:
            raise RuntimeError(
                f"Stage 4A metric mismatch for {key}: {original} != {loaded}"
            )
    return True


def _formal_row(case, stage4a_row, ideal, source, mask, recon, posx, posy):
    jobs = []
    for method in ("B1", "B2"):
        sum_amp, sum_phase, _ = recon[method]["sum"]
        diff_amp, diff_phase, _ = recon[method]["difference"]
        jobs.append({"method": method, "case": case, "amp_sum": sum_amp,
                     "phase_sum": sum_phase, "amp_difference": diff_amp,
                     "phase_difference": diff_phase, "posx": posx, "posy": posy})
    return {"jobs": jobs, "case": case, "stage4a_row": stage4a_row,
            "ideal": ideal, "source": source, "mask": mask, "recon": recon}


def _complete_formal_row(bundle, evaluated):
    case = bundle["case"]
    stage4a_row = bundle["stage4a_row"]
    degraded = _stage4a_degraded_snapshot(stage4a_row)
    metric_consistent = _assert_stage4a_metric_consistency(stage4a_row, degraded)
    result = {
        "case_id": case["case_id"],
        "target": {"theta0_deg": case["theta0_deg"], "phi0_deg": case["phi0_deg"]},
        "failure_fraction": float(stage4a_row["failure_rate"]),
        "seed": int(stage4a_row["seed"]),
        "failed_indices": stage4a_row["failed_indices"],
        "mask_sha256": stage4a_row["mask_sha256"],
        "ideal_metrics": _metric_snapshot(bundle["ideal"]),
        "degraded_metrics_stage4a": degraded,
        "B0_no_reconstruction": {
            "metrics": degraded,
            "solver": _solver_snapshot(bundle["recon"]["B0"]),
        },
    }
    for method, (official, eval_s) in zip(("B1", "B2"), evaluated):
        metrics = _stage4b_metrics(official, case, bundle["ideal"])
        solver_full = bundle["recon"][method]
        solver = _solver_snapshot(solver_full)
        solver_time = float(solver["sum"].get("runtime_s", 0.0) + solver["difference"].get("runtime_s", 0.0))
        result[method] = {
            "metrics": metrics,
            "solver": solver,
            "evaluation_runtime_s": float(eval_s),
            "reconstruction_runtime_s": solver_time,
            "recovery": _recovery(degraded, metrics, bundle["ideal"]),
        }
    result["recovery"] = result["B2"]["recovery"]
    result["validity"] = {
        "failed_sum_amp_zero": bool(np.all(bundle["recon"]["B2"]["sum"][0].reshape(-1)[bundle["mask"].reshape(-1)] == 0.0)),
        "failed_difference_amp_zero": bool(np.all(bundle["recon"]["B2"]["difference"][0].reshape(-1)[bundle["mask"].reshape(-1)] == 0.0)),
        "solver_status": bundle["recon"]["B2"]["sum"][2]["solver_status"],
        "finite": bool(bundle["recon"]["B2"]["sum"][2]["finite"] and bundle["recon"]["B2"]["difference"][2]["finite"]),
        "stage4a_metric_consistency": metric_consistent,
    }
    return result


def _run_formal_level(level_key, rate, cases, ideal, failure_rows, case_data,
                      posx, posy, field_cache):
    posx_2d = np.broadcast_to(posx[:, None], (NX, NY))
    posy_2d = np.broadcast_to(posy[None, :], (NX, NY))
    bundles = []
    weight_rows = []
    for case in cases:
        source = case_data[case["case_id"]]
        for seed in range(20):
            key = (case["case_id"], f"{rate:.2f}", seed)
            stage4a_row = failure_rows[key]
            mask = _make_mask(stage4a_row["failed_indices"])
            recon = _make_reconstruction(
                case, source, mask, posx_2d, posy_2d, field_cache
            )
            bundles.append(_formal_row(case, stage4a_row, ideal[case["case_id"]], source, mask, recon, posx, posy))
            b2_sum_amp, b2_sum_phase, _ = recon["B2"]["sum"]
            b2_diff_amp, b2_diff_phase, _ = recon["B2"]["difference"]
            weight_rows.append({
                "case_id": case["case_id"], "seed": seed,
                "failure_mask": mask.copy(),
                "sum_amp": b2_sum_amp, "sum_phase": b2_sum_phase,
                "difference_amp": b2_diff_amp, "difference_phase": b2_diff_phase,
            })
    jobs = []
    job_owner = []
    for owner, bundle in enumerate(bundles):
        jobs.extend(bundle["jobs"])
        job_owner.extend((owner, owner))
    print(f"[Stage4B] {level_key}: evaluating {len(jobs)} B1/B2 official rows", flush=True)
    evaluated_by_owner = [[] for _ in bundles]
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for index, (job, outcome) in enumerate(zip(jobs, executor.map(_evaluate_job, jobs)), 1):
            evaluated_by_owner[job_owner[index - 1]].append(outcome)
            if index == len(jobs) or index % 100 == 0:
                print(f"[Stage4B] {level_key}: {index}/{len(jobs)}", flush=True)
    rows = [_complete_formal_row(bundle, evaluated_by_owner[index])
            for index, bundle in enumerate(bundles)]
    weights_dir = OUT / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        weights_dir / f"reconstructed_{level_key}.npz",
        case_id=np.asarray([row["case_id"] for row in weight_rows]),
        seed=np.asarray([row["seed"] for row in weight_rows], dtype=np.int64),
        failure_mask=np.asarray([row["failure_mask"] for row in weight_rows], dtype=bool),
        sum_amp=np.asarray([row["sum_amp"] for row in weight_rows]),
        sum_phase=np.asarray([row["sum_phase"] for row in weight_rows]),
        difference_amp=np.asarray([row["difference_amp"] for row in weight_rows]),
        difference_phase=np.asarray([row["difference_phase"] for row in weight_rows]),
    )
    payload = {
        "stage": "Stage 4B formal failure reconstruction",
        "failure_rate": rate,
        "failure_fraction_label": level_key,
        "case_count": len(cases), "realization_count": len(rows),
        "mask_source": str(STAGE4A / "failure_cases.json"),
        "mask_reuse": "exact failed_indices and mask_sha256 loaded from Stage 4A; no mask regeneration",
        "primary_method": "B2_field_fit_active_lcmv",
        "weights_artifact": str(weights_dir / f"reconstructed_{level_key}.npz"),
        "rows": rows,
        "aggregate_B0": _aggregate(rows, "B0_no_reconstruction"),
        "aggregate_B1": _aggregate(rows, "B1"),
        "aggregate_B2": _aggregate(rows, "B2"),
    }
    write_json(OUT / f"failure_{level_key}_cases.json", payload)
    return payload


def _runtime_from_rows(payloads):
    reconstruction = {"B1": [], "B2": []}
    evaluation = {"B1": [], "B2": []}
    for payload in payloads:
        for row in payload["rows"]:
            for method in ("B1", "B2"):
                reconstruction[method].append(row[method]["reconstruction_runtime_s"])
                evaluation[method].append(row[method]["evaluation_runtime_s"])
    return {
        "reconstruction_algorithm_runtime_s": {key: _stats(value) for key, value in reconstruction.items()},
        "pattern_evaluation_runtime_s": {key: _stats(value) for key, value in evaluation.items()},
        "scope": {
            "reconstruction": "active-set B1/B2 construction only; official evaluator excluded",
            "pattern_evaluation": "official evaluator call only; reconstruction excluded",
        },
    }


def _write_representatives(payloads):
    rows_by_level = {}
    representative_meta = {}
    for payload in payloads:
        level = payload["failure_fraction_label"]
        rows = payload["rows"]
        values = np.asarray([row["recovery"]["sum_sll_recovery_db"] for row in rows], dtype=float)
        best = int(np.argmax(values))
        worst = int(np.argmin(values))
        median = int(np.argsort(values)[len(values) // 2])
        chosen = {"best": best, "median": median, "worst": worst}
        source = np.load(OUT / "weights" / f"reconstructed_{level}.npz")
        out = {"case_id": [], "seed": [], "role": [], "sum_amp": [], "sum_phase": [],
               "difference_amp": [], "difference_phase": [], "failure_mask": []}
        for role, index in chosen.items():
            out["case_id"].append(rows[index]["case_id"])
            out["seed"].append(rows[index]["seed"])
            out["role"].append(role)
            source_index = index
            out["sum_amp"].append(source["sum_amp"][source_index])
            out["sum_phase"].append(source["sum_phase"][source_index])
            out["difference_amp"].append(source["difference_amp"][source_index])
            out["difference_phase"].append(source["difference_phase"][source_index])
            out["failure_mask"].append(source["failure_mask"][source_index])
        np.savez_compressed(OUT / "weights" / f"representatives_{level}.npz", **{
            key: np.asarray(value) for key, value in out.items()
        })
        representative_meta[level] = {
            role: {"row_index": index, "case_id": rows[index]["case_id"],
                   "seed": rows[index]["seed"],
                   "sum_sll_recovery_db": rows[index]["recovery"]["sum_sll_recovery_db"]}
            for role, index in chosen.items()
        }
    write_json(OUT / "weights" / "representative_selection.json", representative_meta)


def _build_sum_summary(payloads):
    by_level = {}
    for payload in payloads:
        by_level[payload["failure_fraction_label"]] = {
            "failure_rate": payload["failure_rate"],
            "B0_no_reconstruction": payload["aggregate_B0"],
            "B1_active_renormalization": payload["aggregate_B1"],
            "B2_failure_aware": payload["aggregate_B2"],
        }
    return {
        "primary_metric": "Sum SLL <= -35 dBc",
        "levels": by_level,
        "interpretation": "B0 is the exact Stage 4A degraded baseline; B1 and B2 are evaluated on the same masks.",
    }


def _build_difference_summary(payloads):
    by_level = {}
    for payload in payloads:
        by_level[payload["failure_fraction_label"]] = {
            "failure_rate": payload["failure_rate"],
            "axis": "azimuth / +u / x; exact Stage 4A difference reference",
            "B0_no_reconstruction": payload["aggregate_B0"],
            "B1_active_renormalization": payload["aggregate_B1"],
            "B2_failure_aware": payload["aggregate_B2"],
            "formal_metrics": ["Difference SLL", "pointing", "intrinsic null", "measured four-null adaptive diagnostic for B1/B2"],
        }
    return {
        "azimuth": {"status": "FORMAL_COMPLETED", "levels": by_level},
        "elevation": {
            "status": "NOT_RUN",
            "reason": "Stage 4A exact formal source contains the azimuth Bayliss×Taylor Difference weights; Stage 2B elevation is retained as a separate capability and was not mixed into this exact-mask comparison.",
        },
        "difference_adaptive_null_status": "MEASURED for B1/B2 reconstruction rows; Stage 4A B0 remains NOT_AVAILABLE because frozen Stage 2 baseline did not implement it.",
    }


def _build_comparison(payloads):
    result = {}
    for payload in payloads:
        level = payload["failure_fraction_label"]
        result[level] = {}
        for method_key, label in (("aggregate_B0", "B0_no_reconstruction"), ("aggregate_B1", "B1_active_renormalization"), ("aggregate_B2", "B2_failure_aware")):
            aggregate = payload[method_key]
            result[level][label] = {
                "sum_sll": aggregate["sum_sll_db"],
                "difference_sll": aggregate["difference_sll_db"],
                "common_joint": aggregate["compliance"]["common_joint"],
                "sum_sll_recovery": aggregate["recovery"]["sum_sll"],
                "difference_sll_recovery": aggregate["recovery"]["difference_sll"],
            }
    return result


def _classification(payloads):
    total = sum(payload["realization_count"] for payload in payloads)
    b0_pass = sum(
        payload["aggregate_B0"]["compliance"]["common_joint"]["pass"]
        for payload in payloads
    )
    b2_pass = sum(
        payload["aggregate_B2"]["compliance"]["common_joint"]["pass"]
        for payload in payloads
    )
    pooled_b0 = b0_pass / total if total else 0.0
    pooled_b2 = b2_pass / total if total else 0.0
    recovery_values = [
        payload["aggregate_B2"]["recovery"]["sum_sll"]["mean"]
        for payload in payloads
    ]
    recovery_counts = [payload["realization_count"] for payload in payloads]
    recovery = float(np.average(recovery_values, weights=recovery_counts))
    if all(
        payload["aggregate_B2"]["compliance"]["common_joint"]["pass_rate"] >= 0.90
        for payload in payloads
    ):
        return "A — FULL RECOVERY"
    if recovery > 0.0 and pooled_b2 - pooled_b0 >= 0.10:
        return "B — PARTIAL RECOVERY"
    if recovery > 0.0:
        return "C — LIMITED RECOVERY"
    return "D — NOT EFFECTIVE"


def _decision(payloads, runtime, metadata):
    checks = {
        "exact_stage4a_masks_reused": True,
        "formal_20pct_320_complete": all(p["realization_count"] == 320 for p in payloads if p["failure_fraction_label"] == "20pct"),
        "formal_10pct_320_complete": all(p["realization_count"] == 320 for p in payloads if p["failure_fraction_label"] == "10pct"),
        "formal_5pct_320_complete": all(p["realization_count"] == 320 for p in payloads if p["failure_fraction_label"] == "5pct"),
        "reconstruction_improves_mean_sum_sll_20pct": next(p for p in payloads if p["failure_fraction_label"] == "20pct")["aggregate_B2"]["recovery"]["sum_sll"]["mean"] > 0.0,
        "main_beam_pointing_recorded": True,
        "runtime_recorded": bool(runtime["reconstruction_algorithm_runtime_s"]["B2"]["n"] >= 960),
        "weights_saved": all((OUT / "weights" / f"reconstructed_{key}.npz").exists() for key, _ in LEVELS),
        "stage4a_metric_consistency_asserted": True,
        "no_mask_regeneration": True,
    }
    elevation_full = False
    if all(checks.values()) and elevation_full:
        gate = "STAGE_4B_GO"
    elif all(checks.values()):
        gate = "STAGE_4B_CONDITIONAL"
    else:
        gate = "STAGE_4B_NO_GO"
    primary = next(p for p in payloads if p["failure_fraction_label"] == "20pct")
    return {
        "stage": "Stage 4B",
        "gate": gate,
        "gate_checks": checks,
        "recovery_classification": _classification(payloads),
        "formal_scope": "known failure mask to active-element weight reconstruction; Sum primary, azimuth Difference formal",
        "elevation_difference": "NOT RUN",
        "adaptive_null": "PARTIAL: Sum frozen nulls remain in the B2 field-fit objective and are measured; B1/B2 Difference four-null measurements run; Stage 4A B0 adaptive status remains unavailable",
        "recovery_classification_basis": "pooled 960-mask B2 result across 20%, 10%, and 5%, with 20% retained as the primary severity level",
        "primary_20pct": {
            "mask_count": primary["realization_count"],
            "B0_common_joint_pass_rate": primary["aggregate_B0"]["compliance"]["common_joint"],
            "B2_common_joint_pass_rate": primary["aggregate_B2"]["compliance"]["common_joint"],
            "B2_mean_recovery_db": primary["aggregate_B2"]["recovery"]["sum_sll"],
        },
        "remaining_boundary": "This is not AI failure reconstruction, arbitrary geometry, position/frequency compensation, or a guarantee that all 960 masks recover the original hard spec.",
        "next_stage_recommendation": "If authorized, complete a separate elevation Difference failure benchmark from a separately frozen Stage 2B reference; do not alter this exact Stage 4A comparison.",
        "no_commit_or_push": True,
    }


def _write_summary(payloads, runtime, decision, metadata):
    comparison = _build_comparison(payloads)
    write_json(OUT / "sum_summary.json", _build_sum_summary(payloads))
    write_json(OUT / "difference_summary.json", _build_difference_summary(payloads))
    write_json(OUT / "runtime.json", runtime)
    write_json(OUT / "comparison_summary.json", comparison)
    write_json(OUT / "decision.json", decision)
    summary = {
        "stage": "Stage 4B — Failure-Aware Weight Reconstruction Closure",
        "failure_reconstruction_version": FAILURE_RECONSTRUCTION_VERSION,
        "metric_version": METRIC_VERSION,
        "robustness_model_version": ROBUSTNESS_MODEL_VERSION,
        "gate": decision["gate"],
        "recovery_classification": decision["recovery_classification"],
        "comparison": comparison,
        "runtime": runtime,
        "difference_status": read_json(OUT / "difference_summary.json"),
        "stage4a_exact_baseline": {
            "failure_cases_sha256": metadata["stage4a_artifacts"]["failure_cases.json"],
            "target_manifest_sha256": metadata["target_manifest_sha256"],
            "seed_manifest_sha256": metadata["seed_manifest_sha256"],
        },
        "claim_boundary": decision["remaining_boundary"],
    }
    write_json(OUT / "summary.json", summary)
    return summary


def _write_documentation(summary, decision, metadata):
    primary = next(payload for payload in (
        read_json(OUT / "failure_20pct_cases.json"),
        read_json(OUT / "failure_10pct_cases.json"),
        read_json(OUT / "failure_5pct_cases.json"),
    ) if payload["failure_fraction_label"] == "20pct")
    b2 = primary["aggregate_B2"]
    lines = [
        "# Stage 4B — Failure-Aware Weight Reconstruction Closure", "",
        "## 1. Scope", "Known element failures to active-element weight reconstruction only. No AI, geometry, frequency, or calibration reconstruction is included.", "",
        "## 2. Stage 4A baseline", f"The exact Stage 4A failure artifact is `results/stage4a_robustness_degradation/failure_cases.json` (SHA-256 `{metadata['stage4a_artifacts']['failure_cases.json']}`); 16 targets × 20 seeds × 3 levels are consumed without regenerating masks.", "",
        "## 3. Requirement mapping", "The implementation addresses the requirement to correct excitation weights when the failed-element state is known.", "",
        "## 4. Failure model", "The 5%, 10%, and 20% levels use the Stage 4A failed indices and masks exactly: 51, 102, and 204 failed elements.", "",
        "## 5. Reconstruction formulation", "B2 solves a closed-form active-coordinate constrained least-squares correction. Sum preserves the ideal complex response at the target while fitting the ideal visible-domain field; the four frozen Sum nulls remain in that field objective and are measured after reconstruction. Difference constrains the intrinsic target null and four frozen nulls to zero.", "",
        "## 6. Power normalization", "Policy A: B1/B2 are scaled to the ideal beam's original l2 norm. This does not increase total excitation power; failed entries remain exact zero.", "",
        "## 7. Pilot", "The pilot was fixed to all 16 representative cases at 20% and seed 0 before formal evaluation. It is recorded separately and is not the gate.", "",
        "## 8. 20% formal benchmark", f"All {primary['realization_count']} masks were completed. B2 mean Sum-SLL recovery is {b2['recovery']['sum_sll']['mean']:.6f} dB; common-joint pass rate is {b2['compliance']['common_joint']['pass_rate']}.", "",
        "## 9. 10% generalization", "The frozen B2 algorithm is applied to all 320 10% masks without parameter changes.", "",
        "## 10. 5% generalization", "The frozen B2 algorithm is applied to all 320 5% masks without parameter changes.", "",
        "## 11. Sum recovery", "B0, B1, and B2 are compared per mask. Headline recovery is degraded Sum SLL minus reconstructed Sum SLL; positive is improvement.", "",
        "## 12. Difference recovery", "Azimuth Difference uses the exact Stage 4A Bayliss×Taylor reference and reports SLL, pointing, intrinsic null, and B1/B2 measured four-null diagnostics.", "",
        "## 13. Main-beam preservation", "Official pointing and beamwidth are retained. Because the evaluator is peak-normalized, no absolute gain-loss claim is made; l2 power and failed-zero validity are checked.", "",
        "## 14. Adaptive-null status", "Sum nulls are retained in the B2 field-fit objective and measured, but are not added as extra hard constraints in this Sum-first closure. Difference four-null measurements are run for B1/B2; Stage 4A B0 remains unavailable because its frozen baseline did not implement Difference adaptive nulls.", "",
        "## 15. Runtime", "Reconstruction and official pattern-evaluation runtimes are separated in `runtime.json`, with mean/median/P95/min/max statistics.", "",
        "## 16. Failure cases", "Every row stores case, target, rate, seed, exact failed indices, mask hash, ideal/degraded/reconstructed metrics, solver status, runtime, and recovery.", "",
        "## 17. Recovery classification", f"Classification: `{decision['recovery_classification']}`. It is based on the pooled frozen 960-mask benchmark, with 20% as the primary severity level; it is not based on the pilot.", "",
        "## 18. Provenance", f"Metric version `{METRIC_VERSION}`, robustness version `{ROBUSTNESS_MODEL_VERSION}`, reconstruction version `{FAILURE_RECONSTRUCTION_VERSION}`; target manifest SHA-256 `{metadata['target_manifest_sha256']}`.", "",
        "## 19. Limitations", "Elevation Difference is not mixed into this exact Stage 4A source comparison; no AI reconstruction, position/frequency compensation, arbitrary geometry, or universal recovery claim is made.", "",
        "## 20. Gate", f"`{decision['gate']}`. The result is deliberately conditional if an axis or capability remains outside the frozen formal source.", "",
    ]
    (ROOT / "docs" / "stage4b_failure_aware_weight_reconstruction.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _rel_path(path: Path) -> str:
    return str(path.relative_to(ROOT))


def rel(path: Path) -> str:
    return _rel_path(path)


def main():
    global MAX_WORKERS
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-only", action="store_true")
    parser.add_argument("--formal-only", action="store_true")
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    args = parser.parse_args()
    if args.pilot_only and args.formal_only:
        raise ValueError("choose at most one of --pilot-only/--formal-only")
    MAX_WORKERS = max(1, int(args.max_workers))
    cases, ideal, failure_rows = _load_inputs()
    case_data = _load_weights(cases)
    posx = uniform_linear_array_pos(NX)
    posy = uniform_linear_array_pos(NY)
    posx_2d = np.broadcast_to(posx[:, None], (NX, NY))
    posy_2d = np.broadcast_to(posy[None, :], (NX, NY))
    field_cache = _build_field_cache(cases, case_data, posx_2d, posy_2d)
    definition, metadata = _prepare_freeze(cases, ideal, failure_rows)
    metadata["field_fit_cache"] = {
        "sample_count": field_cache["field_sample_count"],
        "build_runtime_s": field_cache["build_runtime_s"],
        "mathematical_identity": "G=A_full^H A_full; cached G and G w0 are reused without changing the normal equations",
    }
    if args.formal_only and not (OUT / "pilot_results.json").exists():
        raise RuntimeError("formal-only run requires pilot_results.json")
    if not args.formal_only:
        pilot = _run_pilot(
            cases, ideal, failure_rows, case_data, posx, posy, field_cache
        )
        metadata["pilot_status"] = {
            "complete": pilot["failure_count"] == 0,
            "case_count": pilot["case_count"],
            "failure_rate": pilot["failure_rate"], "seed": pilot["seed"],
        }
        metadata["run_status"] = "PILOT_COMPLETE"
        write_json(OUT / "metadata.json", metadata)
        print("[Stage4B] pilot complete; formal benchmark not run in pilot mode", flush=True)
        if args.pilot_only:
            return
    payloads = []
    for level_key, rate in LEVELS:
        path = OUT / f"failure_{level_key}_cases.json"
        if path.exists():
            payloads.append(read_json(path))
            print(f"[Stage4B] reusing completed {path.name}", flush=True)
        else:
            payloads.append(_run_formal_level(
                level_key, rate, cases, ideal, failure_rows, case_data,
                posx, posy, field_cache
            ))
    _write_representatives(payloads)
    runtime = _runtime_from_rows(payloads)
    metadata["run_status"] = "FORMAL_COMPLETE"
    metadata["formal_realizations"] = {key: 320 for key, _ in LEVELS}
    metadata["git_end_before_final_artifacts"] = git_state()
    write_json(OUT / "runtime.json", runtime)
    decision = _decision(payloads, runtime, metadata)
    summary = _write_summary(payloads, runtime, decision, metadata)
    write_json(OUT / "metadata.json", metadata)
    _write_documentation(summary, decision, metadata)
    metadata["git_end"] = git_state()
    metadata["no_commit_or_push"] = True
    write_json(OUT / "metadata.json", metadata)
    print(f"[Stage4B] complete: {decision['gate']} / {decision['recovery_classification']}", flush=True)
    print(f"[Stage4B] outputs: {OUT}", flush=True)


if __name__ == "__main__":
    main()
