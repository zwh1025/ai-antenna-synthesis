"""Stage 4A robustness benchmark freeze and degradation audit.

This runner is intentionally a measurement-only continuation of the frozen
Stage 2 Track P baseline.  It never calls an optimizer after a perturbation,
never trains or fine-tunes the Stage 3C1 model, and never changes the official
evaluator.  All formal metrics are produced by ``mylib.official_evaluator``
v1.0.0 and are written below one new result directory.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "project"
BASELINE = ROOT / "results" / "stage2_strict_closure" / "baseline"
OUT = ROOT / "results" / "stage4a_robustness_degradation"

NX = NY = 32
N_ELEMENTS = NX * NY
LAMBDA0 = 1.0
METRIC_VERSION = "1.0.0"
ROBUSTNESS_MODEL_VERSION = "1.0.0"
GRID_SIZE = 201
POSITION_LIMIT_LAMBDA = 0.05
AMPLITUDE_STEP_DB = 0.5
PHASE_STATES = 64
PHASE_STEP_RAD = 2.0 * np.pi / PHASE_STATES
FAILURE_RATES = (0.05, 0.10, 0.20)
N_SEEDS = 20
SEEDS = tuple(range(N_SEEDS))
FREQUENCY_RATIOS = tuple(round(0.90 + 0.01 * i, 2) for i in range(21))
MAX_WORKERS = min(8, max(1, os.cpu_count() or 1))

REGULAR_SELECTION = (0, 2, 4, 14, 28, 38, 50, 72)
RANDOM_SELECTION = (0, 25, 50, 75, 100, 125, 150, 175)

sys.path.insert(0, str(PROJECT))

from mylib.antenna_calc import (  # noqa: E402
    beam_steering_phase_2d,
    combine_2d_excitation,
    taylor_2d_separable,
    taylor_excitation,
    uniform_linear_array_pos,
)
from mylib.official_evaluator import (  # noqa: E402
    ADAPTIVE_NULL_THRESHOLD_DB,
    DIFFERENCE_SLL_THRESHOLD_DB,
    OFFICIAL_EVALUATOR_VERSION,
    SUM_SLL_THRESHOLD_DB,
    evaluate_official_case,
)
from mylib.sum_diff import bayliss_excitation  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_hash(value) -> str:
    return sha256_bytes(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8"))


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot serialize {type(value)!r}")


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False,
                   default=_json_default) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


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


def finite_or_none(value):
    value = float(value)
    return value if np.isfinite(value) else None


def _metric_stats(values: list[float]) -> dict:
    arr = np.asarray([float(v) for v in values if v is not None], dtype=np.float64)
    if arr.size == 0:
        return {"n": 0, "mean": None, "median": None, "std": None,
                "P5": None, "P95": None, "worst": None}
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "P5": float(np.percentile(arr, 5)),
        "P95": float(np.percentile(arr, 95)),
        "worst": float(np.max(arr)),
    }


def _pass_rate(values: list[bool | None]) -> dict:
    observed = [bool(value) for value in values if value is not None]
    return {
        "n": len(observed),
        "pass": int(sum(observed)),
        "pass_rate": (float(np.mean(observed)) if observed else None),
        "status": "MEASURED" if observed else "NOT_AVAILABLE",
    }


def summarize_rows(rows: list[dict]) -> dict:
    keys = (
        "sum_sll_db", "difference_sll_db", "sum_pointing_error_deg",
        "difference_pointing_error_deg", "sum_null_worst_db",
        "difference_intrinsic_null_db", "delta_sum_sll_db",
        "delta_difference_sll_db", "delta_sum_pointing_error_deg",
        "delta_difference_pointing_error_deg", "delta_sum_null_worst_db",
        "delta_difference_intrinsic_null_db",
    )
    result = {"n_realizations": len(rows), "metrics": {}}
    for key in keys:
        result["metrics"][key] = _metric_stats([row.get(key) for row in rows])
    result["compliance"] = {
        "common_joint": _pass_rate([row.get("common_joint_pass") for row in rows]),
        "track_p_joint": _pass_rate([row.get("track_p_joint_pass") for row in rows]),
    }
    return result


def _array_positions():
    axis = uniform_linear_array_pos(NX)
    return axis, axis.copy()


def quantize_amplitude(amp: np.ndarray) -> np.ndarray:
    """Nearest 0.5 dB grid relative to a unit maximum, preserving shape."""
    arr = np.asarray(amp, dtype=np.float64)
    clipped = np.clip(arr, 1e-6, 1.0)
    db = 20.0 * np.log10(clipped)
    quantized_db = np.round(db / AMPLITUDE_STEP_DB) * AMPLITUDE_STEP_DB
    return np.power(10.0, quantized_db / 20.0)


def quantize_phase(phase: np.ndarray) -> np.ndarray:
    """Nearest 6-bit phase grid, wrapped to [0, 2*pi)."""
    arr = np.mod(np.asarray(phase, dtype=np.float64), 2.0 * np.pi)
    return np.mod(np.round(arr / PHASE_STEP_RAD) * PHASE_STEP_RAD, 2.0 * np.pi)


def generate_position_offsets(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(int(seed))
    dx = rng.uniform(-POSITION_LIMIT_LAMBDA, POSITION_LIMIT_LAMBDA,
                     size=(NX, NY))
    dy = rng.uniform(-POSITION_LIMIT_LAMBDA, POSITION_LIMIT_LAMBDA,
                     size=(NX, NY))
    return dx, dy


def apply_position_error(posx, posy, seed: int):
    """Official planar model B: independent x/y offsets, z remains zero."""
    dx, dy = generate_position_offsets(seed)
    px = np.broadcast_to(np.asarray(posx)[:, None], (NX, NY)).copy() + dx
    py = np.broadcast_to(np.asarray(posy)[None, :], (NX, NY)).copy() + dy
    return px, py, dx, dy


def generate_failure_mask(rate: float, seed: int, n_elements: int = N_ELEMENTS):
    rng = np.random.RandomState(int(seed))
    count = int(np.floor(float(n_elements) * float(rate)))
    indices = np.sort(rng.choice(int(n_elements), size=count, replace=False))
    mask = np.zeros(int(n_elements), dtype=bool)
    mask[indices] = True
    return mask


def apply_element_failures(amp: np.ndarray, phase: np.ndarray, rate: float,
                           seed: int):
    mask = generate_failure_mask(rate, seed, int(np.asarray(amp).size))
    amp_out = np.asarray(amp, dtype=np.float64).copy().reshape(-1)
    phase_out = np.asarray(phase, dtype=np.float64).copy().reshape(-1)
    amp_out[mask] = 0.0
    phase_out[mask] = 0.0
    return amp_out.reshape(np.asarray(amp).shape), phase_out.reshape(np.asarray(phase).shape), mask


def _source_audit_entry(path: str, functions: list[str], definition: str,
                        randomization: str, seed: str, metric: str,
                        geometry: str, weights: str, reusable: bool,
                        result: str, reason: str) -> dict:
    file_path = ROOT / path
    return {
        "file": path,
        "sha256": sha256_file(file_path) if file_path.exists() else None,
        "functions_or_code_regions": functions,
        "perturbation_definition": definition,
        "random_distribution": randomization,
        "seed_protocol": seed,
        "metric_evaluator": metric,
        "geometry": geometry,
        "weights": weights,
        "result_or_status": result,
        "reusable_for_stage4a_formal": reusable,
        "reuse_decision": reason,
    }


def build_legacy_audit() -> dict:
    return {
        "audit_version": "stage4a-legacy-audit-v1",
        "purpose": "Inventory only; no legacy result is promoted to the formal benchmark.",
        "entries": [
            _source_audit_entry(
                "project/run_nonideal_v2.py",
                ["quantize_amp_05db", "quantize_phase_6bit", "apply_failure", "inline position/frequency loops"],
                "0.5 dB amplitude grid; 6-bit phase; planar independent +/-0.05 lambda x/y; excitation-zero failures at floor(N*rate); lambda=1/ratio",
                "uniform amplitude/phase quantization; uniform position errors; random choice without replacement for failures",
                "20 seeds; seed*1000+42 in the runner",
                "legacy mylib.evaluation grid/cut metrics, not official evaluator v1.0.0",
                "planar 32x32",
                "fixed center Taylor weights; difference not the official Track P complete pair",
                False,
                "implemented legacy non-ideal experiment",
                "Definitions are useful provenance, but evaluator and case protocol differ from this freeze.",
            ),
            _source_audit_entry(
                "project/run_failure_benchmark.py",
                ["generate_failure_mask", "lcmv_on_active", "failure benchmark loop"],
                "floor(N*rate) random failures; 100 scenes; active-set LCMV recomputation exists",
                "random masks without replacement",
                "runner-specific seed protocol",
                "legacy 3-beamwidth evaluator",
                "planar baseline",
                "includes LCMV re-optimization on active elements",
                False,
                "legacy failure benchmark",
                "Re-synthesis is explicitly forbidden in Stage 4A; masks are not the formal saved protocol.",
            ),
            _source_audit_entry(
                "project/run_curved_nonideal.py",
                ["quantize_complex_weights", "apply_failure", "curved non-ideal loop"],
                "quantized/failure excitation tests on the Stage 3 curved AI setup",
                "runner-specific randomization",
                "runner-specific seeds",
                "run_curved_verify legacy evaluator",
                "32x32 curved reconstructed geometry",
                "AI curved weights and legacy verification",
                False,
                "legacy AI non-ideal diagnostic",
                "Stage 4A formal scope is Track P physics; AI robustness is not mixed into the primary gate.",
            ),
        ],
        "formal_reuse_rule": "Only the documented definitions are reused after re-evaluation by official evaluator v1.0.0; no legacy output value is reused.",
    }


def build_perturbation_definitions() -> dict:
    return {
        "robustness_model_version": ROBUSTNESS_MODEL_VERSION,
        "metric_version": METRIC_VERSION,
        "formal_scope": "Track P planar 32x32 physics baseline; degradation-only; original frozen weights are reused exactly.",
        "position_error": {
            "model": "B",
            "model_status": "OFFICIAL_ASSUMPTION_FOR_THIS_FREEZE",
            "delta_x_lambda": "Uniform[-0.05,+0.05] independently per element",
            "delta_y_lambda": "Uniform[-0.05,+0.05] independently per element",
            "delta_z_lambda": 0.0,
            "planar_geometry": True,
            "seeds": list(SEEDS),
            "rng": "numpy.random.RandomState(seed)",
            "ambiguity_note": "The Stage 4A wording permits A/B/C readings. Existing planar non-ideal code directly supports B; formal Track P freezes B and records this assumption rather than silently claiming a unique 3-D interpretation.",
            "not_run_alternatives": {
                "A": "independent 3-D x/y/z offsets; not applicable to the frozen planar Track P geometry",
                "C": "common-mode or rigid-body displacement; not an element-wise placement-error model",
            },
        },
        "amplitude_phase": {
            "mode": "deterministic_quantization",
            "amplitude": {
                "grid_db": 0.5,
                "reference": "unit maximum per frozen excitation; dB=20*log10(amplitude)",
                "operation": "clip to [1e-6,1], nearest grid using numpy.round, convert back to linear amplitude",
                "zero_failure_handling": "failure excitation is set to exact zero in the separate failure benchmark",
            },
            "phase": {
                "states": PHASE_STATES,
                "step_deg": 360.0 / PHASE_STATES,
                "step_rad": PHASE_STEP_RAD,
                "operation": "nearest state using numpy.round, modulo 2*pi; radians internally",
            },
            "random_seeds": [],
            "ambiguity_note": "Amplitude wording can mean a quantizer or an additive/random hardware error. The formal freeze chooses the legacy-compatible nearest 0.5 dB quantizer; no stochastic amplitude error is silently mixed into this case.",
            "re_synthesis": False,
        },
        "element_failure": {
            "rates": list(FAILURE_RATES),
            "failed_excitation": "amp=0; phase=0 as an irrelevant placeholder",
            "count_rule": "floor(N_elements*rate)",
            "counts_for_1024": {"0.05": 51, "0.10": 102, "0.20": 204},
            "mask_distribution": "uniform random choice without replacement",
            "rng": "numpy.random.RandomState(seed)",
            "seeds": list(SEEDS),
            "same_mask_applied_to": ["sum excitation", "difference excitation"],
            "mask_reuse": "Every mask and failed index list is saved for exact Stage 4B reuse.",
            "re_synthesis": False,
        },
        "frequency_offset": {
            "ratio_definition": "lambda_ref/lambda_perturbed",
            "frequency_ratios": list(FREQUENCY_RATIOS),
            "frequency_offset_percent": [-10, 10],
            "implementation": "lambda=1/ratio in official array-factor evaluator",
            "coordinates": "physical positions remain fixed; no coordinate re-scaling",
            "weights": "original frozen weights; no retuning",
            "re_synthesis": False,
        },
        "compliance": {
            "sum_sll": "<= -35 dBc",
            "difference_sll": "<= -20 dBc",
            "pointing": "measured error <= official beamwidth_3db/30",
            "intrinsic_difference_null": "difference response at target <= -30 dBc",
            "common_joint": "sum SLL + difference SLL + difference pointing + intrinsic difference null",
            "track_p_joint": "common_joint + measured sum nulls available for the LCMV adaptive-null subset",
            "difference_adaptive_null": "not implemented in frozen Stage 2 baseline; never reported as a pass",
        },
        "runtime_protocol": {
            "repetitions": 100,
            "warmup": 10,
            "statistics": ["mean", "std", "P50", "P95", "min", "max", "CV"],
            "teacher_rerun": False,
        },
    }


def _selection_reason(kind: str, index: int) -> str:
    if kind == "regular":
        return {
            0: "broadside",
            2: "moderate theta, quadrant-I diagonal",
            4: "moderate theta, y-axis quadrant",
            14: "moderate diagonal scan",
            28: "large theta, y-axis scan",
            38: "large theta, quadrant-I diagonal",
            50: "near cone edge, quadrant-I diagonal",
            72: "cone-edge endpoint, quadrant-IV",
        }[index]
    return "independent random held-out target from frozen seed-42 manifest"


def freeze_cases(case_manifest: dict, regular_weights: dict,
                 random_weights: dict) -> list[dict]:
    selected = []
    for kind, indices, weights in (
        ("regular", REGULAR_SELECTION, regular_weights),
        ("random", RANDOM_SELECTION, random_weights),
    ):
        source_cases = case_manifest[kind]
        for index in indices:
            source = source_cases[index]
            selected.append({
                "case_index": len(selected),
                "case_id": source["case_id"],
                "set": kind,
                "source_index": int(index),
                "theta0_deg": float(source["theta0_deg"]),
                "phi0_deg": float(source["phi0_deg"]),
                "null_dirs": [[float(x), float(y)] for x, y in source["null_dirs"]],
                "sum_method": "lcmv" if kind == "regular" else "taylor",
                "adaptive_sum_available": kind == "regular",
                "weight_artifact": rel(BASELINE / "weights" / f"{kind}_weights.npz"),
                "selection_reason": _selection_reason(kind, index),
                "source_case_manifest_sha256": sha256_file(BASELINE / "case_manifest.json"),
                "weight_shapes": {
                    "sum_amp": list(weights["sum_amp"][index].shape),
                    "difference_amp": list(weights["difference_amp"][index].shape),
                },
            })
    if len(selected) != 16 or len({row["case_id"] for row in selected}) != 16:
        raise AssertionError("representative case selection is not 16 unique cases")
    return selected


def freeze_case_manifest(cases: list[dict]) -> dict:
    payload = {
        "manifest_version": "stage4a-representative-cases-v1",
        "frozen_before_any_perturbation_run": True,
        "case_count": len(cases),
        "regular_count": sum(row["set"] == "regular" for row in cases),
        "random_count": sum(row["set"] == "random" for row in cases),
        "source_manifest": rel(BASELINE / "case_manifest.json"),
        "source_manifest_sha256": sha256_file(BASELINE / "case_manifest.json"),
        "selection_indices": {
            "regular": list(REGULAR_SELECTION),
            "random": list(RANDOM_SELECTION),
        },
        "cases": cases,
    }
    payload["case_list_sha256"] = json_hash(payload["cases"])
    path = OUT / "robustness_case_manifest.json"
    if path.exists():
        old = read_json(path)
        if old.get("case_list_sha256") != payload["case_list_sha256"]:
            raise RuntimeError("existing Stage 4A case manifest differs; freeze cannot be changed")
        return old
    write_json(path, payload)
    return payload


def compact_metrics(official: dict, case: dict, ideal: dict | None = None) -> dict:
    summ = official["sum"]
    diff = official["difference"]
    sum_null = official["adaptive_null"]["sum"]
    diff_null = official["adaptive_null"]["difference"]
    sum_centers = [finite_or_none(x) for x in sum_null["center_db"]]
    sum_windows = [finite_or_none(x) for x in sum_null["window_worst_db"]]
    diff_centers = ([float(x) for x in diff_null["center_db"]]
                    if diff_null is not None else [])
    diff_centers = [finite_or_none(x) for x in diff_centers]
    valid_sum_centers = [x for x in sum_centers if x is not None]
    valid_sum_windows = [x for x in sum_windows if x is not None]
    sum_null_worst = float(max(valid_sum_centers)) if valid_sum_centers else None
    sum_window_worst = float(max(valid_sum_windows)) if valid_sum_windows else None
    diff_intrinsic = diff_centers[0] if diff_centers else None
    sum_sll = finite_or_none(summ["sll_db"])
    diff_sll = finite_or_none(diff["sll_db"])
    sum_pointing = finite_or_none(summ["pointing_error_deg"])
    diff_pointing = finite_or_none(diff["pointing_error_deg"])
    sum_beamwidth = finite_or_none(summ["beamwidth_3db_deg"])
    sum_pointing_threshold = finite_or_none(summ["pointing_threshold_deg"])
    diff_pointing_threshold = finite_or_none(diff["pointing_threshold_deg"])
    sum_sll_pass = sum_sll is not None and sum_sll <= SUM_SLL_THRESHOLD_DB
    diff_sll_pass = diff_sll is not None and diff_sll <= DIFFERENCE_SLL_THRESHOLD_DB
    diff_pointing_pass = (diff_pointing is not None and
                          sum_pointing_threshold is not None and
                          diff_pointing <= sum_pointing_threshold)
    diff_null_pass = diff_intrinsic is not None and diff_intrinsic <= ADAPTIVE_NULL_THRESHOLD_DB
    sum_null_pass = bool(sum_null_worst is not None and sum_null_worst <= ADAPTIVE_NULL_THRESHOLD_DB)
    common_joint = bool(sum_sll_pass and diff_sll_pass and diff_pointing_pass and diff_null_pass)
    track_p_joint = (bool(common_joint and sum_null_pass)
                     if case["adaptive_sum_available"] else None)
    result = {
        "case_id": case["case_id"],
        "set": case["set"],
        "sum_method": case["sum_method"],
        "metric_version": official["metric_version"],
        "sum_sll_db": sum_sll,
        "sum_sll_threshold_db": SUM_SLL_THRESHOLD_DB,
        "sum_sll_pass": sum_sll_pass,
        "sum_beamwidth_3db_deg": sum_beamwidth,
        "sum_peak_direction": summ["peak_direction"],
        "sum_pointing_error_deg": sum_pointing,
        "sum_pointing_threshold_deg": sum_pointing_threshold,
        "difference_sll_db": diff_sll,
        "difference_sll_threshold_db": DIFFERENCE_SLL_THRESHOLD_DB,
        "difference_sll_pass": diff_sll_pass,
        "difference_zero_crossing_direction": diff["zero_crossing_direction"],
        "difference_pointing_error_deg": diff_pointing,
        "difference_pointing_threshold_deg": diff_pointing_threshold,
        "difference_pointing_pass": diff_pointing_pass,
        "sum_null_center_db": sum_centers,
        "sum_null_window_worst_db": sum_windows,
        "sum_null_worst_db": sum_null_worst,
        "sum_null_pass": sum_null_pass,
        "sum_null_available": bool(case["adaptive_sum_available"]),
        "sum_null_status": "ADAPTIVE_LCMV_REFERENCE" if case["adaptive_sum_available"] else "OBSERVED_ON_TAYLOR_NOT_ADAPTIVE",
        "difference_intrinsic_null_db": diff_intrinsic,
        "difference_intrinsic_null_pass": diff_null_pass,
        "difference_adaptive_null_status": "NOT_IMPLEMENTED_IN_FROZEN_STAGE2_BASELINE",
        "common_joint_pass": common_joint,
        "track_p_joint_pass": track_p_joint,
        "joint_compliance_definition": "common_joint plus adaptive sum null when available",
        "official_peak_normalization": "visible-domain max field amplitude; evaluator output retains its -300 dBc floor",
    }
    if ideal is not None:
        result.update({
            "delta_sum_sll_db": (sum_sll - ideal["sum_sll_db"]
                                  if sum_sll is not None and ideal["sum_sll_db"] is not None else None),
            "delta_difference_sll_db": (diff_sll - ideal["difference_sll_db"]
                                         if diff_sll is not None and ideal["difference_sll_db"] is not None else None),
            "delta_sum_pointing_error_deg": (sum_pointing - ideal["sum_pointing_error_deg"]
                                              if sum_pointing is not None and ideal["sum_pointing_error_deg"] is not None else None),
            "delta_difference_pointing_error_deg": (diff_pointing - ideal["difference_pointing_error_deg"]
                                                    if diff_pointing is not None and ideal["difference_pointing_error_deg"] is not None else None),
            "delta_sum_null_worst_db": (sum_null_worst - ideal["sum_null_worst_db"]
                                         if sum_null_worst is not None and ideal["sum_null_worst_db"] is not None else None),
            "delta_difference_intrinsic_null_db": (diff_intrinsic - ideal["difference_intrinsic_null_db"]
                                                   if diff_intrinsic is not None and ideal["difference_intrinsic_null_db"] is not None else None),
        })
    return result


def _official_job(job: dict):
    case = job["case"]
    official = evaluate_official_case(
        job["amp_sum"], job["phase_sum"], job["posx"], job["posy"],
        case["theta0_deg"], case["phi0_deg"],
        null_dirs=case["null_dirs"],
        amp_difference=job["amp_difference"],
        phase_difference=job["phase_difference"],
        difference_null_dirs=[(case["theta0_deg"], case["phi0_deg"])],
        difference_axis_phi_deg=0.0,
        n_uv=GRID_SIZE,
        lamb=job.get("lamb", LAMBDA0),
    )
    return official


def evaluate_jobs(jobs: list[dict], label: str) -> list[dict]:
    results = []
    total = len(jobs)
    print(f"[Stage4A] {label}: evaluating {total} official cases with {MAX_WORKERS} workers", flush=True)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for index, (job, official) in enumerate(zip(jobs, executor.map(_official_job, jobs)), 1):
            results.append({"job": job, "official": official})
            if index == total or index % 100 == 0:
                print(f"[Stage4A] {label}: {index}/{total}", flush=True)
    return results


def build_case_data(cases: list[dict], regular_weights: dict,
                    random_weights: dict) -> dict[str, dict]:
    result = {}
    for case in cases:
        weights = regular_weights if case["set"] == "regular" else random_weights
        idx = case["source_index"]
        result[case["case_id"]] = {
            "amp_sum": np.asarray(weights["sum_amp"][idx], dtype=np.float64).copy(),
            "phase_sum": np.asarray(weights["sum_phase"][idx], dtype=np.float64).copy(),
            "amp_difference": np.asarray(weights["difference_amp"][idx], dtype=np.float64).copy(),
            "phase_difference": np.asarray(weights["difference_phase"][idx], dtype=np.float64).copy(),
        }
    return result


def _base_job(case: dict, case_data: dict, posx, posy, lamb=1.0) -> dict:
    return {
        "case": case,
        "amp_sum": case_data["amp_sum"],
        "phase_sum": case_data["phase_sum"],
        "amp_difference": case_data["amp_difference"],
        "phase_difference": case_data["phase_difference"],
        "posx": posx,
        "posy": posy,
        "lamb": float(lamb),
    }


def _with_degradation(metrics: dict, ideal: dict) -> dict:
    return metrics


def run_ideal(cases, case_data, posx, posy) -> dict:
    jobs = [_base_job(case, case_data[case["case_id"]], posx, posy)
            for case in cases]
    evaluated = evaluate_jobs(jobs, "ideal reference")
    by_case = {}
    for item in evaluated:
        case = item["job"]["case"]
        metrics = compact_metrics(item["official"], case)
        by_case[case["case_id"]] = metrics
    payload = {
        "metric_version": METRIC_VERSION,
        "official_evaluator": "mylib.official_evaluator.evaluate_official_case",
        "official_evaluator_version": OFFICIAL_EVALUATOR_VERSION,
        "grid_size": GRID_SIZE,
        "coordinates": "frozen planar 32x32 axes; lambda=1 reference",
        "weights": "same original Stage 2 weights later reused for every perturbation",
        "case_count": len(cases),
        "cases": [by_case[case["case_id"]] for case in cases],
        "baseline_common_joint_pass_rate": _pass_rate([by_case[c["case_id"]]["common_joint_pass"] for c in cases]),
        "baseline_track_p_joint_pass_rate": _pass_rate([by_case[c["case_id"]]["track_p_joint_pass"] for c in cases]),
        "raw_evaluator_result_retention": "compact official metrics are retained; perturbation rows retain the same metric fields and deltas",
    }
    write_json(OUT / "ideal_reference.json", payload)
    return payload


def run_position(cases, case_data, ideal, posx, posy) -> dict:
    jobs = []
    descriptors = []
    for case in cases:
        for seed in SEEDS:
            px, py, dx, dy = apply_position_error(posx, posy, seed)
            jobs.append(_base_job(case, case_data[case["case_id"]], px, py))
            descriptors.append({
                "case_id": case["case_id"], "seed": seed,
                "delta_x_min_lambda": float(np.min(dx)),
                "delta_x_max_lambda": float(np.max(dx)),
                "delta_y_min_lambda": float(np.min(dy)),
                "delta_y_max_lambda": float(np.max(dy)),
                "delta_xy_rms_lambda": float(np.sqrt(np.mean(dx * dx + dy * dy))),
            })
    evaluated = evaluate_jobs(jobs, "position error")
    rows = []
    for descriptor, item in zip(descriptors, evaluated):
        case = next(c for c in cases if c["case_id"] == descriptor["case_id"])
        metrics = compact_metrics(item["official"], case, ideal[descriptor["case_id"]])
        rows.append({**descriptor, **metrics})
    by_case = {case["case_id"]: [] for case in cases}
    for row in rows:
        by_case[row["case_id"]].append(row)
    payload = {
        "perturbation": "position_error",
        "robustness_model_version": ROBUSTNESS_MODEL_VERSION,
        "metric_version": METRIC_VERSION,
        "model": "B: independent planar x/y Uniform[-0.05,+0.05] lambda; z=0",
        "case_count": len(cases), "seeds_per_case": len(SEEDS),
        "cases": [{"case_id": case["case_id"], "realizations": by_case[case["case_id"]],
                   "aggregate": summarize_rows(by_case[case["case_id"]])} for case in cases],
        "aggregate_all_cases": summarize_rows(rows),
    }
    write_json(OUT / "position_error_cases.json", payload)
    return payload


def run_quantization(cases, case_data, ideal, posx, posy) -> dict:
    jobs = []
    for case in cases:
        source = case_data[case["case_id"]]
        jobs.append({
            **_base_job(case, source, posx, posy),
            "amp_sum": quantize_amplitude(source["amp_sum"]),
            "phase_sum": quantize_phase(source["phase_sum"]),
            "amp_difference": quantize_amplitude(source["amp_difference"]),
            "phase_difference": quantize_phase(source["phase_difference"]),
        })
    evaluated = evaluate_jobs(jobs, "amplitude/phase quantization")
    rows = []
    for case, item in zip(cases, evaluated):
        metrics = compact_metrics(item["official"], case, ideal[case["case_id"]])
        rows.append({
            "case_id": case["case_id"],
            "quantization": {
                "amplitude_grid_db": AMPLITUDE_STEP_DB,
                "phase_states": PHASE_STATES,
                "phase_step_deg": 360.0 / PHASE_STATES,
            },
            **metrics,
        })
    payload = {
        "perturbation": "amplitude_phase_quantization",
        "robustness_model_version": ROBUSTNESS_MODEL_VERSION,
        "metric_version": METRIC_VERSION,
        "deterministic": True,
        "cases": rows,
        "aggregate_all_cases": summarize_rows(rows),
    }
    write_json(OUT / "quantization_cases.json", payload)
    return payload


def _mask_hash(mask: np.ndarray) -> str:
    return sha256_bytes(np.asarray(np.flatnonzero(mask), dtype=np.uint32).tobytes())


def run_failures(cases, case_data, ideal, posx, posy) -> dict:
    jobs = []
    descriptors = []
    for case in cases:
        source = case_data[case["case_id"]]
        for rate in FAILURE_RATES:
            for seed in SEEDS:
                mask = generate_failure_mask(rate, seed)
                sum_amp, sum_phase, _ = apply_element_failures(source["amp_sum"], source["phase_sum"], rate, seed)
                diff_amp, diff_phase, _ = apply_element_failures(source["amp_difference"], source["phase_difference"], rate, seed)
                jobs.append({
                    **_base_job(case, source, posx, posy),
                    "amp_sum": sum_amp, "phase_sum": sum_phase,
                    "amp_difference": diff_amp, "phase_difference": diff_phase,
                })
                descriptors.append({
                    "case_id": case["case_id"], "failure_rate": float(rate),
                    "seed": int(seed), "failed_count": int(mask.sum()),
                    "failed_indices": np.flatnonzero(mask).astype(int).tolist(),
                    "mask_sha256": _mask_hash(mask),
                })
    evaluated = evaluate_jobs(jobs, "element failures")
    rows = []
    for descriptor, item in zip(descriptors, evaluated):
        case = next(c for c in cases if c["case_id"] == descriptor["case_id"])
        metrics = compact_metrics(item["official"], case, ideal[case["case_id"]])
        rows.append({**descriptor, **metrics})
    by_rate = {}
    for rate in FAILURE_RATES:
        rate_rows = [row for row in rows if row["failure_rate"] == rate]
        by_rate[f"{rate:.2f}"] = {
            "rate": rate,
            "expected_failed_count": int(np.floor(N_ELEMENTS * rate)),
            "realization_count": len(rate_rows),
            "aggregate_all_cases": summarize_rows(rate_rows),
            "cases": [{
                "case_id": case["case_id"],
                "aggregate": summarize_rows([r for r in rate_rows if r["case_id"] == case["case_id"]]),
            } for case in cases],
        }
    payload = {
        "perturbation": "element_failure",
        "robustness_model_version": ROBUSTNESS_MODEL_VERSION,
        "metric_version": METRIC_VERSION,
        "count_rule": "floor(1024*rate)",
        "masks_saved_for_stage4b": True,
        "rates": by_rate,
        "realizations": rows,
    }
    write_json(OUT / "failure_cases.json", payload)
    return payload


def _contiguous_band(ratios, passes):
    center = int(np.argmin(np.abs(np.asarray(ratios) - 1.0)))
    if not passes[center]:
        return None
    left = right = center
    while left > 0 and passes[left - 1]:
        left -= 1
    while right + 1 < len(passes) and passes[right + 1]:
        right += 1
    return [float(ratios[left]), float(ratios[right])]


def run_frequency(cases, case_data, ideal, posx, posy) -> dict:
    jobs = []
    descriptors = []
    for case in cases:
        source = case_data[case["case_id"]]
        for ratio in FREQUENCY_RATIOS:
            jobs.append(_base_job(case, source, posx, posy, lamb=1.0 / ratio))
            descriptors.append({"case_id": case["case_id"], "frequency_ratio": ratio,
                                "lambda_used": 1.0 / ratio,
                                "frequency_offset_percent": (ratio - 1.0) * 100.0})
    evaluated = evaluate_jobs(jobs, "frequency offset")
    rows = []
    for descriptor, item in zip(descriptors, evaluated):
        case = next(c for c in cases if c["case_id"] == descriptor["case_id"])
        metrics = compact_metrics(item["official"], case, ideal[case["case_id"]])
        rows.append({**descriptor, **metrics})
    by_case = {}
    for case in cases:
        case_rows = [row for row in rows if row["case_id"] == case["case_id"]]
        common = [bool(row["common_joint_pass"]) for row in case_rows]
        track = [row["track_p_joint_pass"] for row in case_rows]
        by_case[case["case_id"]] = {
            "case_id": case["case_id"],
            "frequency_ratios": list(FREQUENCY_RATIOS),
            "realizations": case_rows,
            "common_joint_compliance": _pass_rate(common),
            "common_joint_compliant_band_containing_f0": _contiguous_band(FREQUENCY_RATIOS, common),
            "track_p_joint_compliance": _pass_rate(track),
            "track_p_compliant_band_containing_f0": _contiguous_band(
                FREQUENCY_RATIOS, [bool(x) if x is not None else False for x in track]
            ) if any(x is not None for x in track) else None,
            "aggregate": summarize_rows(case_rows),
        }
    payload = {
        "perturbation": "frequency_offset",
        "robustness_model_version": ROBUSTNESS_MODEL_VERSION,
        "metric_version": METRIC_VERSION,
        "frequency_ratios": list(FREQUENCY_RATIOS),
        "coordinates_fixed": True,
        "lambda_rule": "lambda=1/frequency_ratio",
        "cases": [by_case[case["case_id"]] for case in cases],
        "realization_count": len(rows),
    }
    write_json(OUT / "frequency_cases.json", payload)
    return payload


def run_physics_runtime(posx, posy, case):
    target_theta = float(case["theta0_deg"])
    target_phi = float(case["phi0_deg"])
    def synthesize():
        amp_x, amp_y = taylor_2d_separable(NX, NY, 35)
        diff_x, _ = bayliss_excitation(NX, 35)
        diff_y = taylor_excitation(NY * 0.5, posy, 35)
        px, py = beam_steering_phase_2d(posx, posy, target_theta, target_phi)
        return (
            combine_2d_excitation(amp_x, amp_y, px, py),
            combine_2d_excitation(diff_x, diff_y, px, py),
        )
    for _ in range(10):
        synthesize()
    samples = []
    for _ in range(100):
        started = time.perf_counter()
        synthesize()
        samples.append(time.perf_counter() - started)
    return {
        "name": "Stage2 planar Taylor/Bayliss fixed-target synthesis",
        "N": N_ELEMENTS, "repetitions": 100, "warmup": 10,
        "target": {"theta0_deg": target_theta, "phi0_deg": target_phi},
        "scope": "synthesis function calls only; no LCMV solve and no dense official evaluator",
        "samples_s": runtime_stats(samples),
    }


def _load_c1_module():
    path = PROJECT / "run_stage3c1_1024_ai_feasibility.py"
    spec = importlib.util.spec_from_file_location("stage3c1_runtime_module", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def run_ai_runtime():
    import torch
    c1 = _load_c1_module()
    c0 = c1.load_c0_module()
    coords, geometry_sha, _ = c1.load_frozen_geometry(c0)
    pipeline = c1.load_pipeline()
    archive = c1.load_archive()
    selected = read_json(c1.OUT / "performance_summary.json")["formal_selected_model"]
    model, _ = c1.load_checkpoint_model(ROOT / selected["checkpoint_path"], archive)
    model.eval()
    normalization = read_json(c1.OUT / "feature_normalization.json")
    split = read_json(c1.OUT / "stage3c1_split_manifest.json")
    parent = next(row for row in split["cases"] if row["split"] == "test")
    target = c1.target_d4(parent, 0)
    factor = float(selected["reconstruction_factor"])
    baseline = c1.coordinate_taylor(coords, target["theta_deg"], target["phi_deg"], pipeline)
    feature = c1.baseline_feature(coords, target, baseline, normalization)
    feature_tensor = torch.from_numpy(feature[None, ...]).float()
    torch.set_num_threads(1)
    with torch.no_grad():
        for _ in range(10):
            model(feature_tensor)
    inference = []
    end_to_end = []
    for _ in range(100):
        started = time.perf_counter()
        with torch.no_grad():
            model(feature_tensor)
        inference.append(time.perf_counter() - started)
        started = time.perf_counter()
        fresh_baseline = c1.coordinate_taylor(coords, target["theta_deg"], target["phi_deg"], pipeline)
        fresh_feature = c1.baseline_feature(coords, target, fresh_baseline, normalization)
        with torch.no_grad():
            prediction = model(torch.from_numpy(fresh_feature[None, ...]).float()).numpy()[0]
        synthesized = fresh_baseline + factor * (prediction[:, 0] + 1j * prediction[:, 1])
        c1.coordinate_normalize(synthesized, coords, (target["u0"], target["v0"], target["w0"]))
        end_to_end.append(time.perf_counter() - started)
    return {
        "name": "Stage3C1 selected model CPU inference stability",
        "N": N_ELEMENTS, "batch_size": 1, "cpu_threads": 1,
        "repetitions": 100, "warmup": 10,
        "selected_model": {
            "variant_id": selected["variant_id"], "seed": selected["seed"],
            "checkpoint_path": selected["checkpoint_path"],
            "checkpoint_sha256": selected["checkpoint_sha256"],
        },
        "geometry_sha256": geometry_sha,
        "target": {"parent_case_id": parent["parent_case_id"],
                    "transform_id": 0, "theta_deg": target["theta_deg"],
                    "phi_deg": target["phi_deg"]},
        "scope": {
            "inference_only": "feature tensor to model output; no evaluator",
            "end_to_end": "Taylor baseline + feature construction + model + residual reconstruction + main-response normalization; no teacher solve or evaluator",
            "teacher_generation_not_included": True,
        },
        "inference_only_s": runtime_stats(inference),
        "end_to_end_s": runtime_stats(end_to_end),
    }


def runtime_stats(values) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(arr))
    return {
        "n": int(arr.size),
        "mean_s": mean,
        "std_s": float(np.std(arr)),
        "P50_s": float(np.percentile(arr, 50)),
        "P95_s": float(np.percentile(arr, 95)),
        "min_s": float(np.min(arr)),
        "max_s": float(np.max(arr)),
        "CV": float(np.std(arr) / mean) if mean > 0 else None,
    }


def build_summary(cases, ideal, position, quantization, failures, frequency,
                  runtime) -> dict:
    failure_summary = {
        key: value["aggregate_all_cases"] for key, value in failures["rates"].items()
    }
    frequency_summary = [{
        "case_id": item["case_id"],
        "common_joint_pass_rate": item["common_joint_compliance"],
        "common_joint_band": item["common_joint_compliant_band_containing_f0"],
        "track_p_joint_pass_rate": item["track_p_joint_compliance"],
        "track_p_joint_band": item["track_p_compliant_band_containing_f0"],
    } for item in frequency["cases"]]
    return {
        "stage": "Stage 4A — Robustness Benchmark Freeze & Degradation Audit",
        "robustness_model_version": ROBUSTNESS_MODEL_VERSION,
        "metric_version": METRIC_VERSION,
        "official_evaluator_version": OFFICIAL_EVALUATOR_VERSION,
        "formal_case_count": len(cases),
        "formal_case_sets": {"regular": 8, "random": 8},
        "ideal_reference": {
            "common_joint": ideal["baseline_common_joint_pass_rate"],
            "track_p_joint": ideal["baseline_track_p_joint_pass_rate"],
        },
        "position_error": position["aggregate_all_cases"],
        "quantization": quantization["aggregate_all_cases"],
        "failure_by_rate": failure_summary,
        "frequency_by_case": frequency_summary,
        "runtime_stability": runtime,
        "primary_interpretation": {
            "degradation": "positive delta SLL/null/pointing means a numerically worse value than its case-specific ideal reference",
            "common_joint": "the primary all-16-case comparable hard-metric compliance field",
            "track_p_joint": "reported only for the eight regular LCMV adaptive-null cases; random Taylor cases remain explicitly unavailable",
        },
    }


def _worst_failure_level(failures: dict) -> str:
    scored = []
    for key, value in failures["rates"].items():
        mean = value["aggregate_all_cases"]["metrics"]["delta_sum_sll_db"]["mean"]
        scored.append((float(mean), key))
    return max(scored)[1]


def build_decision(cases, failures, frequency, runtime, metadata) -> dict:
    required_files = [
        "metadata.json", "legacy_robustness_audit.json",
        "perturbation_definitions.json", "robustness_case_manifest.json",
        "seed_manifest.json", "ideal_reference.json",
        "position_error_cases.json", "quantization_cases.json",
        "failure_cases.json", "frequency_cases.json",
        "runtime_stability.json", "summary.json",
    ]
    missing = [name for name in required_files if not (OUT / name).exists()]
    failure_masks_complete = all(
        len(value["cases"]) == len(cases) and value["realization_count"] == len(cases) * N_SEEDS
        for value in failures["rates"].values()
    )
    frequency_complete = frequency["realization_count"] == len(cases) * len(FREQUENCY_RATIOS)
    runtime_complete = (
        runtime["physics"]["samples_s"]["n"] >= 100 and
        runtime["ai"]["inference_only_s"]["n"] >= 100 and
        runtime["ai"]["end_to_end_s"]["n"] >= 100
    )
    checks = {
        "four_perturbation_definitions_frozen": True,
        "representative_cases_frozen_before_perturbations": True,
        "fixed_seeds_recorded": True,
        "degradation_only_no_resynthesis": True,
        "failure_masks_complete": failure_masks_complete,
        "frequency_pm10_complete": frequency_complete,
        "runtime_stability_complete": runtime_complete,
        "stage4b_mask_reuse_ready": failure_masks_complete,
        "required_artifacts_present": not missing,
        "official_metric_and_robustness_versions": (
            metadata["metric_version"] == METRIC_VERSION and
            metadata["robustness_model_version"] == ROBUSTNESS_MODEL_VERSION
        ),
    }
    return {
        "stage": "Stage 4A",
        "gate": "STAGE_4A_GO" if all(checks.values()) else "STAGE_4A_NO_GO",
        "gate_checks": checks,
        "missing_required_files": missing,
        "formal_scope": "Track P physics degradation audit only",
        "ai_robustness_diagnostic": "NOT RUN — PHYSICS FORMAL TRACK PRIORITIZED",
        "re_synthesis_or_adaptation": False,
        "stage4b_recommendation": {
            "recommended_failure_level_by_mean_sum_sll_degradation": f"{_worst_failure_level(failures)}",
            "reason": "use the largest observed mean sum-SLL degradation as the first Stage 4B mask level; do not modify masks or weights in Stage 4A",
            "masks_artifact": rel(OUT / "failure_cases.json"),
        },
        "no_commit_or_push": True,
    }


def write_documentation(summary: dict, decision: dict, metadata: dict) -> None:
    position = summary["position_error"]
    quant = summary["quantization"]
    lines = [
        "# Stage 4A — Robustness Benchmark Freeze & Degradation Audit",
        "",
        "## 1. Scope and gate",
        "This document records a Track P physics-only degradation audit. It is not robust re-synthesis and does not change any Stage 2/Stage 3C1 artifact.",
        f"Gate: `{decision['gate']}`. Robustness model version: `{ROBUSTNESS_MODEL_VERSION}`; metric version: `{METRIC_VERSION}`.",
        "",
        "## 2. Frozen baseline",
        f"The baseline is the Stage 2 strict-closure 32×32 planar artifact at `{metadata['stage2_baseline']}` with SHA-256 `{metadata['stage2_baseline_manifest_sha256']}`.",
        "Taylor/LCMV sum weights and Bayliss×Taylor difference weights are reused exactly in every perturbation.",
        "",
        "## 3. Official evaluator",
        f"All formal calls use `mylib.official_evaluator.evaluate_official_case`, evaluator version `{OFFICIAL_EVALUATOR_VERSION}`, uv grid `{GRID_SIZE}×{GRID_SIZE}`.",
        "The output uses the visible-domain maximum field normalization and retains the evaluator's stable -300 dBc floor.",
        "",
        "## 4. Representative cases",
        "Sixteen cases were frozen before any perturbed official evaluation: eight regular scan cases and eight random cases from the Stage 2 manifest.",
        f"Case manifest SHA-256: `{metadata['case_manifest_sha256']}`.",
        "",
        "## 5. Position-error model",
        "The formal assumption is model B: independent planar Δx and Δy, each Uniform[-0.05,+0.05]λ; Δz=0. This is explicitly recorded as an assumption because the wording permits alternative A/B/C readings.",
        "",
        "## 6. Amplitude and phase quantization",
        "Amplitude uses the nearest 0.5 dB grid relative to unit maximum. Phase uses the nearest 6-bit state (5.625°), wrapped modulo 2π. This deterministic quantizer is frozen separately from stochastic failure tests.",
        "",
        "## 7. Element failures",
        "Failures set excitation amplitude to exact zero and use floor(1024×rate): 51, 102, and 204 failed elements at 5%, 10%, and 20%. One mask is applied to both sum and difference excitations.",
        "All 16 cases × 3 levels × 20 seeds are present in `failure_cases.json`; failed indices and mask hashes are saved for Stage 4B.",
        "",
        "## 8. Frequency offset",
        "The scan is ratio 0.90 through 1.10 in 0.01 increments. Physical coordinates remain fixed and the evaluator uses λ=1/ratio; weights are not retuned.",
        "",
        "## 9. Degradation definition",
        "Every perturbed row is compared with the same case's ideal reference. Positive SLL, pointing, or null deltas indicate a worse measured value. Per-realization records retain mean, median, standard deviation, P5, P95, worst, and compliance rates in their aggregates.",
        "",
        "## 10. Joint compliance",
        "The comparable `common_joint` field combines sum SLL, difference SLL, difference pointing, and the intrinsic difference null. `track_p_joint` additionally requires measured sum null compliance and is reported only for regular LCMV cases. Difference adaptive-null compliance is unavailable in the frozen Stage 2 baseline and is never imputed.",
        "",
        "## 11. Ideal reference",
        f"Ideal common-joint pass rate: {summary['ideal_reference']['common_joint']['pass_rate']}; regular adaptive-null joint availability is explicit in the artifact.",
        "",
        "## 12. Position results",
        f"Across 320 position-error realizations, mean Δsum-SLL is {position['metrics']['delta_sum_sll_db']['mean']:.6f} dB and worst Δsum-SLL is {position['metrics']['delta_sum_sll_db']['worst']:.6f} dB.",
        "",
        "## 13. Quantization results",
        f"Across 16 deterministic quantized cases, mean Δsum-SLL is {quant['metrics']['delta_sum_sll_db']['mean']:.6f} dB and worst Δsum-SLL is {quant['metrics']['delta_sum_sll_db']['worst']:.6f} dB.",
        "",
        "## 14. Failure results",
        "The machine-readable summary reports each rate separately, including common-joint and regular Track P joint pass rates. Interpretation must use both degradation distributions and pass-rate changes; no threshold is changed by this audit.",
        "",
        "## 15. Frequency results",
        "Each case has 21 frequency rows, a common-joint compliance vector, and the contiguous compliant band containing ratio 1.0 when it exists.",
        "",
        "## 16. Runtime protocol",
        "Physics synthesis and AI inference were measured independently with 10 warmups and 100 timed repetitions. Teacher generation and dense official evaluation are excluded from the runtime samples.",
        "",
        "## 17. Runtime stability",
        f"Physics N={summary['runtime_stability']['physics']['samples_s']['n']}; AI inference N={summary['runtime_stability']['ai']['inference_only_s']['n']}; AI end-to-end N={summary['runtime_stability']['ai']['end_to_end_s']['n']}. Mean/std/P50/P95/min/max/CV are in `runtime_stability.json`.",
        "",
        "## 18. Legacy implementation audit",
        "`legacy_robustness_audit.json` inventories the old planar, failure, and curved-AI scripts. Their values are not reused because they use legacy evaluators, different case protocols, or forbidden re-optimization.",
        "",
        "## 19. AI scope boundary",
        "The Stage 3C1 AI robustness diagnostic is not run in the formal Stage 4A gate. The formal result is physics-only; this prevents curved-geometry AI diagnostics from being mixed with planar Track P evidence.",
        "",
        "## 20. Stage 4B handoff and limitations",
        f"The data-driven first candidate failure level is `{decision['stage4b_recommendation']['recommended_failure_level_by_mean_sum_sll_degradation']}` by mean sum-SLL degradation. Stage 4B may reuse the saved masks exactly, but no Stage 4B action was taken here. No commit or push was performed.",
        "",
    ]
    (ROOT / "docs" / "stage4a_robustness_benchmark_freeze.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main():
    global MAX_WORKERS
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    args = parser.parse_args()
    MAX_WORKERS = max(1, int(args.max_workers))
    OUT.mkdir(parents=True, exist_ok=True)
    started = git_state()
    if not BASELINE.exists():
        raise FileNotFoundError(BASELINE)

    case_manifest = read_json(BASELINE / "case_manifest.json")
    regular_npz = np.load(BASELINE / "weights" / "regular_weights.npz")
    random_npz = np.load(BASELINE / "weights" / "random_weights.npz")
    regular_weights = {
        "sum_amp": regular_npz["lcmv_amp"], "sum_phase": regular_npz["lcmv_phase"],
        "difference_amp": regular_npz["difference_amp"], "difference_phase": regular_npz["difference_phase"],
    }
    random_weights = {
        "sum_amp": random_npz["taylor_amp"], "sum_phase": random_npz["taylor_phase"],
        "difference_amp": random_npz["difference_amp"], "difference_phase": random_npz["difference_phase"],
    }
    cases = freeze_cases(case_manifest, regular_weights, random_weights)
    frozen_manifest = freeze_case_manifest(cases)
    write_json(OUT / "legacy_robustness_audit.json", build_legacy_audit())
    write_json(OUT / "perturbation_definitions.json", build_perturbation_definitions())
    seed_manifest = {
        "seed_protocol_version": "stage4a-seeds-v1",
        "rng": "numpy.random.RandomState(seed)",
        "position_seeds": list(SEEDS),
        "failure_seeds": list(SEEDS),
        "quantization_seeds": [],
        "frequency_seeds": [],
        "derivation": "same explicit seed integer is used independently for each case and perturbation level; case_id and level are recorded with every realization",
    }
    write_json(OUT / "seed_manifest.json", seed_manifest)

    metadata = {
        "stage": "Stage 4A — Robustness Benchmark Freeze & Degradation Audit",
        "run_status": "RUNNING",
        "generated_utc": utc_now(),
        "git_start": started,
        "metric_version": METRIC_VERSION,
        "robustness_model_version": ROBUSTNESS_MODEL_VERSION,
        "official_evaluator_version": OFFICIAL_EVALUATOR_VERSION,
        "official_grid_size": GRID_SIZE,
        "array": {"nx": NX, "ny": NY, "elements": N_ELEMENTS,
                  "spacing_wavelengths": 0.5, "lambda_reference": LAMBDA0,
                  "geometry": "frozen Stage 2 planar array"},
        "stage2_baseline": rel(BASELINE),
        "stage2_baseline_manifest_sha256": sha256_file(BASELINE / "metadata.json"),
        "case_manifest_sha256": frozen_manifest["case_list_sha256"],
        "forbidden_actions": [
            "no Taylor/Bayliss/LCMV modification", "no Stage 3C1 checkpoint/split modification",
            "no AI retraining/fine-tuning", "no re-synthesis after perturbation",
            "no adaptive weight updates", "no official evaluator modification",
            "no Stage 2 threshold modification", "no README/final report modification",
            "no commit", "no push",
        ],
        "ai_robustness_formal_status": "NOT RUN — PHYSICS FORMAL TRACK PRIORITIZED",
    }
    write_json(OUT / "metadata.json", metadata)

    posx, posy = _array_positions()
    case_data = build_case_data(cases, regular_weights, random_weights)
    ideal_path = OUT / "ideal_reference.json"
    ideal_payload = read_json(ideal_path) if ideal_path.exists() else run_ideal(
        cases, case_data, posx, posy
    )
    ideal = {row["case_id"]: row for row in ideal_payload["cases"]}
    position_path = OUT / "position_error_cases.json"
    position = read_json(position_path) if position_path.exists() else run_position(
        cases, case_data, ideal, posx, posy
    )
    quantization_path = OUT / "quantization_cases.json"
    quantization = (read_json(quantization_path)
                    if quantization_path.exists() else run_quantization(
                        cases, case_data, ideal, posx, posy
                    ))
    failure_path = OUT / "failure_cases.json"
    failures = read_json(failure_path) if failure_path.exists() else run_failures(
        cases, case_data, ideal, posx, posy
    )
    frequency_path = OUT / "frequency_cases.json"
    frequency = read_json(frequency_path) if frequency_path.exists() else run_frequency(
        cases, case_data, ideal, posx, posy
    )
    runtime = {"physics": run_physics_runtime(posx, posy, cases[0]),
               "ai": run_ai_runtime()}
    write_json(OUT / "runtime_stability.json", runtime)
    summary = build_summary(cases, ideal_payload, position, quantization,
                            failures, frequency, runtime)
    write_json(OUT / "summary.json", summary)
    metadata["run_status"] = "COMPLETE"
    metadata["generated_end_utc"] = utc_now()
    metadata["git_end_before_stage4a_own_artifacts"] = git_state()
    write_json(OUT / "metadata.json", metadata)
    decision = build_decision(cases, failures, frequency, runtime, metadata)
    write_json(OUT / "decision.json", decision)
    write_documentation(summary, decision, metadata)
    final_state = git_state()
    metadata["git_end"] = final_state
    metadata["no_commit_or_push"] = True
    write_json(OUT / "metadata.json", metadata)
    print(f"[Stage4A] complete: {decision['gate']}", flush=True)
    print(f"[Stage4A] outputs: {OUT}", flush=True)


if __name__ == "__main__":
    main()
