"""Stage 2 AI-v3 official-caliber acceptance: three arms on the frozen 273 cases.

目的：把 upstream v3 的平面阵 AI 验收（此前为 legacy evaluate_uv 口径）
补一版 official evaluator 1.0.0 口径，与已落盘的
results/stage2_strict_closure/baseline（Taylor/LCMV 基线）同方向、同评估
器、同门槛直接可比。

三条测试臂（与 case_manifest 的 regular 73 + random 200 完全一致）：
  ai_direct : v3 模型直接输出（坐标 Taylor 基线 + ΔW_AI，无 LCMV）
  ai_lcmv   : v3 输出 + capon_nulling_2d 加零陷（完整 AI 管线）
  参照      : 基线工件中的 taylor / lcmv 行（不重跑，直接引用文件）

说明：
  - v3 平面训练样本目标 ΔW=0（平面阵 Taylor 已最优），模型在平面阵上
    是"Taylor 复现器"，自身不产生零陷；零陷由 LCMV 提供。
  - 随机集的 4 个零陷方向按 case_manifest 原值使用，与基线完全一致。
  - 本脚本只评估、不训练、不改评估器；输出目录不覆盖任何现有结果。

用法：
  python run_stage2_ai_v3_official.py            # 默认模型
  python run_stage2_ai_v3_official.py --model_path <pt>

输出: results/stage2_ai_v3_official/
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "project"
RESULTS = ROOT / "results"
BASELINE = RESULTS / "stage2_strict_closure" / "baseline"
OUT = RESULTS / "stage2_ai_v3_official"

sys.path.insert(0, str(PROJECT))

from mylib.antenna_calc import (  # noqa: E402
    taylor_2d_separable,
    taylor_excitation,
    uniform_linear_array_pos,
)
from mylib.deepsets import DeepSetsModel  # noqa: E402
from mylib.official_evaluator import (  # noqa: E402
    ADAPTIVE_NULL_THRESHOLD_DB,
    DEFAULT_GRID_SIZE,
    OFFICIAL_EVALUATOR_VERSION,
    STRICT_SUM_NULL_THRESHOLD_DB,
    SUM_SLL_THRESHOLD_DB,
    evaluate_official_case,
)
from mylib.sum_diff import capon_nulling_2d  # noqa: E402

NX = NY = 32
SLL_DESIGN = 35
COORD_NORM = 8.0
SLL_NORM = 50.0
WEIGHT_SCALE = 1024.0
HIDDEN_DIM = 256
DEFAULT_MODEL = PROJECT / "outputs" / "deepsets_model_v3_256.pt"


def model_predict(model, px_flat, py_flat, pz_flat, amp_x, amp_y, theta0, phi0):
    """v3 前向：坐标 Taylor 基线 + ΔW_AI（与训练管线一致）。"""
    from run_curved_verify import coordinate_taylor_3d
    from run_generate_teacher import normalize_weights

    w_t = coordinate_taylor_3d(px_flat, py_flat, pz_flat, amp_x, amp_y,
                               theta0, phi0)
    u0 = np.sin(np.deg2rad(theta0)) * np.cos(np.deg2rad(phi0))
    v0 = np.sin(np.deg2rad(theta0)) * np.sin(np.deg2rad(phi0))
    w0 = np.cos(np.deg2rad(theta0))
    w_t = normalize_weights(w_t, px_flat, py_flat, pz_flat, u0, v0, w0)

    n = len(px_flat)
    feat = np.stack([
        px_flat / COORD_NORM, py_flat / COORD_NORM, pz_flat / COORD_NORM,
        w_t.real * WEIGHT_SCALE, w_t.imag * WEIGHT_SCALE,
        np.full(n, u0), np.full(n, v0), np.full(n, w0),
        np.full(n, SLL_DESIGN / SLL_NORM),
    ], axis=-1).astype(np.float32)

    with torch.no_grad():
        delta = model(torch.as_tensor(feat[None]))[0].numpy()
    return w_t + (delta[:, 0] + 1j * delta[:, 1]) / WEIGHT_SCALE


def _aggregate(rows, value_key, pass_key):
    values = np.asarray(
        [r[value_key] for r in rows if r.get(value_key) is not None],
        dtype=float)
    if values.size == 0:
        return {"cases": len(rows), "pass": 0, "fail": 0,
                "not_tested": len(rows), "worst": None, "mean": None}
    passes = np.asarray(
        [bool(r[pass_key]) for r in rows if r.get(value_key) is not None])
    return {
        "cases": len(rows),
        "pass": int(np.sum(passes)),
        "fail": int(np.sum(~passes)),
        "not_tested": 0,
        "worst": float(np.max(values)),
        "best": float(np.min(values)),
        "mean": float(np.mean(values)),
    }


def _null_aggregate(rows, strict=False):
    key = ("all_center_pass_strict_minus65" if strict
           else "all_center_pass_minus30")
    return {
        "cases": len(rows),
        "pass": int(sum(bool(row[key]) for row in rows)),
        "fail": int(sum(not bool(row[key]) for row in rows)),
        "not_tested": 0,
        "worst": float(max(max(row["center_db"]) for row in rows)),
        "requirement": (STRICT_SUM_NULL_THRESHOLD_DB if strict
                        else ADAPTIVE_NULL_THRESHOLD_DB),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=str(DEFAULT_MODEL))
    args = parser.parse_args()

    if OUT.exists():
        raise SystemExit(f"Refusing to overwrite existing output: {OUT}")
    OUT.mkdir(parents=True)

    manifest = json.loads((BASELINE / "case_manifest.json").read_text())
    cases = manifest["regular"] + manifest["random"]

    model = DeepSetsModel(input_dim=9, hidden_dim=HIDDEN_DIM, output_dim=2)
    model.load_state_dict(torch.load(args.model_path, map_location="cpu",
                                     weights_only=True))
    model.eval()
    print(f"model: {args.model_path}")

    posx = uniform_linear_array_pos(NX)
    posy = uniform_linear_array_pos(NY)
    amp_x_sum, amp_y_sum = taylor_2d_separable(NX, NY, SLL_DESIGN)
    _ = taylor_excitation(NY * 0.5, posy, SLL_DESIGN)  # amp_y unused for AI sum
    amp_x_diff = None  # difference arms reuse baseline weights

    px_flat = np.tile(posx[:, None], (1, NY)).ravel()
    py_flat = np.tile(posy[None, :], (NX, 1)).ravel()
    pz_flat = np.zeros(NX * NY)

    direct_rows = []
    lcmv_rows = []
    adaptive_rows = []

    t0 = time.perf_counter()
    for index, case in enumerate(cases):
        theta0 = case["theta0_deg"]
        phi0 = case["phi0_deg"]
        null_dirs = [tuple(n) for n in case["null_dirs"]]

        w_ai = model_predict(model, px_flat, py_flat, pz_flat,
                             amp_x_sum, amp_y_sum, theta0, phi0)
        amp_a = np.abs(w_ai).reshape(NX, NY)
        phase_a = np.angle(w_ai).reshape(NX, NY)

        r_direct = evaluate_official_case(
            amp_a, phase_a, posx, posy, theta0, phi0,
            null_dirs=null_dirs, n_uv=DEFAULT_GRID_SIZE)
        s = r_direct["sum"]
        direct_rows.append({
            "set": case["set"], "case_id": case["case_id"],
            "method": "ai_direct", "theta0_deg": theta0, "phi0_deg": phi0,
            "metric_version": r_direct["metric_version"],
            "sll_db": s["sll_db"], "sll_threshold_db": SUM_SLL_THRESHOLD_DB,
            "sll_pass": bool(s["sll_db"] <= SUM_SLL_THRESHOLD_DB),
            "beamwidth_3db_deg": s["beamwidth_3db_deg"],
            "pointing_error_deg": s["pointing_error_deg"],
            "pointing_threshold_deg": s["pointing_threshold_deg"],
            "delta_w_ratio": float(
                np.linalg.norm(w_ai - _baseline_cache[index])
                / np.linalg.norm(_baseline_cache[index])),
        })

        lcmv_amp, lcmv_phase = capon_nulling_2d(
            posx, posy, amp_a, phase_a, theta0, phi0, null_dirs)
        r_lcmv = evaluate_official_case(
            lcmv_amp, lcmv_phase, posx, posy, theta0, phi0,
            null_dirs=null_dirs, n_uv=DEFAULT_GRID_SIZE)
        sl = r_lcmv["sum"]
        row = {
            "set": case["set"], "case_id": case["case_id"],
            "method": "ai_lcmv", "theta0_deg": theta0, "phi0_deg": phi0,
            "metric_version": r_lcmv["metric_version"],
            "sll_db": sl["sll_db"], "sll_threshold_db": SUM_SLL_THRESHOLD_DB,
            "sll_pass": bool(sl["sll_db"] <= SUM_SLL_THRESHOLD_DB),
            "beamwidth_3db_deg": sl["beamwidth_3db_deg"],
            "pointing_error_deg": sl["pointing_error_deg"],
            "pointing_threshold_deg": sl["pointing_threshold_deg"],
        }
        lcmv_rows.append(row)

        nulls = r_lcmv["adaptive_null"]["sum"]
        centers = nulls["center_db"]
        adaptive_rows.append({
            "set": case["set"], "case_id": case["case_id"],
            "theta0_deg": theta0, "phi0_deg": phi0,
            "metric_version": r_lcmv["metric_version"],
            "null_count": len(centers),
            "sum_sll_db": sl["sll_db"],
            "sum_sll_pass": bool(sl["sll_db"] <= SUM_SLL_THRESHOLD_DB),
            "center_db": centers,
            "window_worst_db": nulls["window_worst_db"],
            "all_center_pass_minus30": bool(
                len(centers) >= 4 and
                all(v <= ADAPTIVE_NULL_THRESHOLD_DB for v in centers)),
            "all_center_pass_strict_minus65": bool(
                len(centers) >= 4 and
                all(v <= STRICT_SUM_NULL_THRESHOLD_DB for v in centers)),
            "joint_pass_minus30": bool(
                sl["sll_db"] <= SUM_SLL_THRESHOLD_DB and
                len(centers) >= 4 and
                all(v <= ADAPTIVE_NULL_THRESHOLD_DB for v in centers)),
        })

        if (index + 1) % 10 == 0 or index + 1 == len(cases):
            print(f"  {index + 1}/{len(cases)} "
                  f"({time.perf_counter() - t0:.0f}s) "
                  f"th={theta0:.1f} direct={s['sll_db']:.2f} "
                  f"lcmv={sl['sll_db']:.2f}", flush=True)

    direct_regular = [r for r in direct_rows if r["set"] == "regular"]
    direct_random = [r for r in direct_rows if r["set"] == "random"]
    lcmv_regular = [r for r in lcmv_rows if r["set"] == "regular"]
    lcmv_random = [r for r in lcmv_rows if r["set"] == "random"]
    adaptive_regular = [r for r in adaptive_rows if r["set"] == "regular"]

    table = []
    for name, rows in [
        ("AI direct SLL (regular 73)", direct_regular),
        ("AI direct SLL (random 200)", direct_random),
        ("AI+LCMV SLL (regular 73)", lcmv_regular),
        ("AI+LCMV SLL (random 200)", lcmv_random),
    ]:
        table.append({
            "metric": name,
            "aggregate": _aggregate(rows, "sll_db", "sll_pass"),
            "requirement": "<= -35 dBc",
        })
    table.append({
        "metric": "AI+LCMV adaptive null center (regular 73)",
        "aggregate": _null_aggregate(adaptive_regular, strict=False),
        "requirement": "4 nulls, each <= -30 dBc",
    })
    table.append({
        "metric": "AI+LCMV strict null center (regular 73)",
        "aggregate": _null_aggregate(adaptive_regular, strict=True),
        "requirement": "4 nulls, each <= -65 dBc",
    })

    base_summary = json.loads(
        (BASELINE / "stage2_baseline_summary.json").read_text())
    base_taylor = base_summary["baseline_table"][0]["aggregate"]
    base_lcmv = base_summary["baseline_table"][1]["aggregate"]

    summary = {
        "stage": "STAGE_2_AI_V3_OFFICIAL",
        "metric_version": OFFICIAL_EVALUATOR_VERSION,
        "model": os.path.basename(args.model_path),
        "arms": ["ai_direct", "ai_lcmv"],
        "case_source": "results/stage2_strict_closure/baseline/case_manifest.json",
        "baseline_reference": {
            "sum_taylor_sll_worst": base_taylor["worst"],
            "sum_taylor_pass": f"{base_taylor['pass']}/{base_taylor['cases']}",
            "sum_lcmv_sll_worst": base_lcmv["worst"],
            "sum_lcmv_pass": f"{base_lcmv['pass']}/{base_lcmv['cases']}",
        },
        "table": table,
        "counts": {
            "regular": len(direct_regular), "random": len(direct_random),
        },
        "provenance": {
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                            time.gmtime()),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
        },
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    (OUT / "ai_direct_cases.json").write_text(
        json.dumps(direct_rows, indent=2))
    (OUT / "ai_lcmv_cases.json").write_text(json.dumps(lcmv_rows, indent=2))
    (OUT / "adaptive_null_cases.json").write_text(
        json.dumps(adaptive_rows, indent=2))

    print("\n=== AI v3 official-caliber summary ===")
    for row in table:
        agg = row["aggregate"]
        if agg.get("worst") is None:
            print(f"  {row['metric']}: not tested")
            continue
        print(f"  {row['metric']}: pass {agg['pass']}/{agg['cases']}, "
              f"worst {agg['worst']:.3f} dBc")
    print(f"\nwritten: {OUT}")


_baseline_cache = []


if __name__ == "__main__":
    # 预载坐标 Taylor 基线用于 delta_w_ratio 统计
    from mylib.antenna_calc import (beam_steering_phase_2d,
                                    combine_2d_excitation)
    from run_curved_verify import coordinate_taylor_3d
    from run_generate_teacher import normalize_weights

    _manifest = json.loads((BASELINE / "case_manifest.json").read_text())
    _posx = uniform_linear_array_pos(NX)
    _posy = uniform_linear_array_pos(NY)
    _ax, _ay = taylor_2d_separable(NX, NY, SLL_DESIGN)
    _pxf = np.tile(_posx[:, None], (1, NY)).ravel()
    _pyf = np.tile(_posy[None, :], (NX, 1)).ravel()
    _pzf = np.zeros(NX * NY)
    for _case in _manifest["regular"] + _manifest["random"]:
        _t, _p = _case["theta0_deg"], _case["phi0_deg"]
        _w = coordinate_taylor_3d(_pxf, _pyf, _pzf, _ax, _ay, _t, _p)
        _u = np.sin(np.deg2rad(_t)) * np.cos(np.deg2rad(_p))
        _v = np.sin(np.deg2rad(_t)) * np.sin(np.deg2rad(_p))
        _wc = np.cos(np.deg2rad(_t))
        _baseline_cache.append(
            normalize_weights(_w, _pxf, _pyf, _pzf, _u, _v, _wc))

    main()
