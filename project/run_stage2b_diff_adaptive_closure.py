"""Stage 2B+: difference-beam adaptive-null baseline closure.

补齐 results/stage2_strict_closure/baseline 中标注
BASELINE_NOT_IMPLEMENTED 的差波束自适应零陷（73 regular cases）。

实现：capon_nulling_difference_2d（PR1 已提供的差波束 LCMV 置零，
含内在零点保护约束）作用于冻结的 Bayliss 差波束权值，
官方评估器 evaluate_official_case 的 difference 通道评估：
  - adaptive: 4 个零陷方向 center_db 全部 <= -30 dBc
  - strict:   全部 <= -50 dBc
  - 同时记录差波束 SLL 是否保持 <= -20 dBc（联合门槛）

不修改任何已冻结工件；输出到 results/stage2b_diff_adaptive_closure/。
"""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "project"
RESULTS = ROOT / "results"
BASELINE = RESULTS / "stage2_strict_closure" / "baseline"
OUT = RESULTS / "stage2b_diff_adaptive_closure"

sys.path.insert(0, str(PROJECT))

from mylib.antenna_calc import (  # noqa: E402
    angular_distance_deg,
    beam_steering_phase_2d,
    combine_2d_excitation,
    taylor_2d_separable,
    taylor_excitation,
    uniform_linear_array_pos,
)
from mylib.official_evaluator import (  # noqa: E402
    ADAPTIVE_NULL_THRESHOLD_DB,
    DEFAULT_GRID_SIZE,
    DIFFERENCE_SLL_THRESHOLD_DB,
    OFFICIAL_EVALUATOR_VERSION,
    STRICT_DIFFERENCE_NULL_THRESHOLD_DB,
    evaluate_official_case,
)
from mylib.sum_diff import (  # noqa: E402
    bayliss_excitation,
    capon_nulling_difference_2d,
    difference_null_is_legal,
)

NX = NY = 32
SLL_DESIGN = 35


def main():
    if OUT.exists():
        raise SystemExit(f"Refusing to overwrite existing output: {OUT}")
    OUT.mkdir(parents=True)

    manifest = json.loads((BASELINE / "case_manifest.json").read_text())
    regular = manifest["regular"]

    posx = uniform_linear_array_pos(NX)
    posy = uniform_linear_array_pos(NY)
    amp_x_sum, amp_y_sum = taylor_2d_separable(NX, NY, SLL_DESIGN)
    amp_x_diff, _ = bayliss_excitation(NX, SLL_DESIGN)
    amp_y_diff = taylor_excitation(NY * 0.5, posy, SLL_DESIGN)

    rows = []
    t0 = time.perf_counter()
    for index, case in enumerate(regular):
        theta0 = case["theta0_deg"]
        phi0 = case["phi0_deg"]
        null_dirs = [tuple(n) for n in case["null_dirs"]]

        px, py = beam_steering_phase_2d(posx, posy, theta0, phi0)
        diff_amp, diff_phase = combine_2d_excitation(amp_x_diff, amp_y_diff,
                                                     px, py)

        new_amp, new_phase = capon_nulling_difference_2d(
            posx, posy, diff_amp, diff_phase, theta0, phi0, null_dirs,
            difference_axis="azimuth")

        r = evaluate_official_case(
            diff_amp, diff_phase, posx, posy, theta0, phi0,
            amp_difference=new_amp, phase_difference=new_phase,
            difference_null_dirs=[(theta0, phi0)] + null_dirs,
            difference_axis_phi_deg=0.0,
            n_uv=DEFAULT_GRID_SIZE)

        diff = r["difference"]
        adaptive = r["adaptive_null"]["difference"]
        centers = adaptive["center_db"]
        windows = adaptive["window_worst_db"]

        # intrinsic center 是第一个（目标方向），其余为自适应零陷
        intrinsic = centers[0] if centers else None
        adaptive_centers = centers[1:] if len(centers) > 1 else []

        legal_flags = []
        if diff is not None:
            for tn, pn in null_dirs:
                legal_flags.append(
                    difference_null_is_legal(diff["main_lobe"], tn, pn))

        rows.append({
            "set": "regular",
            "case_id": case["case_id"],
            "theta0_deg": theta0, "phi0_deg": phi0,
            "metric_version": OFFICIAL_EVALUATOR_VERSION,
            "status": "CLOSED",
            "difference_sll_db": diff["sll_db"],
            "difference_sll_pass": bool(
                diff["sll_db"] <= DIFFERENCE_SLL_THRESHOLD_DB),
            "intrinsic_null_center_db": intrinsic,
            "adaptive_center_db": adaptive_centers,
            "adaptive_window_worst_db": (windows[1:] if len(windows) > 1
                                         else []),
            "null_count": len(adaptive_centers),
            "nulls_outside_main_lobe": legal_flags,
            "all_adaptive_pass_minus30": bool(
                len(adaptive_centers) >= 4 and
                all(v <= ADAPTIVE_NULL_THRESHOLD_DB
                    for v in adaptive_centers)),
            "all_adaptive_pass_strict_minus50": bool(
                len(adaptive_centers) >= 4 and
                all(v <= STRICT_DIFFERENCE_NULL_THRESHOLD_DB
                    for v in adaptive_centers)),
            "joint_pass_minus30": bool(
                diff["sll_db"] <= DIFFERENCE_SLL_THRESHOLD_DB and
                len(adaptive_centers) >= 4 and
                all(v <= ADAPTIVE_NULL_THRESHOLD_DB
                    for v in adaptive_centers)),
            "joint_pass_strict_minus50": bool(
                diff["sll_db"] <= DIFFERENCE_SLL_THRESHOLD_DB and
                len(adaptive_centers) >= 4 and
                all(v <= STRICT_DIFFERENCE_NULL_THRESHOLD_DB
                    for v in adaptive_centers)),
        })

        if (index + 1) % 10 == 0 or index + 1 == len(regular):
            print(f"  {index + 1}/{len(regular)} "
                  f"({time.perf_counter() - t0:.0f}s) "
                  f"th={theta0:.0f} ph={phi0:.0f} "
                  f"diff_sll={diff['sll_db']:.2f} "
                  f"adaptive_worst={max(adaptive_centers) if adaptive_centers else float('nan'):.1f}",
                  flush=True)

    n = len(rows)
    agg_adaptive = {
        "cases": n,
        "pass": sum(r["all_adaptive_pass_minus30"] for r in rows),
        "fail": n - sum(r["all_adaptive_pass_minus30"] for r in rows),
        "worst_center": float(max(max(r["adaptive_center_db"])
                                  for r in rows)),
        "requirement": ADAPTIVE_NULL_THRESHOLD_DB,
    }
    agg_strict = {
        "cases": n,
        "pass": sum(r["all_adaptive_pass_strict_minus50"] for r in rows),
        "fail": n - sum(r["all_adaptive_pass_strict_minus50"] for r in rows),
        "worst_center": float(max(max(r["adaptive_center_db"])
                                  for r in rows)),
        "requirement": STRICT_DIFFERENCE_NULL_THRESHOLD_DB,
    }
    agg_sll = {
        "cases": n,
        "pass": sum(r["difference_sll_pass"] for r in rows),
        "fail": n - sum(r["difference_sll_pass"] for r in rows),
        "worst": float(max(r["difference_sll_db"] for r in rows)),
        "requirement": DIFFERENCE_SLL_THRESHOLD_DB,
    }
    agg_joint = {
        "cases": n,
        "pass": sum(r["joint_pass_minus30"] for r in rows),
        "fail": n - sum(r["joint_pass_minus30"] for r in rows),
        "requirement": "diff SLL <= -20 and 4 adaptive nulls <= -30",
    }

    summary = {
        "stage": "STAGE_2B_DIFF_ADAPTIVE_CLOSURE",
        "metric_version": OFFICIAL_EVALUATOR_VERSION,
        "method": "capon_nulling_difference_2d on frozen Bayliss weights",
        "baseline_reference": ("closes BASELINE_NOT_IMPLEMENTED in "
                               "stage2_strict_closure/baseline"),
        "table": [
            {"metric": "Diff SLL after nulling", "aggregate": agg_sll},
            {"metric": "Diff adaptive null (4x <= -30)", "aggregate": agg_adaptive},
            {"metric": "Diff strict null (4x <= -50)", "aggregate": agg_strict},
            {"metric": "Diff joint gate (-20 & -30)", "aggregate": agg_joint},
        ],
        "provenance": {
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                            time.gmtime()),
            "python": platform.python_version(),
        },
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    (OUT / "cases.json").write_text(json.dumps(rows, indent=2))

    print("\n=== Stage 2B difference adaptive closure ===")
    for row in summary["table"]:
        a = row["aggregate"]
        print(f"  {row['metric']}: pass {a['pass']}/{a['cases']}"
              + (f", worst {a.get('worst_center', a.get('worst')):.3f}"
                 if a.get("worst_center") or a.get("worst") else ""))
    print(f"\nwritten: {OUT}")


if __name__ == "__main__":
    main()
