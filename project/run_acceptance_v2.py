"""修正后的正式竞赛验收程序。

修正项:
  1. θ=0°只取φ=0°一个方向(其余φ重复) → 73个独立方向
  2. 指向误差: 球面角距离(θ+φ)
  3. 3dB波束宽度: 经过主瓣方向的截面
  4. 零深: 目标点响应+实际最深+位置偏差
  5. SLL: 方向自适应第一零点(主) + 3×3dB_BW(对比)

θ = 0, 10, 20, 30, 40, 50, 60 (7值)
θ=0时φ=0(1方向), θ>0时φ=0,30,...330(12方向)
共 1 + 6×12 = 73 个独立方向
"""

import os
import sys
import time
import json

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mylib.antenna_calc import (
    uniform_linear_array_pos,
    taylor_2d_separable,
    taylor_excitation,
    beam_steering_phase_2d,
    combine_2d_excitation,
)
from mylib.sum_diff import bayliss_excitation, capon_nulling_2d
from mylib.official_evaluator import (
    OFFICIAL_EVALUATOR_VERSION,
    SUM_SLL_THRESHOLD_DB,
    ADAPTIVE_NULL_THRESHOLD_DB,
    evaluate_official_case,
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')

NX = NY = 32
SLL = 35


def get_scan_directions():
    """73个独立扫描方向。"""
    dirs = [(0, 0)]
    for theta in [10, 20, 30, 40, 50, 60]:
        for phi in range(0, 360, 30):
            dirs.append((theta, phi))
    return dirs


def get_null_dirs(theta0, phi0):
    """4个零陷方向，与主瓣至少15°角距离。"""
    candidates = [
        (30, (phi0 + 90) % 360),
        (30, (phi0 + 180) % 360),
        (30, (phi0 + 270) % 360),
        (min(theta0 + 25, 85), (phi0 + 45) % 360),
    ]
    nulls = []
    from mylib.antenna_calc import angular_distance_deg
    for tn, pn in candidates:
        dist = angular_distance_deg(tn, pn, theta0, phi0)
        if dist >= 15.0:
            nulls.append((tn, pn))
    while len(nulls) < 4:
        nulls.append((85, (phi0 + len(nulls) * 60) % 360))
    return nulls[:4]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    dirs = get_scan_directions()

    print("=" * 80)
    print(f"Formal Acceptance v2 ({NX}×{NY} = {NX*NY} elements)")
    print(f"Directions: {len(dirs)} independent (θ=0→1, θ>0→12 each)")
    print(f"Official evaluator: {OFFICIAL_EVALUATOR_VERSION}")
    print(f"SLL: first-null uv envelope (official) + 3×3dB_BW (diagnostic)")
    print("=" * 80)

    posx = uniform_linear_array_pos(NX)
    posy = uniform_linear_array_pos(NY)
    amp_x_sum, amp_y_sum = taylor_2d_separable(NX, NY, SLL)
    amp_x_diff, _ = bayliss_excitation(NX, SLL)
    amp_y_diff = taylor_excitation(NY * 0.5, posy, SLL)

    results_taylor = []
    results_lcmv = []
    t0 = time.time()

    for i, (theta0, phi0) in enumerate(dirs):
        px, py = beam_steering_phase_2d(posx, posy, theta0, phi0)
        amp_sum, phase_sum = combine_2d_excitation(amp_x_sum, amp_y_sum, px, py)
        null_dirs = get_null_dirs(theta0, phi0)

        # Taylor
        r_t = evaluate_official_case(
            amp_sum, phase_sum, posx, posy, theta0, phi0, null_dirs=null_dirs)
        t_sum = r_t['sum']
        results_taylor.append({
            'theta0': theta0, 'phi0': phi0,
            'metric_version': r_t['metric_version'],
            'sll_official': t_sum['sll_db'],
            'sll_3bw_diagnostic': t_sum['diagnostic']['sll_3bw_db'],
            'pointing_err': t_sum['pointing_error_deg'],
            'bw_3db': t_sum['beamwidth_3db_deg'],
            'official': r_t,
        })

        # LCMV
        amp_l, phase_l = capon_nulling_2d(
            posx, posy, amp_sum, phase_sum, theta0, phi0, null_dirs)
        r_l = evaluate_official_case(
            amp_l, phase_l, posx, posy, theta0, phi0, null_dirs=null_dirs)
        l_sum = r_l['sum']

        w_t = (amp_sum * np.exp(1j*phase_sum)).ravel()
        w_l = (amp_l * np.exp(1j*phase_l)).ravel()
        dw = np.linalg.norm(w_l/np.abs(w_l).max() - w_t/np.abs(w_t).max()) / \
             np.linalg.norm(w_t/np.abs(w_t).max())

        null_depths = r_l['adaptive_null']['sum']['center_db']
        worst_null = max(null_depths)

        results_lcmv.append({
            'theta0': theta0, 'phi0': phi0,
            'metric_version': r_l['metric_version'],
            'sll_official': l_sum['sll_db'],
            'sll_3bw_diagnostic': l_sum['diagnostic']['sll_3bw_db'],
            'pointing_err': l_sum['pointing_error_deg'],
            'bw_3db': l_sum['beamwidth_3db_deg'],
            'worst_null_center_db': worst_null,
            'null_window_worst_db': r_l['adaptive_null']['sum']['window_worst_db'],
            'delta_w': dw,
            'official': r_l,
        })

        st = "✓" if t_sum['sll_db'] <= SUM_SLL_THRESHOLD_DB else "✗"
        sl = "✓" if l_sum['sll_db'] <= SUM_SLL_THRESHOLD_DB else "✗"
        sn = "✓" if worst_null <= ADAPTIVE_NULL_THRESHOLD_DB else "✗"
        print(f"  {i+1:3d}/{len(dirs)} θ={theta0:>3.0f}° φ={phi0:>3.0f}°: "
              f"Taylor={t_sum['sll_db']:>6.1f}{st} "
              f"LCMV={l_sum['sll_db']:>6.1f}{sl} "
              f"null={worst_null:>6.1f}{sn} "
              f"Δw={dw:.3f} pt={l_sum['pointing_error_deg']:.2f}°")

    t1 = time.time()
    print(f"\n  Time: {t1-t0:.1f}s")

    # 汇总
    def summarize(results, key, target=None):
        vals = [r[key] for r in results if not np.isnan(r.get(key, float('nan')))]
        vals = np.array(vals)
        mean = np.mean(vals)
        p95 = np.percentile(vals, 95)
        worst = np.max(vals)
        if target:
            pass_rate = np.mean(vals <= target) * 100
        else:
            pass_rate = -1
        return mean, p95, worst, pass_rate

    print(f"\n{'='*80}")
    print("TAYLOR BASELINE")
    print(f"{'='*80}")
    print(f"{'Metric':>25} {'Mean':>8} {'P95':>8} {'Worst':>8} {'Target':>8} {'Pass':>6}")
    for name, key, target in [
        ('SLL (official first-null)', 'sll_official', SUM_SLL_THRESHOLD_DB),
        ('SLL (3×3dB_BW diagnostic)', 'sll_3bw_diagnostic', None),
        ('Pointing err (°)', 'pointing_err', None),
        ('3dB BW (°)', 'bw_3db', None),
    ]:
        m, p, w, pr = summarize(results_taylor, key, target)
        t_str = f"{target}" if target else "—"
        p_str = f"{pr:.0f}%" if pr >= 0 else "—"
        print(f"  {name:>25}: {m:>8.1f} {p:>8.1f} {w:>8.1f} {t_str:>8} {p_str:>6}")

    print(f"\n{'='*80}")
    print("LCMV (corrected)")
    print(f"{'='*80}")
    print(f"{'Metric':>25} {'Mean':>8} {'P95':>8} {'Worst':>8} {'Target':>8} {'Pass':>6}")
    for name, key, target in [
        ('SLL (official first-null)', 'sll_official', SUM_SLL_THRESHOLD_DB),
        ('SLL (3×3dB_BW diagnostic)', 'sll_3bw_diagnostic', None),
        ('Worst null center', 'worst_null_center_db', ADAPTIVE_NULL_THRESHOLD_DB),
        ('Pointing err (°)', 'pointing_err', None),
        ('Δ||w||', 'delta_w', None),
    ]:
        m, p, w, pr = summarize(results_lcmv, key, target)
        t_str = f"{target}" if target else "—"
        p_str = f"{pr:.0f}%" if pr >= 0 else "—"
        print(f"  {name:>25}: {m:>8.3f} {p:>8.3f} {w:>8.3f} {t_str:>8} {p_str:>6}")

    # 通过率
    t_fn_pass = np.mean([r['sll_official'] <= SUM_SLL_THRESHOLD_DB for r in results_taylor]) * 100
    l_fn_pass = np.mean([r['sll_official'] <= SUM_SLL_THRESHOLD_DB for r in results_lcmv]) * 100
    l_null_pass = np.mean([r['worst_null_center_db'] <= ADAPTIVE_NULL_THRESHOLD_DB for r in results_lcmv]) * 100

    print(f"\n  Pass rates (official first-null, ≤-35 dBc):")
    print(f"    Taylor SLL:  {t_fn_pass:.0f}% ({sum(r['sll_official'] <= SUM_SLL_THRESHOLD_DB for r in results_taylor)}/{len(dirs)})")
    print(f"    LCMV SLL:    {l_fn_pass:.0f}% ({sum(r['sll_official'] <= SUM_SLL_THRESHOLD_DB for r in results_lcmv)}/{len(dirs)})")
    print(f"    LCMV nulls:  {l_null_pass:.0f}% ({sum(r['worst_null_center_db'] <= ADAPTIVE_NULL_THRESHOLD_DB for r in results_lcmv)}/{len(dirs)})")

    results_file = os.path.join(OUTPUT_DIR, 'acceptance_v2.json')
    with open(results_file, 'w') as f:
        json.dump({'taylor': results_taylor, 'lcmv': results_lcmv}, f, indent=2, default=str)
    print(f"\n  Results: {results_file}")
    print(f"\n{'='*80}")


if __name__ == '__main__':
    main()
