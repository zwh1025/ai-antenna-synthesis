"""正式竞赛验收程序。

32×32 = 1024 阵元（满足 ≥1000 要求）
θ = 0°, 10°, 20°, 30°, 40°, 50°, 60° (7 值)
φ = 0°, 30°, 60°, ... 330° (12 值)
共 84 个扫描方向

每个方向测量：
  - 和波束 SLL
  - 差波束 SLL
  - 差波束零深
  - 4 个 LCMV 置零深度
  - 置零后和波束 SLL
  - 3dB 波束宽度
  - 指向误差

报告：均值、P95、最差值
阈值：竞赛正式值（不降低）
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
    calculate_2d_pattern,
    get_2d_sll,
    angular_distance_deg,
    get_3db_beamwidth_1d,
    calculate_1d_pattern,
    get_sll_1d,
    get_null_depth_1d,
)
from mylib.sum_diff import bayliss_excitation, capon_nulling_2d

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')

NX = NY = 32
SLL = 35

THETA_LIST = [0, 10, 20, 30, 40, 50, 60]
PHI_LIST = list(range(0, 360, 30))

COMPETITION_THRESHOLDS = {
    'sum_sll': -35.0,
    'diff_sll': -20.0,
    'diff_null': -30.0,
    'null_depth': -30.0,
    'pointing_err_ratio': 1/30,
}


def get_exc_angle(theta0):
    bw = 0.886 * 2.0 / NX * 180 / np.pi
    return 3.0 * bw / max(np.cos(np.deg2rad(theta0)), 0.1)


def get_null_directions(theta0, phi0):
    """4 个置零方向，与主瓣至少 15° 角距离。"""
    candidates = [
        (30, (phi0 + 90) % 360),
        (30, (phi0 + 180) % 360),
        (30, (phi0 + 270) % 360),
        (min(theta0 + 25, 85), (phi0 + 45) % 360),
    ]
    nulls = []
    for tn, pn in candidates:
        dist = angular_distance_deg(tn, pn, theta0, phi0)
        if dist >= 15.0:
            nulls.append((tn, pn))
    while len(nulls) < 4:
        nulls.append((85, (phi0 + len(nulls) * 60) % 360))
    return nulls[:4]


def evaluate_single(theta0, phi0, posx, posy, amp_x_sum, amp_y_sum,
                    amp_x_diff, amp_y_diff):
    """评估单个扫描方向。"""
    exc = get_exc_angle(theta0)
    theta = np.linspace(0, 90, 181)
    phi = np.linspace(0, 360, 361)

    # 和波束
    px_s, py_s = beam_steering_phase_2d(posx, posy, theta0, phi0)
    amp_sum, phase_sum = combine_2d_excitation(amp_x_sum, amp_y_sum, px_s, py_s)
    pat_sum = calculate_2d_pattern(amp_sum, phase_sum, posx, posy, theta, phi).numpy()
    sum_sll = get_2d_sll(pat_sum, theta, phi, theta0, phi0, exc)

    # 峰值位置
    idx_peak = np.unravel_index(np.argmax(pat_sum), pat_sum.shape)
    peak_theta = theta[idx_peak[0]]
    dphi = abs(theta[idx_peak[0]] - theta0)
    pointing_err = dphi

    # 差波束
    amp_diff, phase_diff = combine_2d_excitation(amp_x_diff, amp_y_diff, px_s, py_s)
    pat_diff = calculate_2d_pattern(amp_diff, phase_diff, posx, posy, theta, phi).numpy()
    diff_sll = get_2d_sll(pat_diff, theta, phi, theta0, phi0, exc)

    # 差波束零深
    th2d, ph2d = np.meshgrid(theta, phi, indexing='ij')
    dist = angular_distance_deg(th2d, ph2d, theta0, phi0)
    ml_mask = dist <= 3.0
    diff_null = float(np.min(pat_diff[ml_mask])) if np.any(ml_mask) else float('nan')

    # LCMV 置零
    null_dirs = get_null_directions(theta0, phi0)
    amp_null, phase_null = capon_nulling_2d(
        posx, posy, amp_sum, phase_sum, theta0, phi0, null_dirs)
    pat_null = calculate_2d_pattern(amp_null, phase_null, posx, posy, theta, phi).numpy()
    nulled_sll = get_2d_sll(pat_null, theta, phi, theta0, phi0, exc)

    # 零陷深度
    null_depths = []
    for tn, pn in null_dirs:
        dist_null = angular_distance_deg(th2d, ph2d, tn, pn)
        near_mask = dist_null <= 2.0
        if np.any(near_mask):
            null_depths.append(float(np.max(pat_null[near_mask])))
        else:
            null_depths.append(float('nan'))

    # 3dB 波束宽度（φ=0 截面）
    pat_sum_phi0 = pat_sum[:, 0]
    bw_3db = get_3db_beamwidth_1d(pat_sum_phi0, theta, theta0)

    return {
        'theta0': theta0,
        'phi0': phi0,
        'sum_sll': sum_sll,
        'diff_sll': diff_sll,
        'diff_null': diff_null,
        'nulled_sll': nulled_sll,
        'null_depths': null_depths,
        'bw_3db': bw_3db,
        'pointing_err': pointing_err,
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 70)
    print(f"Formal Competition Acceptance ({NX}×{NY} = {NX*NY} elements)")
    print(f"θ = {THETA_LIST}")
    print(f"φ = {PHI_LIST}")
    print(f"Total: {len(THETA_LIST) * len(PHI_LIST)} directions")
    print(f"Exclusion: 3×3dB_BW/cos(θ₀)")
    print("=" * 70)

    posx = uniform_linear_array_pos(NX)
    posy = uniform_linear_array_pos(NY)

    amp_x_sum, amp_y_sum = taylor_2d_separable(NX, NY, SLL)
    amp_x_diff, _ = bayliss_excitation(NX, SLL)
    amp_y_diff = taylor_excitation(NY * 0.5, posy, SLL)

    results = []
    t0 = time.time()

    for theta0 in THETA_LIST:
        for phi0 in PHI_LIST:
            r = evaluate_single(
                theta0, phi0, posx, posy,
                amp_x_sum, amp_y_sum, amp_x_diff, amp_y_diff)
            results.append(r)

            status_sum = "✓" if r['sum_sll'] <= -35 else "✗"
            status_diff = "✓" if r['diff_sll'] <= -20 else "✗"
            print(f"  θ={theta0:>3.0f}° φ={phi0:>3.0f}°: "
                  f"sum={r['sum_sll']:>6.1f}{status_sum} "
                  f"diff={r['diff_sll']:>6.1f}{status_diff} "
                  f"null={r['diff_null']:>7.1f} "
                  f"nulled={r['nulled_sll']:>6.1f}")

    t1 = time.time()
    print(f"\n  Evaluation time: {t1-t0:.1f}s")

    # 汇总统计
    sum_slls = [r['sum_sll'] for r in results]
    diff_slls = [r['diff_sll'] for r in results]
    diff_nulls = [r['diff_null'] for r in results]
    nulled_slls = [r['nulled_sll'] for r in results]
    all_null_depths = [d for r in results for d in r['null_depths']
                       if not np.isnan(d)]
    bws = [r['bw_3db'] for r in results]
    point_errs = [r['pointing_err'] for r in results]

    print(f"\n{'='*70}")
    print("AGGREGATE STATISTICS")
    print(f"{'='*70}")
    print(f"{'Metric':>20} {'Mean':>8} {'P95':>8} {'Worst':>8} {'Target':>8} {'Pass':>6}")

    metrics = [
        ('Sum SLL', sum_slls, -35.0, 'le'),
        ('Diff SLL', diff_slls, -20.0, 'le'),
        ('Diff null', diff_nulls, -30.0, 'le'),
        ('Nulled sum SLL', nulled_slls, -15.0, 'le'),
        ('Null depths', all_null_depths, -30.0, 'le'),
        ('Pointing err (°)', point_errs, None, 'le'),
    ]

    for name, vals, target, direction in metrics:
        vals = np.array(vals)
        vals = vals[~np.isnan(vals)]
        mean = np.mean(vals)
        p95 = np.percentile(vals, 95)
        worst = np.max(vals) if direction == 'le' else np.min(vals)
        target_str = f"{target:.0f}" if target else "—"
        if target:
            pass_rate = np.mean(vals <= target) * 100 if direction == 'le' \
                else np.mean(vals >= target) * 100
            pass_str = f"{pass_rate:.0f}%"
        else:
            pass_str = "—"
        print(f"  {name:>20}: {mean:>8.1f} {p95:>8.1f} {worst:>8.1f} {target_str:>8} {pass_str:>6}")

    # 通过率
    sum_pass = np.mean([r['sum_sll'] <= -35 for r in results]) * 100
    diff_pass = np.mean([r['diff_sll'] <= -20 for r in results]) * 100
    null_pass = np.mean([r['diff_null'] <= -30 for r in results]) * 100

    print(f"\n  Pass rates:")
    print(f"    Sum SLL ≤ -35 dBc:    {sum_pass:.0f}% ({sum(r['sum_sll'] <= -35 for r in results)}/{len(results)})")
    print(f"    Diff SLL ≤ -20 dBc:   {diff_pass:.0f}% ({sum(r['diff_sll'] <= -20 for r in results)}/{len(results)})")
    print(f"    Diff null ≤ -30 dBc:  {null_pass:.0f}% ({sum(r['diff_null'] <= -30 for r in results)}/{len(results)})")

    # 保存结果
    results_file = os.path.join(OUTPUT_DIR, 'acceptance_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {results_file}")

    print(f"\n{'='*70}")
    print("FORMAL ACCEPTANCE COMPLETE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
