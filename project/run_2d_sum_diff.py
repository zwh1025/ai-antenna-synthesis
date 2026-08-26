"""2D 和差波束验证。

将 1D 和差波束扩展到 2D 平面阵：
  - 和波束：2D Taylor 可分离
  - 差波束：x 方向 Bayliss + y 方向 Taylor
  - Capon 2D 置零
"""

import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mylib.antenna_calc import (
    uniform_linear_array_pos, taylor_2d_separable, taylor_excitation,
    beam_steering_phase_2d, combine_2d_excitation,
    calculate_2d_pattern, get_2d_sll, angular_distance_deg,
)
from mylib.sum_diff import bayliss_excitation, capon_nulling_2d, monopulse_metrics


def get_exc(N, theta0):
    bw = 0.886 * 2.0 / N * 180 / np.pi
    return 3.0 * bw / max(np.cos(np.deg2rad(theta0)), 0.1)


def main():
    Nx, Ny = 48, 48
    SLL = 35
    theta0, phi0 = 0.0, 0.0

    posx = uniform_linear_array_pos(Nx)
    posy = uniform_linear_array_pos(Ny)

    # 和波束: 2D Taylor
    amp_x_sum, amp_y_sum = taylor_2d_separable(Nx, Ny, SLL)
    px_sum, py_sum = beam_steering_phase_2d(posx, posy, theta0, phi0)
    amp_sum_2d, phase_sum_2d = combine_2d_excitation(
        amp_x_sum, amp_y_sum, px_sum, py_sum)

    # 差波束: x方向Bayliss + y方向Taylor
    amp_x_diff, _ = bayliss_excitation(Nx, 35)
    amp_y_diff = taylor_excitation(Ny * 0.5, posy, SLL)
    px_diff, py_diff = beam_steering_phase_2d(posx, posy, theta0, phi0)
    amp_diff_2d, phase_diff_2d = combine_2d_excitation(
        amp_x_diff, amp_y_diff, px_diff, py_diff)

    theta = np.linspace(0, 90, 181)
    phi = np.linspace(0, 360, 361)
    exc = get_exc(Nx, theta0)

    # 计算方向图
    pat_sum = calculate_2d_pattern(
        amp_sum_2d, phase_sum_2d, posx, posy, theta, phi).numpy()
    pat_diff = calculate_2d_pattern(
        amp_diff_2d, phase_diff_2d, posx, posy, theta, phi).numpy()

    # 指标
    sll_sum = get_2d_sll(pat_sum, theta, phi, theta0, phi0, exc)
    sll_diff = get_2d_sll(pat_diff, theta, phi, theta0, phi0, exc)

    # 差波束零深：在主瓣方向找最小值
    th2d, ph2d = np.meshgrid(theta, phi, indexing='ij')
    dist = angular_distance_deg(th2d, ph2d, theta0, phi0)
    ml_mask = dist <= 3.0
    null_depth = float(np.min(pat_diff[ml_mask]))

    # 主瓣峰值
    sum_peak = float(np.max(pat_sum))

    print("=" * 60)
    print(f"2D Sum-Difference Beam ({Nx}×{Ny})")
    print("=" * 60)
    print(f"\n  Sum beam:")
    print(f"    SLL:       {sll_sum:.1f} dB  (target ≤ -35)")
    print(f"    Peak:      {sum_peak:.1f} dB")

    print(f"\n  Diff beam:")
    print(f"    SLL:       {sll_diff:.1f} dB  (target ≤ -20)")
    print(f"    Null depth: {null_depth:.1f} dB  (target ≤ -30)")

    # Capon 2D 置零
    print(f"\n  Capon nulling (4 nulls):")
    null_dirs = [(30, 0), (30, 90), (30, 180), (30, 270)]
    amp_nulled, phase_nulled = capon_nulling_2d(
        posx, posy, amp_sum_2d, phase_sum_2d, theta0, phi0, null_dirs)

    pat_nulled = calculate_2d_pattern(
        amp_nulled, phase_nulled, posx, posy, theta, phi).numpy()
    sll_nulled = get_2d_sll(pat_nulled, theta, phi, theta0, phi0, exc)

    null_depths = []
    for tn, pn in null_dirs:
        nearby = dist <= 5.0
        tn2d = angular_distance_deg(th2d, ph2d, tn, pn) <= 5.0
        if np.any(tn2d):
            null_depths.append(float(np.max(pat_nulled[tn2d])))

    print(f"    SLL after nulling: {sll_nulled:.1f} dB")
    for i, (tn, pn) in enumerate(null_dirs):
        nd = null_depths[i] if i < len(null_depths) else float('nan')
        print(f"    Null at ({tn}°,{pn}°): {nd:.1f} dB  (target ≤ -30)")

    # 3dB 波束宽度
    pat_sum_phi0 = pat_sum[:, 0]
    from mylib.antenna_calc import get_3db_beamwidth_1d
    bw = get_3db_beamwidth_1d(pat_sum_phi0, theta, theta0)
    print(f"\n  3dB beamwidth: {bw:.2f}°")
    print(f"  Pointing accuracy target: ≤ {bw/30:.3f}°")

    # 汇总
    print(f"\n{'='*60}")
    print("2D Sum-Diff Metrics Summary")
    print(f"{'='*60}")
    metrics = [
        ("Sum SLL", sll_sum, -35),
        ("Diff SLL", sll_diff, -20),
        ("Diff null depth", null_depth, -30),
        ("3dB BW", bw, None),
    ]
    for name, val, target in metrics:
        if target is not None:
            status = "✓" if val <= target else "✗"
            print(f"  {name:>20}: {val:>8.1f}  (target ≤ {target})  {status}")
        else:
            print(f"  {name:>20}: {val:>8.2f}°")

    for i, (tn, pn) in enumerate(null_dirs):
        nd = null_depths[i] if i < len(null_depths) else float('nan')
        status = "✓" if nd <= -30 else "✗"
        print(f"  {'Null at '+str(tn)+'°,'+str(pn)+'°':>20}: {nd:>8.1f}  (target ≤ -30)  {status}")


if __name__ == '__main__':
    main()
