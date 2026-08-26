"""60° 扫描 SLL 退化原因诊断。

分别在 u-v 方向余弦域和球面 θ-φ 角度域计算方向图和 SLL，
对比不同主瓣排除方法、采样精度和坐标变换的影响。
"""

import numpy as np
import torch
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mylib.antenna_calc import (
    uniform_linear_array_pos,
    taylor_2d_separable,
    beam_steering_phase_2d,
    combine_2d_excitation,
    calculate_2d_pattern,
    get_2d_sll,
    angular_distance_deg,
    get_sll_1d,
    calculate_1d_pattern,
)


def calc_pattern_theta_phi(amp_2d, phase_2d, posx, posy,
                            theta_deg, phi_deg, lamb=1.0):
    """球面角度域计算方向图 (dB)。"""
    return calculate_2d_pattern(
        amp_2d, phase_2d, posx, posy, theta_deg, phi_deg, lamb=lamb).numpy()


def calc_pattern_uv(amp_2d, phase_2d, posx, posy,
                    u_grid, v_grid, lamb=1.0):
    """u-v 方向余弦域直接计算方向图 (dB)。

    u = sin(θ)cos(φ), v = sin(θ)sin(φ)
    在均匀网格上计算，避免球面坐标的非均匀采样。
    """
    posx = np.asarray(posx, dtype=np.float64)
    posy = np.asarray(posy, dtype=np.float64)
    Nx, Ny = amp_2d.shape
    k = 2 * np.pi / lamb

    u_2d, v_2d = np.meshgrid(u_grid, v_grid, indexing='ij')
    pattern = np.zeros_like(u_2d)

    for i in range(len(u_grid)):
        for j in range(len(v_grid)):
            u = u_grid[i]
            v = v_grid[j]
            if u**2 + v**2 > 1.0:
                pattern[i, j] = -300
                continue
            psi = k * (posx[:, None] * u + posy[None, :] * v) - phase_2d
            real = np.sum(amp_2d * np.cos(psi))
            imag = np.sum(amp_2d * np.sin(psi))
            mag = np.sqrt(real**2 + imag**2)
            pattern[i, j] = mag

    peak = pattern.max()
    pattern_db = 20 * np.log10(np.where(pattern > 0, pattern / (peak + 1e-30), 1e-12))
    return pattern_db, u_2d, v_2d


def sll_theta_phi(pattern_db, theta_deg, phi_deg, theta0, phi0,
                   Nx=32, exclude_method='2bw'):
    """球面域 SLL，多种排除方法。bw 用阵列大小计算，不是网格大小。"""
    th2d, ph2d = np.meshgrid(theta_deg, phi_deg, indexing='ij')
    dist = angular_distance_deg(th2d, ph2d, theta0, phi0)

    bw = 0.886 * 2.0 / Nx * 180 / np.pi  # 用阵列大小，不是网格大小
    if exclude_method == '2bw':
        cos_scan = max(np.cos(np.deg2rad(theta0)), 0.1)
        exc = 2.0 * bw / cos_scan
        mask = dist >= exc
    elif exclude_method == 'fixed5':
        mask = dist >= 5.0
    elif exclude_method == 'fixed10':
        mask = dist >= 10.0
    elif exclude_method == 'first_null':
        exc = 1.5 * bw / max(np.cos(np.deg2rad(theta0)), 0.1)
        mask = dist >= exc
    elif exclude_method == 'fixed3':
        mask = dist >= 3.0
    else:
        mask = np.ones_like(dist, dtype=bool)

    if not np.any(mask):
        return float('nan')
    return float(np.max(pattern_db[mask]))


def sll_uv(pattern_db, u_2d, v_2d, u0, v0, exclude_radius=0.05):
    """u-v 域 SLL，圆形排除区。"""
    dist_uv = np.sqrt((u_2d - u0)**2 + (v_2d - v0)**2)
    visible = (u_2d**2 + v_2d**2) <= 1.0
    mask = (dist_uv >= exclude_radius) & visible
    if not np.any(mask):
        return float('nan')
    return float(np.max(pattern_db[mask]))


def main():
    Nx = Ny = 32
    SLL = 35
    posx = uniform_linear_array_pos(Nx)
    posy = uniform_linear_array_pos(Ny)
    amp_x, amp_y = taylor_2d_separable(Nx, Ny, SLL)

    print("=" * 80)
    print("60° 扫描 SLL 退化原因诊断 (32×32, Taylor 35dB)")
    print("=" * 80)

    # 测试角度
    scan_cases = [
        (0, 0), (15, 0), (30, 0), (30, 45), (45, 0),
        (60, 0), (60, 45), (60, 90),
    ]

    # 采样精度对比
    grids = [
        ("coarse", np.linspace(0, 90, 46), np.linspace(0, 360, 91)),
        ("medium", np.linspace(0, 90, 181), np.linspace(0, 360, 361)),
        ("fine", np.linspace(0, 90, 901), np.linspace(0, 360, 1801)),
    ]

    print(f"\n[1] 球面 θ-φ 域: 采样精度对比")
    print(f"    {'θ0':>4} {'φ0':>4} {'coarse':>8} {'medium':>8} {'fine':>8} {'exc_2bw':>8} {'exc_5°':>7} {'exc_10°':>7}")

    for theta0, phi0 in scan_cases:
        px, py = beam_steering_phase_2d(posx, posy, theta0, phi0)
        amp_2d, phase_2d = combine_2d_excitation(amp_x, amp_y, px, py)

        results = {}
        for name, th, ph in grids:
            pat = calc_pattern_theta_phi(amp_2d, phase_2d, posx, posy, th, ph)
            sll = sll_theta_phi(pat, th, ph, theta0, phi0, Nx=Nx, exclude_method='2bw')
            results[name] = sll

        # 不同排除方法 (medium 网格)
        th_m, ph_m = grids[1][1], grids[1][2]
        pat_m = calc_pattern_theta_phi(amp_2d, phase_2d, posx, posy, th_m, ph_m)
        sll_2bw = sll_theta_phi(pat_m, th_m, ph_m, theta0, phi0, Nx=Nx, exclude_method='2bw')
        sll_5 = sll_theta_phi(pat_m, th_m, ph_m, theta0, phi0, Nx=Nx, exclude_method='fixed5')
        sll_10 = sll_theta_phi(pat_m, th_m, ph_m, theta0, phi0, Nx=Nx, exclude_method='fixed10')
        sll_3 = sll_theta_phi(pat_m, th_m, ph_m, theta0, phi0, Nx=Nx, exclude_method='fixed3')

        print(f"    {theta0:>4.0f} {phi0:>4.0f} {results['coarse']:>8.1f} {results['medium']:>8.1f} "
              f"{results['fine']:>8.1f} {sll_2bw:>8.1f} {sll_5:>7.1f} {sll_10:>7.1f}")
    # u-v 域对比
    print(f"\n[2] u-v 方向余弦域: 均匀网格")
    u_grid = np.linspace(-1, 1, 401)
    v_grid = np.linspace(-1, 1, 401)

    print(f"    {'θ0':>4} {'φ0':>4} {'u0':>6} {'v0':>6} {'uv_SLL':>8} {'θφ_SLL':>8} {'diff':>6}")

    for theta0, phi0 in scan_cases:
        px, py = beam_steering_phase_2d(posx, posy, theta0, phi0)
        amp_2d, phase_2d = combine_2d_excitation(amp_x, amp_y, px, py)

        u0 = np.sin(np.deg2rad(theta0)) * np.cos(np.deg2rad(phi0))
        v0 = np.sin(np.deg2rad(theta0)) * np.sin(np.deg2rad(phi0))

        pat_uv, u_2d, v_2d = calc_pattern_uv(
            amp_2d, phase_2d, posx, posy, u_grid, v_grid)
        sll_uv_val = sll_uv(pat_uv, u_2d, v_2d, u0, v0, exclude_radius=0.08)

        # 对应的 θ-φ 域
        th_m = np.linspace(0, 90, 181)
        ph_m = np.linspace(0, 360, 361)
        pat_tp = calc_pattern_theta_phi(amp_2d, phase_2d, posx, posy, th_m, ph_m)
        sll_tp = sll_theta_phi(pat_tp, th_m, ph_m, theta0, phi0, Nx=Nx, exclude_method='2bw')

        diff = sll_uv_val - sll_tp
        print(f"    {theta0:>4.0f} {phi0:>4.0f} {u0:>6.3f} {v0:>6.3f} "
              f"{sll_uv_val:>8.1f} {sll_tp:>8.1f} {diff:>+6.1f}")

    # 找到 60° 扫描的峰值副瓣位置
    print(f"\n[3] 60° 扫描峰值副瓣定位 (θ-φ 域)")
    theta0, phi0 = 60, 0
    px, py = beam_steering_phase_2d(posx, posy, theta0, phi0)
    amp_2d, phase_2d = combine_2d_excitation(amp_x, amp_y, px, py)

    th_f = np.linspace(0, 90, 901)
    ph_f = np.linspace(0, 360, 1801)
    pat = calc_pattern_theta_phi(amp_2d, phase_2d, posx, posy, th_f, ph_f)

    bw = 0.886 * 2.0 / 32 * 180 / np.pi  # 阵列大小=32，不是网格大小
    cos_scan = max(np.cos(np.deg2rad(theta0)), 0.1)
    exc = 2.0 * bw / cos_scan
    th2d, ph2d = np.meshgrid(th_f, ph_f, indexing='ij')
    dist = angular_distance_deg(th2d, ph2d, theta0, phi0)
    mask = (dist >= exc) & (pat > -50)

    if np.any(mask):
        idx = np.unravel_index(np.argmax(np.where(mask, pat, -np.inf)), pat.shape)
        peak_theta = th_f[idx[0]]
        peak_phi = ph_f[idx[1]]
        peak_val = pat[idx[0], idx[1]]

        u_peak = np.sin(np.deg2rad(peak_theta)) * np.cos(np.deg2rad(peak_phi))
        v_peak = np.sin(np.deg2rad(peak_theta)) * np.sin(np.deg2rad(peak_phi))
        dist_from_main = angular_distance_deg(peak_theta, peak_phi, theta0, phi0)

        print(f"  Peak sidelobe: θ={peak_theta:.1f}°, φ={peak_phi:.1f}°")
        print(f"  SLL = {peak_val:.1f} dB")
        print(f"  u={u_peak:.4f}, v={v_peak:.4f}")
        print(f"  Angular distance from main lobe: {dist_from_main:.1f}°")
        print(f"  Exclusion radius: {exc:.1f}°")

        # 计算该位置对应的 1D 方向图值
        u_main = np.sin(np.deg2rad(theta0)) * np.cos(np.deg2rad(phi0))
        v_main = np.sin(np.deg2rad(theta0)) * np.sin(np.deg2rad(phi0))
        print(f"\n  Main lobe: u={u_main:.4f}, v={v_main:.4f}")

        # 分别看 x 和 y 方向的 1D 方向图在该位置的值
        pat_x = calculate_1d_pattern(
            posx, amp_x, px, np.linspace(0, 180, 1801)).numpy()
        pat_y = calculate_1d_pattern(
            posy, amp_y, py, np.linspace(0, 180, 1801)).numpy()

        # x 方向在 u_peak 处的值
        u_1d = np.cos(np.linspace(0, 180, 1801) * np.pi / 180)
        x_val = np.interp(u_peak, u_1d[::-1], pat_x[::-1])
        # y 方向在 v_peak 处的值
        y_val = np.interp(v_peak, u_1d[::-1], pat_y[::-1])

        print(f"\n  1D pattern at sidelobe location:")
        print(f"    Fx(u={u_peak:.4f}) = {x_val:.1f} dB")
        print(f"    Fy(v={v_peak:.4f}) = {y_val:.1f} dB")
        print(f"    Fx+Fy = {x_val+y_val:.1f} dB (2D should be ≈ this)")
        print(f"    Actual 2D = {peak_val:.1f} dB")

    # 结论
    print(f"\n{'='*80}")
    print("DIAGNOSIS SUMMARY")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
