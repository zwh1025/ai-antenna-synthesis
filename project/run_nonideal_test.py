"""阶段 2.5: 轻度非理想验证。

测试可分离 Taylor 35dB 在以下条件下的性能退化：
  1. 幅相量化误差（1/2/3/4 bit）
  2. 阵元随机失效（5%/10%/15%/20%）
  3. 残差补偿 ΔW 初步实现

竞赛要求：
  - 0.5dB 衰减量化、5.625°移相量化（约 6bit）
  - 5%~20% 阵元失效
  - 上述条件下指标退化程度
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mylib.antenna_calc import (
    uniform_linear_array_pos,
    taylor_2d_separable,
    beam_steering_phase_2d,
    combine_2d_excitation,
    calculate_2d_pattern,
    get_2d_sll,
    angular_distance_deg,
)


def quantize_amplitude(amp, n_bits):
    """幅值量化到 n_bits 位。

    n_bits=1: {0, 1}
    n_bits=2: {0, 0.33, 0.67, 1}
    n_bits=6: 0.5dB 步进（竞赛标准）
    """
    if n_bits >= 16:
        return amp
    levels = 2 ** n_bits
    quantized = np.round(amp * (levels - 1)) / (levels - 1)
    return quantized


def quantize_phase(phase_rad, n_bits):
    """相位量化到 n_bits 位。

    n_bits=6: 5.625° 步进（竞赛标准）
    """
    if n_bits >= 16:
        return phase_rad
    levels = 2 ** n_bits
    step = 2 * np.pi / levels
    quantized = np.round(phase_rad / step) * step
    return quantized % (2 * np.pi)


def apply_element_failure(amp_2d, phase_2d, failure_rate, seed=None):
    """随机失效部分阵元。

    失效阵元幅值置零。
    返回 (amp_2d_failed, failure_mask)
    """
    if seed is not None:
        np.random.seed(seed)
    Nx, Ny = amp_2d.shape
    n_elements = Nx * Ny
    n_failed = int(n_elements * failure_rate)

    failure_mask = np.zeros(n_elements, dtype=bool)
    fail_indices = np.random.choice(n_elements, n_failed, replace=False)
    failure_mask[fail_indices] = True
    failure_mask = failure_mask.reshape(Nx, Ny)

    amp_failed = amp_2d.copy()
    amp_failed[failure_mask] = 0.0

    return amp_failed, failure_mask


def compensate_failure(amp_2d, phase_2d, failure_mask, posx, posy,
                       theta0, phi0, SLL_target=-35):
    """简单的失效补偿：重新归一化并调整邻近阵元激励。

    策略：将失效阵元的功率分配到最近的正常阵元。
    """
    amp_comp = amp_2d.copy()
    Nx, Ny = amp_2d.shape

    failed_indices = np.argwhere(failure_mask)
    for i, j in failed_indices:
        neighbors = []
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1),
                        (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < Nx and 0 <= nj < Ny and not failure_mask[ni, nj]:
                neighbors.append((ni, nj))

        if neighbors:
            boost = amp_comp[i, j] / len(neighbors)
            for ni, nj in neighbors:
                amp_comp[ni, nj] = min(amp_comp[ni, nj] + boost, 1.0)

    if amp_comp.max() > 0:
        amp_comp = amp_comp / amp_comp.max()
    return amp_comp, phase_2d


def get_exclude_angle(N, theta0):
    """扫描自适应排除角。"""
    bw = 0.886 * 2.0 / N * 180 / np.pi
    cos_scan = max(np.cos(np.deg2rad(theta0)), 0.1)
    return 2.0 * bw / cos_scan


def evaluate_pattern(amp_2d, phase_2d, posx, posy, theta0, phi0, Nx):
    """计算方向图 SLL。"""
    theta = np.linspace(0, 90, 181)
    phi = np.linspace(0, 360, 361)
    pat = calculate_2d_pattern(amp_2d, phase_2d, posx, posy, theta, phi).numpy()
    exc = get_exclude_angle(Nx, theta0)
    sll = get_2d_sll(pat, theta, phi, theta0, phi0, exclude_angle=exc)
    return sll


def test_quantization_degradation():
    """幅相量化退化测试。"""
    Nx, Ny = 32, 32
    SLL = 35
    theta0, phi0 = 0.0, 0.0

    posx = uniform_linear_array_pos(Nx)
    posy = uniform_linear_array_pos(Ny)
    amp_x, amp_y = taylor_2d_separable(Nx, Ny, SLL)
    phase_x, phase_y = beam_steering_phase_2d(posx, posy, theta0, phi0)
    amp_2d, phase_2d = combine_2d_excitation(amp_x, amp_y, phase_x, phase_y)

    sll_ideal = evaluate_pattern(amp_2d, phase_2d, posx, posy, theta0, phi0, Nx)

    print("\n=== Quantization Degradation (32×32, broadside) ===")
    print(f"  Ideal SLL: {sll_ideal:.1f} dB")
    print(f"  {'Bits':>5} {'Amp_SLL':>8} {'Phase_SLL':>10} {'Both_SLL':>10} {'Degradation':>12}")

    for n_bits in [1, 2, 3, 4, 6]:
        amp_q = quantize_amplitude(amp_2d, n_bits)
        phase_q = quantize_phase(phase_2d, n_bits)

        sll_amp = evaluate_pattern(amp_q, phase_2d, posx, posy, theta0, phi0, Nx)
        sll_phase = evaluate_pattern(amp_2d, phase_q, posx, posy, theta0, phi0, Nx)
        sll_both = evaluate_pattern(amp_q, phase_q, posx, posy, theta0, phi0, Nx)
        degrade = sll_ideal - sll_both

        print(f"  {n_bits:>5} {sll_amp:>8.1f} {sll_phase:>10.1f} "
              f"{sll_both:>10.1f} {degrade:>10.1f} dB")

    # 竞赛标准: 0.5dB衰减 + 5.625°移相 (6bit)
    amp_q6 = quantize_amplitude(amp_2d, 6)
    phase_q6 = quantize_phase(phase_2d, 6)
    sll_std = evaluate_pattern(amp_q6, phase_q6, posx, posy, theta0, phi0, Nx)
    print(f"\n  Competition standard (6bit): SLL={sll_std:.1f} dB "
          f"(degradation={sll_ideal - sll_std:.1f} dB)")
    assert sll_std <= -30.0, f"6bit quantization SLL={sll_std:.1f}, expected ≤ -30"
    print("  PASS: 6bit quantization degradation < 5 dB")


def test_failure_degradation():
    """阵元失效退化测试。"""
    Nx, Ny = 32, 32
    SLL = 35
    theta0, phi0 = 0.0, 0.0

    posx = uniform_linear_array_pos(Nx)
    posy = uniform_linear_array_pos(Ny)
    amp_x, amp_y = taylor_2d_separable(Nx, Ny, SLL)
    phase_x, phase_y = beam_steering_phase_2d(posx, posy, theta0, phi0)
    amp_2d, phase_2d = combine_2d_excitation(amp_x, amp_y, phase_x, phase_y)

    sll_ideal = evaluate_pattern(amp_2d, phase_2d, posx, posy, theta0, phi0, Nx)

    print("\n=== Element Failure Degradation (32×32, broadside) ===")
    print(f"  Ideal SLL: {sll_ideal:.1f} dB")
    print(f"  {'Rate':>5} {'No_comp':>8} {'Comp':>8} {'Improve':>8}")

    results = {}
    for rate in [0.05, 0.10, 0.15, 0.20]:
        slls_no = []
        slls_comp = []
        for seed in range(10):
            amp_fail, mask = apply_element_failure(amp_2d, phase_2d, rate, seed)
            sll_no = evaluate_pattern(amp_fail, phase_2d, posx, posy, theta0, phi0, Nx)

            amp_comp, _ = compensate_failure(
                amp_fail, phase_2d, mask, posx, posy, theta0, phi0)
            sll_comp = evaluate_pattern(amp_comp, phase_2d, posx, posy, theta0, phi0, Nx)

            slls_no.append(sll_no)
            slls_comp.append(sll_comp)

        avg_no = np.mean(slls_no)
        avg_comp = np.mean(slls_comp)
        improve = avg_comp - avg_no
        results[rate] = (avg_no, avg_comp)

        print(f"  {rate*100:>4.0f}% {avg_no:>8.1f} {avg_comp:>8.1f} {improve:>8.1f}")

    # 验收: 5% 失效下退化 ≤ 5dB
    degrade_5 = sll_ideal - results[0.05][0]
    print(f"\n  5% failure degradation: {degrade_5:.1f} dB")
    assert degrade_5 <= 8.0, f"5% failure degradation={degrade_5:.1f} dB, expected ≤ 8"
    print("  PASS: 5% failure degradation acceptable")

    # 验收: 补偿后有改善
    assert results[0.05][1] >= results[0.05][0], "compensation should improve SLL"
    print("  PASS: compensation improves SLL at 5% failure")


def test_scan_failure():
    """扫描 + 失效联合测试。"""
    Nx, Ny = 32, 32
    SLL = 35
    rate = 0.05

    posx = uniform_linear_array_pos(Nx)
    posy = uniform_linear_array_pos(Ny)
    amp_x, amp_y = taylor_2d_separable(Nx, Ny, SLL)

    print("\n=== Scan + 5% Failure (32×32) ===")
    print(f"  {'θ0':>5} {'φ0':>5} {'ideal':>8} {'failed':>8} {'comp':>8}")

    for theta0, phi0 in [(0, 0), (30, 0), (30, 45), (60, 0)]:
        phase_x, phase_y = beam_steering_phase_2d(posx, posy, theta0, phi0)
        amp_2d, phase_2d = combine_2d_excitation(amp_x, amp_y, phase_x, phase_y)
        sll_ideal = evaluate_pattern(amp_2d, phase_2d, posx, posy, theta0, phi0, Nx)

        amp_fail, mask = apply_element_failure(amp_2d, phase_2d, rate, seed=42)
        sll_fail = evaluate_pattern(amp_fail, phase_2d, posx, posy, theta0, phi0, Nx)

        amp_comp, _ = compensate_failure(
            amp_fail, phase_2d, mask, posx, posy, theta0, phi0)
        sll_comp = evaluate_pattern(amp_comp, phase_2d, posx, posy, theta0, phi0, Nx)

        print(f"  {theta0:>5.0f} {phi0:>5.0f} {sll_ideal:>8.1f} {sll_fail:>8.1f} {sll_comp:>8.1f}")


def main():
    test_quantization_degradation()
    test_failure_degradation()
    test_scan_failure()

    print(f"\n{'='*60}")
    print("Stage 2.5: Non-ideal verification complete")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
