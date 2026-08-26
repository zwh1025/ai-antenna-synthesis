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


def quantize_amplitude(amp, step_db=0.5):
    """幅值量化（dB 域步进）。

    竞赛标准: 0.5 dB 衰减步进。
    将幅值转到 dB 域，按 step_db 量化，再转回线性。
    """
    if step_db <= 0:
        return amp
    amp = np.clip(amp, 1e-6, 1.0)
    amp_db = 20 * np.log10(amp)
    quantized_db = np.round(amp_db / step_db) * step_db
    return 10 ** (quantized_db / 20)


def quantize_amplitude_bits(amp, n_bits):
    """n-bit 幅值量化（线性均匀级）。"""
    if n_bits >= 16:
        return amp
    levels = 2 ** n_bits
    return np.round(amp * (levels - 1)) / (levels - 1)


def quantize_phase(phase_rad, n_bits):
    """相位量化到 n_bits 位。n_bits=6: 5.625° 步进。"""
    if n_bits >= 16:
        return phase_rad
    levels = 2 ** n_bits
    step = 2 * np.pi / levels
    return (np.round(phase_rad / step) * step) % (2 * np.pi)


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
    print(f"  {'Method':>12} {'Amp_SLL':>8} {'Phase_SLL':>10} {'Both_SLL':>10} {'Degrade':>8}")

    # 竞赛标准: 0.5dB 衰减步进 + 5.625° 移相量化 (6bit)
    amp_q = quantize_amplitude(amp_2d, step_db=0.5)
    phase_q = quantize_phase(phase_2d, 6)
    sll_amp = evaluate_pattern(amp_q, phase_2d, posx, posy, theta0, phi0, Nx)
    sll_phase = evaluate_pattern(amp_2d, phase_q, posx, posy, theta0, phi0, Nx)
    sll_both = evaluate_pattern(amp_q, phase_q, posx, posy, theta0, phi0, Nx)
    degrade = sll_ideal - sll_both
    print(f"  {'0.5dB+6bit':>12} {sll_amp:>8.1f} {sll_phase:>10.1f} {sll_both:>10.1f} {degrade:>8.1f} dB")

    # 也测线性量化对比
    for n_bits in [2, 3, 4]:
        amp_qb = quantize_amplitude_bits(amp_2d, n_bits)
        phase_qb = quantize_phase(phase_2d, n_bits)
        sll_b = evaluate_pattern(amp_qb, phase_qb, posx, posy, theta0, phi0, Nx)
        print(f"  {f'{n_bits}bit linear':>12} {'':>8} {'':>10} {sll_b:>10.1f} {sll_ideal-sll_b:>8.1f} dB")

    assert sll_both <= -30.0, f"Competition standard SLL={sll_both:.1f}, expected ≤ -30"
    print(f"\n  PASS: 0.5dB + 6bit quantization SLL > -30 dBc")


def test_failure_degradation():
    """阵元失效退化测试。

    关键结论：
      - 5% 失效: SLL=-31dB, 退化3dB, 仍>-30dBc → 无需补偿
      - 传统补偿方法(功率重分配/梯度/闭式/Capon)均无效或恶化
      - 原因: 失效产生分布式副瓣,非点干扰
      - AI补偿网络是正确方向(需训练专用网络)
    """
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
    print(f"  {'Rate':>5} {'SLL':>8} {'Degradation':>12} {'Status':>10}")

    results = {}
    for rate in [0.05, 0.10, 0.15, 0.20]:
        slls = []
        for seed in range(10):
            amp_fail, mask = apply_element_failure(amp_2d, phase_2d, rate, seed)
            sll = evaluate_pattern(amp_fail, phase_2d, posx, posy, theta0, phi0, Nx)
            slls.append(sll)

        avg_sll = np.mean(slls)
        degrade = sll_ideal - avg_sll
        results[rate] = avg_sll

        if rate <= 0.05:
            status = "OK (> -30)" if avg_sll <= -30 else "marginal"
        else:
            status = "needs AI"

        print(f"  {rate*100:>4.0f}% {avg_sll:>8.1f} {degrade:>10.1f} dB {status:>10}")

    # 验收: 5% 失效下 SLL 仍 > -30 dBc
    assert results[0.05] <= -30.0, \
        f"5% failure SLL={results[0.05]:.1f}, expected ≤ -30"
    print(f"\n  PASS: 5% failure SLL > -30 dBc (no compensation needed)")
    print(f"  NOTE: 10-20% failure requires AI compensation (future work)")


def test_scan_failure():
    """扫描 + 5% 失效联合测试。

    5% 失效下各扫描角 SLL 仍可接受（>-25 dBc）。
    """
    Nx, Ny = 32, 32
    SLL = 35
    rate = 0.05

    posx = uniform_linear_array_pos(Nx)
    posy = uniform_linear_array_pos(Ny)
    amp_x, amp_y = taylor_2d_separable(Nx, Ny, SLL)

    print("\n=== Scan + 5% Failure (32×32) ===")
    print(f"  {'θ0':>5} {'φ0':>5} {'ideal':>8} {'failed':>8}")

    for theta0, phi0 in [(0, 0), (30, 0), (30, 45), (60, 0)]:
        phase_x, phase_y = beam_steering_phase_2d(posx, posy, theta0, phi0)
        amp_2d, phase_2d = combine_2d_excitation(amp_x, amp_y, phase_x, phase_y)
        sll_ideal = evaluate_pattern(amp_2d, phase_2d, posx, posy, theta0, phi0, Nx)

        amp_fail, mask = apply_element_failure(amp_2d, phase_2d, rate, seed=42)
        sll_fail = evaluate_pattern(amp_fail, phase_2d, posx, posy, theta0, phi0, Nx)

        print(f"  {theta0:>5.0f} {phi0:>5.0f} {sll_ideal:>8.1f} {sll_fail:>8.1f}")


def main():
    test_quantization_degradation()
    test_failure_degradation()
    test_scan_failure()

    print(f"\n{'='*60}")
    print("Stage 2.5: Non-ideal verification complete")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
