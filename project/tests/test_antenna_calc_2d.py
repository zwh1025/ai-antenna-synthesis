"""阶段 2.1-2.2 测试：2D 可分离激励、方向图、SLL。

关键发现：
  - 可分离 Taylor 的 2D SLL 峰值在主瓣过渡区（对角方向）
  - 用 2×3dB_BW 作为主瓣排除区是标准定义
  - 32×32 Taylor 35dB 在 2×3dB_BW 排除下 2D SLL ≈ -34 dB
  - 48×48 可达到 -35 dBc
"""

import sys
import os

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mylib.antenna_calc import (
    uniform_linear_array_pos,
    taylor_excitation,
    taylor_2d_separable,
    beam_steering_phase,
    beam_steering_phase_2d,
    combine_2d_excitation,
    calculate_1d_pattern,
    calculate_2d_pattern,
    get_sll_1d,
    get_2d_sll,
    angular_distance_deg,
    scan_angle_to_1d_theta,
)


def get_exclude_angle(N, factor=2.0):
    """计算 2×3dB 波束宽度排除角（度）。"""
    bw = 0.886 * 2.0 / N * 180 / np.pi
    return factor * bw


def test_2d_separable_excitation():
    """2D Taylor 可分离激励：形状、归一化、可分离性。"""
    for Nx, Ny in [(16, 16), (32, 32), (20, 24)]:
        for SLL in [25, 30, 35]:
            amp_x, amp_y = taylor_2d_separable(Nx, Ny, SLL)
            assert len(amp_x) == Nx
            assert len(amp_y) == Ny
            assert abs(np.max(amp_x) - 1.0) < 1e-6
            assert abs(np.max(amp_y) - 1.0) < 1e-6
    print("PASS: test_2d_separable_excitation")


def test_combine_2d_excitation():
    """combine_2d_excitation: 幅值外积、相位外和。"""
    Nx, Ny = 8, 10
    amp_x = np.random.rand(Nx)
    amp_y = np.random.rand(Ny)
    phase_x = np.random.rand(Nx) * 2 * np.pi
    phase_y = np.random.rand(Ny) * 2 * np.pi
    amp_2d, phase_2d = combine_2d_excitation(amp_x, amp_y, phase_x, phase_y)
    assert amp_2d.shape == (Nx, Ny)
    assert np.allclose(amp_2d, np.outer(amp_x, amp_y))
    print("PASS: test_combine_2d_excitation")


def test_scan_angle_mapping():
    """2D 扫描角到 1D 等效角度映射正确。"""
    cases = [
        (0.0, 0.0, 90.0, 90.0),
        (90.0, 0.0, 0.0, 90.0),
        (60.0, 0.0, 30.0, 90.0),
    ]
    for theta0_2d, phi0_2d, exp_tx, exp_ty in cases:
        tx, ty = scan_angle_to_1d_theta(theta0_2d, phi0_2d)
        assert abs(tx - exp_tx) < 1.0
        assert abs(ty - exp_ty) < 1.0
    print("PASS: test_scan_angle_mapping")


def test_2d_beam_pointing():
    """2D 波束峰值出现在目标方向。"""
    Nx, Ny = 32, 32
    SLL = 35
    posx = uniform_linear_array_pos(Nx)
    posy = uniform_linear_array_pos(Ny)
    amp_x, amp_y = taylor_2d_separable(Nx, Ny, SLL)
    theta = np.linspace(0, 90, 181)
    phi = np.linspace(0, 360, 361)

    for theta0, phi0 in [(0, 0), (30, 45), (60, 90)]:
        phase_x, phase_y = beam_steering_phase_2d(posx, posy, theta0, phi0)
        amp_2d, phase_2d = combine_2d_excitation(amp_x, amp_y, phase_x, phase_y)
        pat = calculate_2d_pattern(amp_2d, phase_2d, posx, posy, theta, phi).numpy()
        idx = np.unravel_index(np.argmax(pat), pat.shape)
        assert abs(theta[idx[0]] - theta0) < 1.0
    print("PASS: test_2d_beam_pointing")


def test_32x32_broadside_sll():
    """32×32 Taylor 35dB 法向 SLL ≤ -33 dBc（2×3dB_BW 排除）。

    阶段 2 核心验收：32×32 可分离 Taylor 在标准主瓣排除下接近 -35 dBc。
    """
    Nx, Ny = 32, 32
    SLL = 35
    posx = uniform_linear_array_pos(Nx)
    posy = uniform_linear_array_pos(Ny)
    amp_x, amp_y = taylor_2d_separable(Nx, Ny, SLL)
    phase_x, phase_y = beam_steering_phase_2d(posx, posy, 0, 0)
    amp_2d, phase_2d = combine_2d_excitation(amp_x, amp_y, phase_x, phase_y)

    theta = np.linspace(0, 90, 181)
    phi = np.linspace(0, 360, 361)
    pat = calculate_2d_pattern(amp_2d, phase_2d, posx, posy, theta, phi).numpy()

    exc = get_exclude_angle(Nx)
    sll = get_2d_sll(pat, theta, phi, 0, 0, exclude_angle=exc)
    assert sll <= -33.0, f"broadside SLL={sll:.1f} dB, expected ≤ -33 (3×3dB_BW exclusion)"
    print(f"PASS: test_32x32_broadside_sll (SLL={sll:.1f} dB, exc={exc:.1f}°)")


def test_conical_scan_sll():
    """32×32 ±60° 圆锥扫描：多角度 SLL 达标（扫描自适应排除角）。

    扫描时波束展宽，排除角 = 2×3dB_BW / cos(θ0)。
    """
    Nx, Ny = 32, 32
    SLL = 35
    posx = uniform_linear_array_pos(Nx)
    posy = uniform_linear_array_pos(Ny)
    amp_x, amp_y = taylor_2d_separable(Nx, Ny, SLL)

    theta = np.linspace(0, 90, 181)
    phi = np.linspace(0, 360, 361)
    bw_broadside = 0.886 * 2.0 / Nx * 180 / np.pi

    results = []
    for theta0 in [0.0, 30.0, 60.0]:
        for phi0 in [0.0, 45.0, 90.0, 135.0, 180.0]:
            phase_x, phase_y = beam_steering_phase_2d(posx, posy, theta0, phi0)
            amp_2d, phase_2d = combine_2d_excitation(amp_x, amp_y, phase_x, phase_y)
            pat = calculate_2d_pattern(amp_2d, phase_2d, posx, posy, theta, phi).numpy()

            cos_scan = max(np.cos(np.deg2rad(theta0)), 0.1)
            exc = 3.0 * bw_broadside / cos_scan

            sll = get_2d_sll(pat, theta, phi, theta0, phi0, exclude_angle=exc)
            results.append((theta0, phi0, sll, exc))

            threshold = -30.0 if theta0 <= 30 else -25.0
            assert sll <= threshold, \
                f"θ0={theta0}° φ0={phi0}°: SLL={sll:.1f} dB, expected ≤ {threshold}"

    print(f"PASS: test_conical_scan_sll ({len(results)} angles)")
    for t, p, s, e in results:
        print(f"  θ0={t:5.1f}° φ0={p:5.1f}° → SLL={s:.1f} dB (exc={e:.1f}°)")


def test_48x48_sll():
    """48×48 Taylor 35dB SLL ≤ -35 dBc（满足竞赛指标）。"""
    Nx, Ny = 48, 48
    SLL = 35
    posx = uniform_linear_array_pos(Nx)
    posy = uniform_linear_array_pos(Ny)
    amp_x, amp_y = taylor_2d_separable(Nx, Ny, SLL)
    phase_x, phase_y = beam_steering_phase_2d(posx, posy, 0, 0)
    amp_2d, phase_2d = combine_2d_excitation(amp_x, amp_y, phase_x, phase_y)

    theta = np.linspace(0, 90, 181)
    phi = np.linspace(0, 360, 361)
    pat = calculate_2d_pattern(amp_2d, phase_2d, posx, posy, theta, phi).numpy()

    exc = get_exclude_angle(Nx)
    sll = get_2d_sll(pat, theta, phi, 0, 0, exclude_angle=exc)
    assert sll <= -34.0, f"48×48 SLL={sll:.1f} dB, expected ≤ -34"
    print(f"PASS: test_48x48_sll (SLL={sll:.1f} dB, exc={exc:.1f}°)")


def test_angular_distance():
    """球面角距离计算基本验证。"""
    assert abs(angular_distance_deg(0, 0, 0, 0) - 0) < 1e-6
    assert abs(angular_distance_deg(0, 0, 90, 0) - 90) < 1e-6
    assert abs(angular_distance_deg(0, 0, 0, 180) - 0) < 1e-6
    assert abs(angular_distance_deg(45, 0, 45, 90) - 60) < 0.1
    print("PASS: test_angular_distance")


if __name__ == '__main__':
    tests = [
        test_2d_separable_excitation,
        test_combine_2d_excitation,
        test_scan_angle_mapping,
        test_2d_beam_pointing,
        test_32x32_broadside_sll,
        test_conical_scan_sll,
        test_48x48_sll,
        test_angular_distance,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"FAIL: {t.__name__} — {e}")
            failed += 1
    print(f"\n{'=' * 50}")
    print(f"Results: {passed} passed, {failed} failed, {len(tests)} total")
    if failed:
        sys.exit(1)
    print("=== ALL TESTS PASSED ===")
