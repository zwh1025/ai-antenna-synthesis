"""阶段 1.1–1.4 物理层一致性测试。

运行方式：
  python -m pytest tests/test_antenna_calc.py -v
  或直接 python tests/test_antenna_calc.py

验收标准（来自项目计划方案 3.2）：
  - 相同输入激励下，方向图最大绝对误差 ≤ 1e-4 (float32)
  - 主瓣指向误差 = 0°（采样分辨率内）
  - 3 dB 波束宽度误差 ≤ 0.5°
  - SLL 旁瓣搜索区间显式记录
  - 以上统计基于 ≥ 20 个测试样例
"""

import sys
import os

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mylib.antenna_calc import (
    uniform_linear_array_pos,
    chebyshev_excitation,
    taylor_excitation,
    beam_steering_phase,
    calculate_1d_pattern,
    calculate_2d_pattern,
    get_sll_1d,
    get_null_depth_1d,
    get_3db_beamwidth_1d,
    get_2d_pattern_sll,
)


# ---- numpy 参考实现（独立于主模块，用于交叉验证） ----

def np_ref_1d_pattern(pos, amp, phase_rad, theta_deg, lamb=1.0):
    """numpy 参考方向图（弧度 phase, psi = k*x - phase）。"""
    theta = np.asarray(theta_deg, dtype=np.float64).reshape(-1, 1) * np.pi / 180
    pos = np.asarray(pos, dtype=np.float64).reshape(1, -1)
    amp = np.asarray(amp, dtype=np.float64).reshape(1, -1)
    phase = np.asarray(phase_rad, dtype=np.float64).reshape(1, -1)
    x = np.cos(theta) * pos
    psi = (2 * np.pi / lamb) * x - phase
    real = np.sum(amp * np.cos(psi), axis=1)
    imag = np.sum(amp * np.sin(psi), axis=1)
    pattern = np.sqrt(real ** 2 + imag ** 2)
    peak = np.max(pattern)
    ratio = np.clip(pattern / (peak + 1e-30), 1e-12, None)
    return np.log10(ratio) * 20


def np_ref_2d_pattern(amp, phase_rad, posx, posy,
                      theta_deg, phi_deg, lamb=1.0):
    """numpy 参考 2D 方向图。"""
    amp = np.asarray(amp, dtype=np.float64)
    phase = np.asarray(phase_rad, dtype=np.float64)
    posx = np.asarray(posx, dtype=np.float64)
    posy = np.asarray(posy, dtype=np.float64)
    Nx, Ny = len(posx), len(posy)
    theta = np.asarray(theta_deg).reshape(1, 1, -1, 1) * np.pi / 180
    phi = np.asarray(phi_deg).reshape(1, 1, 1, -1) * np.pi / 180
    posx = posx.reshape(Nx, 1, 1, 1)
    posy = posy.reshape(1, Ny, 1, 1)
    amp = amp.reshape(Nx, Ny, 1, 1)
    phase = phase.reshape(Nx, Ny, 1, 1)
    x = np.sin(theta) * np.cos(phi) * posx
    y = np.sin(theta) * np.sin(phi) * posy
    psi = (2 * np.pi / lamb) * (x + y) - phase
    real = np.sum(amp * np.cos(psi), axis=(0, 1))
    imag = np.sum(amp * np.sin(psi), axis=(0, 1))
    pattern = np.sqrt(real ** 2 + imag ** 2)
    peak = np.max(pattern)
    ratio = np.clip(pattern / (peak + 1e-30), 1e-12, None)
    return np.log10(ratio) * 20


# ============================================================
#  测试 1: 激励生成基本性质
# ============================================================

def test_chebyshev_basic():
    """Chebyshev 激励：阵元数、归一化、对称性。"""
    for N in [16, 20, 25, 30]:
        for SLL in [20, 25, 30]:
            amp = chebyshev_excitation(N, SLL)
            assert len(amp) == N, f"N={N}, got len={len(amp)}"
            assert abs(np.max(amp) - 1.0) < 1e-6
            assert np.allclose(amp, amp[::-1], atol=1e-6)
    print("PASS: test_chebyshev_basic")


def test_taylor_basic():
    """Taylor 激励：阵元数、归一化、对称性。"""
    for N in [16, 20, 25, 30]:
        pos = uniform_linear_array_pos(N)
        amp = taylor_excitation(N * 0.5, pos, 30)
        assert len(amp) == N
        assert abs(np.max(amp) - 1.0) < 1e-6
        assert np.allclose(amp, amp[::-1], atol=1e-6)
    print("PASS: test_taylor_basic")


def test_chebyshev_vs_taylor_sll():
    """Chebyshev 与 Taylor 激励的 SLL 应接近设计值。

    旁瓣搜索区间：[0, 90-8] ∪ [90+8, 180]，
    exclude_half_width=8.0 确保排除主瓣及第一零点。
    """
    N = 25
    pos = uniform_linear_array_pos(N)
    theta = np.linspace(0, 180, 1801)
    phase = beam_steering_phase(pos, 90)

    for SLL_design in [20, 25, 30]:
        amp_ch = chebyshev_excitation(N, SLL_design)
        amp_ta = taylor_excitation(N * 0.5, pos, SLL_design)
        phase = beam_steering_phase(pos, 90)

        pat_ch = calculate_1d_pattern(pos, amp_ch, phase, theta).numpy()
        pat_ta = calculate_1d_pattern(pos, amp_ta, phase, theta).numpy()

        sll_ch = get_sll_1d(pat_ch, theta, 90, exclude_half_width=8.0)
        sll_ta = get_sll_1d(pat_ta, theta, 90, exclude_half_width=8.0)

        assert abs(sll_ch - (-SLL_design)) < 3.0, \
            f"Chebyshev SLL={sll_ch:.1f}, design={-SLL_design}"
        assert abs(sll_ta - (-SLL_design)) < 3.0, \
            f"Taylor SLL={sll_ta:.1f}, design={-SLL_design}"
    print("PASS: test_chebyshev_vs_taylor_sll")


# ============================================================
#  测试 2: 波束指向修正（≥20 样例）
# ============================================================

def test_beam_steering_accuracy():
    """beam_steering_phase 修正后，波束峰值应在目标角度（采样分辨率内）。

    覆盖 N=15–30、theta0=30–150、SLL=10–30，共 > 20 组样例。
    """
    theta = np.linspace(0, 180, 361)
    d_theta = theta[1] - theta[0]

    count = 0
    max_error = 0.0
    for N in [15, 18, 20, 25, 30]:
        pos = uniform_linear_array_pos(N)
        amp = taylor_excitation(N * 0.5, pos, 25)
        for theta0 in [30, 45, 60, 75, 90, 105, 120, 135, 150]:
            phase = beam_steering_phase(pos, theta0)
            pattern = calculate_1d_pattern(pos, amp, phase, theta).numpy()
            peak_idx = int(np.argmax(pattern))
            peak_theta = theta[peak_idx]
            error = abs(peak_theta - theta0)
            max_error = max(max_error, error)
            assert error <= d_theta + 0.01, \
                f"N={N}, theta0={theta0}, peak={peak_theta}, err={error}"
            count += 1

    assert count >= 20, f"only {count} test cases"
    print(f"PASS: test_beam_steering_accuracy ({count} cases, max_err={max_error:.3f}°)")


# ============================================================
#  测试 3: 方向图一致性（torch vs numpy 参考实现）
# ============================================================

def test_1d_pattern_torch_vs_numpy():
    """torch 方向图 vs numpy 参考实现，逐点比对（≥20 样例）。

    使用 float64 保证精度对齐，验收阈值 1e-4。
    """
    theta = np.linspace(0, 180, 361).astype(np.float64)
    max_err_overall = 0.0
    count = 0

    for N in [15, 20, 25, 30]:
        pos = uniform_linear_array_pos(N).astype(np.float64)
        amp = taylor_excitation(N * 0.5, pos, 25).astype(np.float64)
        for theta0 in [30, 60, 90, 120, 150]:
            phase = beam_steering_phase(pos, theta0).astype(np.float64)
            pt = calculate_1d_pattern(pos, amp, phase, theta).numpy()
            pn = np_ref_1d_pattern(pos, amp, phase, theta)
            err = np.max(np.abs(pt - pn))
            max_err_overall = max(max_err_overall, err)
            assert err < 1e-4, f"N={N}, theta0={theta0}: max_err={err:.2e}"
            count += 1

    assert count >= 20
    print(f"PASS: test_1d_pattern_torch_vs_numpy "
          f"({count} cases, max_err={max_err_overall:.2e})")


def test_2d_pattern_torch_vs_numpy():
    """2D 方向图 torch vs numpy 参考实现。"""
    Nx, Ny = 8, 8
    posx = uniform_linear_array_pos(Nx).astype(np.float64)
    posy = uniform_linear_array_pos(Ny).astype(np.float64)
    amp = (chebyshev_excitation(Nx, 20)[:, None]
           * chebyshev_excitation(Ny, 20)[None, :]).astype(np.float64)
    phase = np.zeros((Nx, Ny), dtype=np.float64)
    theta = np.linspace(0, 90, 91).astype(np.float64)
    phi = np.linspace(0, 360, 361).astype(np.float64)

    pt = calculate_2d_pattern(amp, phase, posx, posy, theta, phi).numpy()
    pn = np_ref_2d_pattern(amp, phase, posx, posy, theta, phi)
    err = np.max(np.abs(pt - pn))
    assert err < 1e-3, f"2D max_err={err:.2e}"
    print(f"PASS: test_2d_pattern_torch_vs_numpy (max_err={err:.2e})")


# ============================================================
#  测试 4: 批处理一致性
# ============================================================

def test_batch_vs_single():
    """批处理方向图与逐个处理结果一致。"""
    N = 20
    pos = uniform_linear_array_pos(N)
    theta = np.linspace(0, 180, 361)
    B = 10

    sll_list = [20, 25, 30]
    theta0_list = [60, 90, 120]
    amps = []
    phases = []
    for i in range(B):
        s = sll_list[i % len(sll_list)]
        t = theta0_list[i % len(theta0_list)]
        amps.append(taylor_excitation(N * 0.5, pos, s))
        phases.append(beam_steering_phase(pos, t))

    amps = np.stack(amps)
    phases = np.stack(phases)
    pos_batch = np.tile(pos, (B, 1))

    pat_batch = calculate_1d_pattern(pos_batch, amps, phases, theta).numpy()

    max_err = 0.0
    for i in range(B):
        pat_single = calculate_1d_pattern(pos, amps[i], phases[i], theta).numpy()
        err = np.max(np.abs(pat_batch[i] - pat_single))
        max_err = max(max_err, err)
        assert err < 1e-5, f"sample {i}: err={err:.2e}"

    print(f"PASS: test_batch_vs_single (B={B}, max_err={max_err:.2e})")


# ============================================================
#  测试 5: 指标计算
# ============================================================

def test_sll_values():
    """Taylor 设计 SLL=25/30 的实际 SLL 应在设计值 4dB 以内。

    旁瓣搜索区间：[0, center-exclude] ∪ [center+exclude, 180]，
    exclude_half_width=8.0（约 2 倍主瓣零点宽度，确保排除主瓣）。
    """
    N = 25
    pos = uniform_linear_array_pos(N)
    theta = np.linspace(0, 180, 1801)
    phase = beam_steering_phase(pos, 90)

    for SLL_design in [25, 30, 35]:
        amp = taylor_excitation(N * 0.5, pos, SLL_design)
        pat = calculate_1d_pattern(pos, amp, phase, theta).numpy()
        sll = get_sll_1d(pat, theta, 90, exclude_half_width=8.0)
        assert abs(sll - (-SLL_design)) < 4.0, \
            f"design={SLL_design}, got SLL={sll:.1f}"

    print("PASS: test_sll_values")


def test_3db_beamwidth():
    """均匀激励 3dB 波束宽度与理论值对比。

    使用 0.1° 采样分辨率减小量化误差。
    """
    N = 25
    pos = uniform_linear_array_pos(N)
    amp = np.ones(N) / N
    phase = beam_steering_phase(pos, 90)
    theta = np.linspace(0, 180, 1801)

    pat = calculate_1d_pattern(pos, amp, phase, theta).numpy()
    bw = get_3db_beamwidth_1d(pat, theta, 90)

    expected = 0.886 * 2.0 / N * 180 / np.pi
    assert abs(bw - expected) < 0.5, f"BW={bw:.2f}°, expected~{expected:.2f}°"
    print(f"PASS: test_3db_beamwidth (BW={bw:.2f}°, theory={expected:.2f}°)")


def test_null_depth():
    """差波束零深：反相馈电左右半阵应在法向产生零深。"""
    N = 24
    pos = uniform_linear_array_pos(N)
    amp = taylor_excitation(N * 0.5, pos, 25)
    phase = beam_steering_phase(pos, 90)
    phase[:N // 2] += np.pi

    theta = np.linspace(0, 180, 361)
    pat = calculate_1d_pattern(pos, amp, phase, theta).numpy()
    nd = get_null_depth_1d(pat, theta, 90, search_half_width=3.0)

    assert nd < -20, f"null depth={nd:.1f} dB, expected < -20"
    print(f"PASS: test_null_depth (null={nd:.1f} dB)")


def test_2d_sll():
    """2D SLL 计算基本验证。"""
    Nx, Ny = 8, 8
    posx = uniform_linear_array_pos(Nx)
    posy = uniform_linear_array_pos(Ny)
    amp = (chebyshev_excitation(Nx, 20)[:, None]
           * chebyshev_excitation(Ny, 20)[None, :])
    phase = np.zeros((Nx, Ny))
    theta = np.linspace(0, 90, 91)
    phi = np.linspace(0, 360, 361)

    pat = calculate_2d_pattern(amp, phase, posx, posy, theta, phi).numpy()
    sll = get_2d_pattern_sll(pat, threshold=-5.0)
    assert not np.isnan(sll), "2D SLL is NaN"
    print(f"PASS: test_2d_sll (SLL={sll:.1f} dB)")


# ============================================================
#  测试 6: 可微性
# ============================================================

def test_differentiable():
    """方向图计算支持反向传播（为物理损失预留）。"""
    N = 25
    pos = uniform_linear_array_pos(N)
    amp = taylor_excitation(N * 0.5, pos, 25)
    theta = np.linspace(0, 180, 361)

    phase = torch.tensor(beam_steering_phase(pos, 90), requires_grad=True)
    amp_t = torch.tensor(amp, requires_grad=True)
    pos_t = torch.tensor(pos, dtype=torch.float32)

    pattern = calculate_1d_pattern(pos_t, amp_t, phase, theta)
    loss = torch.sum(pattern ** 2)
    loss.backward()

    assert phase.grad is not None
    assert amp_t.grad is not None
    assert not torch.any(torch.isnan(phase.grad))
    assert not torch.any(torch.isnan(amp_t.grad))
    print("PASS: test_differentiable")


# ============================================================
#  主入口
# ============================================================

if __name__ == '__main__':
    tests = [
        test_chebyshev_basic,
        test_taylor_basic,
        test_chebyshev_vs_taylor_sll,
        test_beam_steering_accuracy,
        test_1d_pattern_torch_vs_numpy,
        test_2d_pattern_torch_vs_numpy,
        test_batch_vs_single,
        test_sll_values,
        test_3db_beamwidth,
        test_null_depth,
        test_2d_sll,
        test_differentiable,
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
