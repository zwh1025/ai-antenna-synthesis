"""阶段 4: 强鲁棒性与 NPU 部署时延对比。

测试内容：
  1. 位置扰动 ±λ/20 验证
  2. 频带 ±10% 泛化
  3. NPU vs CPU 端到端时延对比
  4. 与 GA/PSO 加速比（理论对比）
"""

import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mylib.antenna_calc import (
    uniform_linear_array_pos,
    taylor_2d_separable,
    beam_steering_phase_2d,
    combine_2d_excitation,
    calculate_2d_pattern,
    calculate_2d_pattern_arbitrary,
    get_2d_sll,
    angular_distance_deg,
)
from mylib.train import get_device


def get_exclude_angle(N, theta0):
    bw = 0.886 * 2.0 / N * 180 / np.pi
    return 2.0 * bw / max(np.cos(np.deg2rad(theta0)), 0.1)


def eval_sll(amp, phase, posx, posy, theta0, phi0, Nx):
    theta = np.linspace(0, 90, 181)
    phi = np.linspace(0, 360, 361)
    pat = calculate_2d_pattern(amp, phase, posx, posy, theta, phi).numpy()
    exc = get_exclude_angle(Nx, theta0)
    return get_2d_sll(pat, theta, phi, theta0, phi0, exc)


def test_position_perturbation():
    """位置扰动 ±λ/20 验证（固定移相器，不重算相位）。

    竞赛场景: 移相器按理想位置设计，实际安装有位置误差。
    方向图用扰动后位置计算，但激励相位保持理想值。
    每个阵元独立 2D 扰动（不是行/列整体偏移）。
    """
    Nx, Ny = 32, 32
    SLL = 35
    theta0, phi0 = 0.0, 0.0

    posx = uniform_linear_array_pos(Nx)
    posy = uniform_linear_array_pos(Ny)
    amp_x, amp_y = taylor_2d_separable(Nx, Ny, SLL)
    # 理想相位（按理想位置设计，固定不变）
    phase_x, phase_y = beam_steering_phase_2d(posx, posy, theta0, phi0)
    amp_2d, phase_2d = combine_2d_excitation(amp_x, amp_y, phase_x, phase_y)

    sll_ideal = eval_sll(amp_2d, phase_2d, posx, posy, theta0, phi0, Nx)

    print("\n=== Position Perturbation (±λ/20, fixed phase) ===")
    print(f"  Ideal SLL: {sll_ideal:.1f} dB")
    print(f"  {'Perturb':>8} {'Mean':>8} {'Std':>8} {'Worst':>8} {'Degrade':>8}")

    all_results = {}
    for perturb in [0.025, 0.05, 0.1]:
        slls = []
        for seed in range(10):
            np.random.seed(seed)
            px_perturbed = posx[:, None] + np.random.uniform(-perturb, perturb, (Nx, Ny))
            py_perturbed = posy[None, :] + np.random.uniform(-perturb, perturb, (Nx, Ny))
            theta = np.linspace(0, 90, 181)
            phi = np.linspace(0, 360, 361)
            pat = calculate_2d_pattern_arbitrary(
                amp_2d, phase_2d, px_perturbed, py_perturbed,
                theta, phi).numpy()
            exc = get_exclude_angle(Nx, theta0)
            sll = get_2d_sll(pat, theta, phi, theta0, phi0, exc)
            slls.append(sll)

        slls = np.array(slls)
        all_results[perturb] = slls
        avg = np.mean(slls)
        std = np.std(slls)
        worst = np.max(slls)
        degrade = sll_ideal - worst
        print(f"  ±{perturb:.3f}λ {avg:>8.1f} {std:>8.1f} {worst:>8.1f} {degrade:>8.1f} dB")

    # 正确判定：用 ±λ/20=0.05 的结果
    degrade_05 = sll_ideal - np.max(all_results[0.05])
    if degrade_05 <= 5.0:
        print(f"  PASS: ±λ/20 worst-case degradation < 5 dB")
    else:
        print(f"  NOTE: ±λ/20 worst-case degradation = {degrade_05:.1f} dB")


def test_frequency_band():
    """频带 ±10% 泛化验证（固定移相器）。

    竞赛场景: 移相器按中心频率(λ=1)设计，工作频率偏离时
    电相位关系变化（k=2π/λ 改变），但移相值固定不变。
    这才是真正的宽带波束偏斜。
    """
    Nx, Ny = 32, 32
    SLL = 35
    theta0, phi0 = 0.0, 0.0

    posx = uniform_linear_array_pos(Nx)
    posy = uniform_linear_array_pos(Ny)
    amp_x, amp_y = taylor_2d_separable(Nx, Ny, SLL)
    # 按中心频率设计相位，固定不变
    phase_x, phase_y = beam_steering_phase_2d(posx, posy, theta0, phi0, lamb=1.0)
    amp_2d, phase_2d = combine_2d_excitation(amp_x, amp_y, phase_x, phase_y)

    sll_center = eval_sll(amp_2d, phase_2d, posx, posy, theta0, phi0, Nx)

    print("\n=== Frequency Band (±10%, fixed phase) ===")
    print(f"  Center freq SLL: {sll_center:.1f} dB")
    print(f"  {'freq_ratio':>10} {'SLL':>8} {'Degrade':>8}")

    freq_results = {}
    for freq_ratio in [0.90, 0.95, 1.0, 1.05, 1.10]:
        lamb = 1.0 / freq_ratio
        theta = np.linspace(0, 90, 181)
        phi = np.linspace(0, 360, 361)
        pat = calculate_2d_pattern(
            amp_2d, phase_2d, posx, posy, theta, phi, lamb=lamb).numpy()
        exc = get_exclude_angle(Nx, theta0)
        sll = get_2d_sll(pat, theta, phi, theta0, phi0, exc)
        freq_results[freq_ratio] = sll

        degrade = sll_center - sll if freq_ratio != 1.0 else 0.0
        print(f"  {freq_ratio:>10.2f} {sll:>8.1f} {degrade:>8.1f} dB")

    # 正确判定：报告 -10% 和 +10% 中的最差结果
    worst_degrade = max(
        sll_center - freq_results[0.90],
        sll_center - freq_results[1.10]
    )
    if worst_degrade <= 3.0:
        print(f"  PASS: ±10% worst-case degradation < 3 dB")
    else:
        print(f"  NOTE: ±10% worst-case degradation = {worst_degrade:.1f} dB")


def test_npu_latency():
    """NPU vs CPU 端到端时延对比（48×48 完整 2D 综合）。

    测量对象: 48×48 可分离 Taylor 综合 + Capon 置零 + 2D 方向图验证。
    包含: 解析激励生成 + 2D 组合 + Capon 求解 + 方向图计算。
    """
    print("\n=== NPU vs CPU Latency (48×48 full synthesis) ===")

    Nx, Ny = 48, 48
    SLL = 35
    posx = uniform_linear_array_pos(Nx)
    posy = uniform_linear_array_pos(Ny)

    # 1. 解析 Taylor 激励 + 2D 组合
    from mylib.sum_diff import capon_nulling_2d
    null_dirs = [(30, 0), (30, 90), (30, 180), (30, 270)]

    def full_synthesis():
        amp_x, amp_y = taylor_2d_separable(Nx, Ny, SLL)
        phase_x, phase_y = beam_steering_phase_2d(posx, posy, 0, 0)
        amp_2d, phase_2d = combine_2d_excitation(amp_x, amp_y, phase_x, phase_y)
        amp_n, phase_n = capon_nulling_2d(posx, posy, amp_2d, phase_2d,
                                          0, 0, null_dirs)
        theta = np.linspace(0, 90, 91)
        phi = np.linspace(0, 360, 181)
        pat = calculate_2d_pattern(amp_n, phase_n, posx, posy, theta, phi).numpy()
        return pat

    # 预热
    for _ in range(3):
        full_synthesis()

    n_runs = 10
    t0 = time.time()
    for _ in range(n_runs):
        pat = full_synthesis()
    t1 = time.time()
    synthesis_time = (t1 - t0) / n_runs * 1000

    # SLL 验证
    exc = get_exclude_angle(Nx, 0)
    sll = get_2d_sll(pat, np.linspace(0, 90, 91), np.linspace(0, 360, 181),
                     0, 0, exc)

    print(f"  48×48 full synthesis (Taylor + Capon + pattern): {synthesis_time:.1f} ms")
    print(f"  Resulting SLL: {sll:.1f} dB")

    # LSTM 推理时延（1D，作为对比）
    from mylib.models import Seq2SeqModel, count_parameters, predict_sequence
    from mylib.dataset import create_dataset, get_dataset_config

    config = get_dataset_config(np.arange(15, 31, 1))
    model = Seq2SeqModel(32, 32, [[512,512],[256,256,128,128],[256,256,128,128],[64,32,16],[64,32,16]])
    model_path = os.path.join(os.path.dirname(__file__), 'outputs', 'model_final.pt')
    has_model = os.path.exists(model_path)
    if has_model:
        model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=False))

    dev = get_device()
    model = model.to(dev)
    model.eval()
    X_test, _ = create_dataset([48], [90], [35], reference='Taylor')
    input_seq = torch.as_tensor(X_test[:1], dtype=torch.float32, device=dev)

    for _ in range(3):
        with torch.no_grad():
            predict_sequence(model, input_seq, 32, (0.,1.), (0.,2*np.pi), max_steps=50)

    t0 = time.time()
    for _ in range(n_runs):
        with torch.no_grad():
            predict_sequence(model, input_seq, 32, (0.,1.), (0.,2*np.pi), max_steps=50)
    if dev.type == 'npu':
        torch.npu.synchronize()
    t1 = time.time()
    lstm_time = (t1 - t0) / n_runs * 1000

    print(f"\n  LSTM inference (1D, N={Nx}): {lstm_time:.1f} ms ({'trained' if has_model else 'RANDOM weights'})")

    print(f"\n  Comparison:")
    print(f"    Analytical synthesis (48×48): {synthesis_time:.1f} ms")
    print(f"    LSTM inference (1D):         {lstm_time:.1f} ms")
    print(f"    GA (200 gen, literature):     ~30000 ms")
    print(f"    Analytical vs GA speedup:     {30000/synthesis_time:.0f}x")
    print(f"    LSTM vs GA speedup:           {30000/lstm_time:.0f}x")
    print(f"  Note: GA value from literature, not same hardware/config")


def main():
    print("=" * 60)
    print("Stage 4: Robustness & NPU Deployment")
    print("=" * 60)

    test_position_perturbation()
    test_frequency_band()
    test_npu_latency()

    print(f"\n{'='*60}")
    print("Stage 4 complete")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
