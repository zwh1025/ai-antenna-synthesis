"""阶段 2.3-2.7: 2D 可分离网络训练数据与 baseline。

策略：2D 可分离 = 两个 1D 问题
  - x 方向：1D Taylor 35dB + 波束指向 u₀ = sin(θ₀)cos(φ₀)
  - y 方向：1D Taylor 35dB + 波束指向 v₀ = sin(θ₀)sin(φ₀)
  - 网络复用阶段 1 的 1D Seq2Seq，对 x/y 各推理一次
  - 组合：W_2d[i,j] = Wx[i] * Wy[j]

训练数据与阶段 1 相同（1D Taylor），仅需调整 SLL=35 和 N 范围。
"""

import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mylib.dataset import create_dataset, prepare_training_data, get_dataset_config
from mylib.models import Seq2SeqModel, count_parameters, predict_sequence
from mylib.train import train_model, evaluate_model
from mylib.antenna_calc import (
    uniform_linear_array_pos,
    taylor_2d_separable,
    beam_steering_phase_2d,
    combine_2d_excitation,
    scan_angle_to_1d_theta,
    calculate_2d_pattern,
    get_2d_sll,
    angular_distance_deg,
)
from mylib.embedding import map_vec_to_num_np

output_dir = os.path.join(os.path.dirname(__file__), 'outputs')


def get_exclude_angle(N, factor=2.0):
    """2×3dB 波束宽度排除角。"""
    bw = 0.886 * 2.0 / N * 180 / np.pi
    return factor * bw


def predict_2d_separable(model, Nx, Ny, theta0_2d, phi0_2d, config):
    """用 1D 网络做 2D 可分离推理。

    对 x/y 方向各运行一次 1D 推理，组合为 2D 激励。
    """
    theta_x, theta_y = scan_angle_to_1d_theta(theta0_2d, phi0_2d)

    from mylib.dataset import create_dataset
    X_x, _ = create_dataset([Nx], [theta_x], [35], reference='Taylor')
    X_y, _ = create_dataset([Ny], [theta_y], [35], reference='Taylor')

    input_dim = config['input_dim']
    output_dim = config['output_dim']

    enc_x = torch.as_tensor(X_x[:1], dtype=torch.float32)
    enc_y = torch.as_tensor(X_y[:1], dtype=torch.float32)

    gen_x, _ = predict_sequence(
        model, enc_x, output_dim,
        config['amp_range'], config['phase_range'],
        max_steps=config['N_units_max'] + 5)
    gen_y, _ = predict_sequence(
        model, enc_y, output_dim,
        config['amp_range'], config['phase_range'],
        max_steps=config['N_units_max'] + 5)

    if len(gen_x) == 0 or len(gen_y) == 0:
        return None, None

    amp_x = np.array([g[0] for g in gen_x[:Nx]])
    phase_x = np.array([g[1] for g in gen_x[:Nx]])
    amp_y = np.array([g[0] for g in gen_y[:Ny]])
    phase_y = np.array([g[1] for g in gen_y[:Ny]])

    if len(amp_x) < Nx or len(amp_y) < Ny:
        return None, None

    posx = uniform_linear_array_pos(Nx)
    posy = uniform_linear_array_pos(Ny)

    _, phase_2d = combine_2d_excitation(amp_x, amp_y, phase_x, phase_y)
    return combine_2d_excitation(amp_x, amp_y, phase_x, phase_y)


def evaluate_2d_analytical(Nx, Ny, SLL=35):
    """解析 Taylor 2D SLL 评估（无网络，作为基线）。"""
    posx = uniform_linear_array_pos(Nx)
    posy = uniform_linear_array_pos(Ny)
    amp_x, amp_y = taylor_2d_separable(Nx, Ny, SLL)

    theta = np.linspace(0, 90, 181)
    phi = np.linspace(0, 360, 361)
    bw = 0.886 * 2.0 / Nx * 180 / np.pi

    results = []
    for theta0 in [0.0, 30.0, 60.0]:
        for phi0 in [0.0, 45.0, 90.0, 135.0, 180.0]:
            phase_x, phase_y = beam_steering_phase_2d(posx, posy, theta0, phi0)
            amp_2d, phase_2d = combine_2d_excitation(amp_x, amp_y, phase_x, phase_y)
            pat = calculate_2d_pattern(amp_2d, phase_2d, posx, posy, theta, phi).numpy()

            cos_scan = max(np.cos(np.deg2rad(theta0)), 0.1)
            exc = 2.0 * bw / cos_scan
            sll = get_2d_sll(pat, theta, phi, theta0, phi0, exclude_angle=exc)
            results.append((theta0, phi0, sll, exc))

    return results


def main():
    os.makedirs(output_dir, exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)

    Nx, Ny = 32, 32

    print("=" * 60)
    print(f"Phase 2: 2D Separable Beam Synthesis ({Nx}×{Ny})")
    print("=" * 60)

    print("\n[1] Analytical baseline (Taylor 35dB, no network):")
    results = evaluate_2d_analytical(Nx, Ny, SLL=35)
    sll_values = [r[2] for r in results]
    print(f"  SLL range: [{min(sll_values):.1f}, {max(sll_values):.1f}] dB")
    print(f"  Broadside: {[r[2] for r in results if r[0]==0][0]:.1f} dB")
    print(f"  30° scan:  {[r[2] for r in results if r[0]==30 and r[1]==0][0]:.1f} dB")
    print(f"  60° scan:  {[r[2] for r in results if r[0]==60 and r[1]==0][0]:.1f} dB")

    print("\n[2] Training 1D network (SLL=35, N=15-32)...")
    N_list = np.arange(15, 33, 2)
    theta0_list = np.arange(30, 151, 15)
    SLL_list = [35]
    n_samples = len(N_list) * len(theta0_list) * len(SLL_list)
    print(f"  Dataset: {n_samples} samples")

    X, Y = create_dataset(N_list, theta0_list, SLL_list, reference='Taylor')
    enc_in, dec_in, dec_out = prepare_training_data(X, Y)
    config = get_dataset_config(N_list)

    main_neurons = [128, 128]
    branch_neurons = [64, 64, 32, 32]
    dense_neurons = [32, 16, 8]
    num_neurons = [main_neurons, branch_neurons, branch_neurons,
                   dense_neurons, dense_neurons]
    model = Seq2SeqModel(config['input_dim'], config['output_dim'], num_neurons)
    print(f"  Model: {count_parameters(model):,} params")

    model, history = train_model(
        model, enc_in, dec_in, dec_out,
        batch_size=16, epochs=15, learning_rate=1e-3,
        patience_lr=3, patience_stop=8, verbose=True)

    loss, acc = evaluate_model(model, enc_in, dec_in, dec_out, batch_size=16)
    print(f"\n  Train: loss={loss:.6f}, acc={acc:.4f}")

    torch.save(model.state_dict(),
               os.path.join(output_dir, 'model_2d_baseline.pt'))

    print(f"\n[3] 2D verification (network-based, selected angles):")
    test_cases = [
        (0, 0), (15, 0), (30, 0), (30, 45), (45, 90), (60, 0),
    ]
    posx = uniform_linear_array_pos(Nx)
    posy = uniform_linear_array_pos(Ny)
    theta = np.linspace(0, 90, 181)
    phi = np.linspace(0, 360, 361)
    bw = 0.886 * 2.0 / Nx * 180 / np.pi

    print(f"  {'θ0':>5} {'φ0':>5} {'net_SLL':>8} {'ana_SLL':>8} {'status':>8}")
    for theta0, phi0 in test_cases:
        result = predict_2d_separable(model, Nx, Ny, theta0, phi0, config)
        if result[0] is None:
            print(f"  {theta0:>5.0f} {phi0:>5.0f}  {'FAILED':>8}")
            continue

        amp_2d, phase_2d = result
        pat = calculate_2d_pattern(amp_2d, phase_2d, posx, posy, theta, phi).numpy()
        cos_scan = max(np.cos(np.deg2rad(theta0)), 0.1)
        exc = 2.0 * bw / cos_scan
        sll_net = get_2d_sll(pat, theta, phi, theta0, phi0, exclude_angle=exc)

        amp_x_a, amp_y_a = taylor_2d_separable(Nx, Ny, 35)
        px_a, py_a = beam_steering_phase_2d(posx, posy, theta0, phi0)
        amp_a, phase_a = combine_2d_excitation(amp_x_a, amp_y_a, px_a, py_a)
        pat_a = calculate_2d_pattern(amp_a, phase_a, posx, posy, theta, phi).numpy()
        sll_ana = get_2d_sll(pat_a, theta, phi, theta0, phi0, exclude_angle=exc)

        status = "✓" if sll_net <= -30 else "~"
        print(f"  {theta0:>5.0f} {phi0:>5.0f} {sll_net:>8.1f} {sll_ana:>8.1f} {status:>8}")

    print(f"\n{'='*60}")
    print("Phase 2 baseline complete")
    print(f"  Analytical: -34 dB (broadside), -20~-24 dB (60° scan)")
    print(f"  Network: requires GPU for full-scale training")
    print(f"  Next: AI + physics loss to improve scan-angle SLL")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
