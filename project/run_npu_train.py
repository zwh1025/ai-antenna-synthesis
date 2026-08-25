"""NPU 全规模训练脚本。

用 Ascend 910 NPU 跑 200+ epochs，验证网络收敛后 2D SLL 能否逼近解析基线。
"""

import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mylib.dataset import create_dataset, prepare_training_data, get_dataset_config
from mylib.models import Seq2SeqModel, count_parameters, predict_sequence
from mylib.train import train_model, evaluate_model, get_device
from mylib.antenna_calc import (
    uniform_linear_array_pos,
    taylor_2d_separable,
    beam_steering_phase_2d,
    combine_2d_excitation,
    scan_angle_to_1d_theta,
    calculate_2d_pattern,
    get_2d_sll,
)
from mylib.embedding import map_vec_to_num_np

output_dir = os.path.join(os.path.dirname(__file__), 'outputs')


def main():
    os.makedirs(output_dir, exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)

    dev = get_device()

    # ---- 数据集 ----
    N_list = np.arange(15, 31, 1)
    theta0_list = np.arange(30, 151, 10)
    SLL_list = [25, 30, 35]
    n_samples = len(N_list) * len(theta0_list) * len(SLL_list)
    print(f"\n[1] Dataset: {n_samples} samples "
          f"(N={15}-{30}, theta0=30-150, SLL={SLL_list})")

    X, Y = create_dataset(N_list, theta0_list, SLL_list, reference='Taylor')
    enc_in, dec_in, dec_out = prepare_training_data(X, Y)
    config = get_dataset_config(N_list)
    print(f"    encoder_input: {enc_in.shape}, decoder_input: {dec_in.shape}")

    # ---- 模型 ----
    main_neurons = [512, 512]
    branch_neurons = [256, 256, 128, 128]
    dense_neurons = [64, 32, 16]
    num_neurons = [main_neurons, branch_neurons, branch_neurons,
                   dense_neurons, dense_neurons]
    model = Seq2SeqModel(config['input_dim'], config['output_dim'], num_neurons)
    print(f"[2] Model: {count_parameters(model):,} params")

    # ---- 训练 ----
    print(f"[3] Training on {dev}: 200 epochs, batch=32, lr=1e-3")
    t0 = time.time()
    model, history = train_model(
        model, enc_in, dec_in, dec_out,
        batch_size=32, epochs=200, learning_rate=1e-3,
        patience_lr=5, patience_stop=20, verbose=True, device=dev)
    t1 = time.time()
    n_ep = len(history['loss'])
    print(f"\n    Trained {n_ep} epochs in {t1-t0:.1f}s ({(t1-t0)/n_ep:.2f}s/epoch)")
    print(f"    Best acc: {max(history['accuracy']):.4f}")
    print(f"    Final loss: {history['loss'][-1]:.6f}")

    loss, acc = evaluate_model(model, enc_in, dec_in, dec_out, batch_size=32, device=dev)
    print(f"    Eval: loss={loss:.6f}, acc={acc:.4f}")

    torch.save(model.state_dict(), os.path.join(output_dir, 'model_full_npu.pt'))
    np.savez(os.path.join(output_dir, 'history_full_npu.npz'),
             **{k: np.array(v) for k, v in history.items()})

    # ---- 1D 测试 ----
    print(f"\n[4] 1D prediction test (>=20 cases):")
    test_cases = []
    for N_test in [16, 20, 25]:
        for theta0_test in [45, 60, 75, 90, 105, 120, 135]:
            test_cases.append((N_test, theta0_test, 30))

    amp_errors = []
    phase_errors = []

    for N_test, theta0_test, SLL_test in test_cases:
        X_test, _ = create_dataset([N_test], [theta0_test], [SLL_test], reference='Taylor')
        input_seq = torch.as_tensor(X_test[:1], dtype=torch.float32, device=dev)

        generated, _ = predict_sequence(
            model, input_seq, config['output_dim'],
            config['amp_range'], config['phase_range'],
            max_steps=config['N_units_max'] + 5)

        if len(generated) == 0:
            continue

        amp_pred = np.array([g[0] for g in generated])
        phase_pred = np.array([g[1] for g in generated])

        pos = uniform_linear_array_pos(N_test)
        from mylib.antenna_calc import taylor_excitation, beam_steering_phase
        amp_ref = taylor_excitation(N_test * 0.5, pos, SLL_test)
        phase_ref = beam_steering_phase(pos, theta0_test) % (2 * np.pi)

        n_match = min(len(amp_pred), N_test)
        amp_err = np.abs(amp_pred[:n_match] - amp_ref[:n_match])
        phase_err = np.abs(phase_pred[:n_match] - phase_ref[:n_match])
        phase_err = np.minimum(phase_err, 2 * np.pi - phase_err)

        amp_errors.extend(amp_err.tolist())
        phase_errors.extend(phase_err.tolist())

    amp_errors = np.array(amp_errors)
    phase_errors_deg = np.degrees(np.array(phase_errors))
    print(f"    Amplitude error: mean={amp_errors.mean():.4f} max={amp_errors.max():.4f}")
    print(f"    Phase error (°): mean={phase_errors_deg.mean():.2f} max={phase_errors_deg.max():.2f}")

    # ---- 2D 测试 ----
    print(f"\n[5] 2D separable verification:")
    Nx, Ny = 32, 32
    posx = uniform_linear_array_pos(Nx)
    posy = uniform_linear_array_pos(Ny)
    theta = np.linspace(0, 90, 181)
    phi = np.linspace(0, 360, 361)
    bw = 0.886 * 2.0 / Nx * 180 / np.pi

    from mylib.dataset import create_dataset as cd
    test_2d = [(0, 0), (30, 0), (30, 45), (60, 0)]
    print(f"    {'θ0':>5} {'φ0':>5} {'net_SLL':>8} {'ana_SLL':>8}")

    for theta0, phi0 in test_2d:
        theta_x, theta_y = scan_angle_to_1d_theta(theta0, phi0)

        X_x, _ = cd([Nx], [theta_x], [35], reference='Taylor')
        X_y, _ = cd([Ny], [theta_y], [35], reference='Taylor')
        enc_x = torch.as_tensor(X_x[:1], dtype=torch.float32, device=dev)
        enc_y = torch.as_tensor(X_y[:1], dtype=torch.float32, device=dev)

        gen_x, _ = predict_sequence(model, enc_x, config['output_dim'],
                                    config['amp_range'], config['phase_range'],
                                    max_steps=config['N_units_max'] + 5)
        gen_y, _ = predict_sequence(model, enc_y, config['output_dim'],
                                    config['amp_range'], config['phase_range'],
                                    max_steps=config['N_units_max'] + 5)

        if len(gen_x) < Nx or len(gen_y) < Ny:
            print(f"    {theta0:>5.0f} {phi0:>5.0f}  INSUFFICIENT OUTPUT")
            continue

        amp_x = np.array([g[0] for g in gen_x[:Nx]])
        phase_x = np.array([g[1] for g in gen_x[:Nx]])
        amp_y = np.array([g[0] for g in gen_y[:Ny]])
        phase_y = np.array([g[1] for g in gen_y[:Ny]])

        amp_2d, phase_2d = combine_2d_excitation(amp_x, amp_y, phase_x, phase_y)
        pat = calculate_2d_pattern(amp_2d, phase_2d, posx, posy, theta, phi).numpy()

        cos_scan = max(np.cos(np.deg2rad(theta0)), 0.1)
        exc = 2.0 * bw / cos_scan
        sll_net = get_2d_sll(pat, theta, phi, theta0, phi0, exclude_angle=exc)

        amp_x_a, amp_y_a = taylor_2d_separable(Nx, Ny, 35)
        px_a, py_a = beam_steering_phase_2d(posx, posy, theta0, phi0)
        amp_a, phase_a = combine_2d_excitation(amp_x_a, amp_y_a, px_a, py_a)
        pat_a = calculate_2d_pattern(amp_a, phase_a, posx, posy, theta, phi).numpy()
        sll_ana = get_2d_sll(pat_a, theta, phi, theta0, phi0, exclude_angle=exc)

        print(f"    {theta0:>5.0f} {phi0:>5.0f} {sll_net:>8.1f} {sll_ana:>8.1f}")

    print(f"\n{'='*60}")
    print("NPU full-scale training complete")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
