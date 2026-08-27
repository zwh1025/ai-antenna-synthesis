"""阶段 1.8: 1D baseline 端到端训练与验收。

流程：
  1. 生成数据集（Taylor 参考，N=15–30，theta0=30–150，SLL=20/25/30）
  2. 训练 Seq2SeqModel（50 epochs，早停）
  3. 在 ≥ 20 个测试样例上评估
  4. 报告幅相误差统计 + 方向图 SLL

运行：python run_baseline.py
"""

import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mylib.dataset import (
    create_dataset, prepare_training_data, get_dataset_config)
from mylib.models import Seq2SeqModel, count_parameters, predict_sequence
from mylib.train import train_model, evaluate_model, custom_accuracy
from mylib.antenna_calc import (
    uniform_linear_array_pos, taylor_excitation, beam_steering_phase,
    calculate_1d_pattern, get_sll_1d, get_3db_beamwidth_1d)
from mylib.embedding import map_vec_to_num_np


# ============================== 配置 ==============================

N_range = (15, 30, 2)
theta0_range = (30, 150, 15)
SLL_list = [25]

main_neurons = [128, 128]
branch_neurons = [64, 64, 32, 32]
dense_neurons = [32, 16, 8]
num_neurons = [main_neurons, branch_neurons, branch_neurons,
               dense_neurons, dense_neurons]

batch_size = 16
epochs = 40
learning_rate = 1e-3

output_dir = os.path.join(os.path.dirname(__file__), 'outputs')


# ============================== 主流程 ==============================

def main():
    os.makedirs(output_dir, exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)

    # ---- 1. 数据集 ----
    N_list = np.arange(N_range[0], N_range[1] + 1, N_range[2])
    theta0_list = np.arange(theta0_range[0], theta0_range[1] + 1, theta0_range[2])
    n_samples = len(N_list) * len(theta0_list) * len(SLL_list)
    print(f"[1/4] Dataset: {n_samples} samples "
          f"(N={N_range[0]}-{N_range[1]}, theta0={theta0_range[0]}-{theta0_range[1]}, "
          f"SLL={SLL_list})")

    X, Y = create_dataset(N_list, theta0_list, SLL_list, reference='Taylor')
    enc_in, dec_in, dec_out = prepare_training_data(X, Y)
    config = get_dataset_config(N_list)
    print(f"      encoder_input: {enc_in.shape}, decoder_input: {dec_in.shape}")

    # ---- 2. 模型 ----
    print(f"[2/4] Model: {count_parameters(Seq2SeqModel(32, 32, num_neurons)):,} params")
    model = Seq2SeqModel(config['input_dim'], config['output_dim'], num_neurons)

    # ---- 3. 训练 ----
    print(f"[3/4] Training: {epochs} epochs, batch={batch_size}, lr={learning_rate}")
    t0 = time.time()
    model, history = train_model(
        model, enc_in, dec_in, dec_out,
        batch_size=batch_size, epochs=epochs, learning_rate=learning_rate,
        patience_lr=3, patience_stop=12, verbose=True)
    t1 = time.time()
    print(f"      Trained in {t1-t0:.1f}s ({len(history['loss'])} epochs, "
          f"best val_loss={min(history['val_loss']):.6f})")

    loss, acc = evaluate_model(model, enc_in, dec_in, dec_out, batch_size)
    print(f"      Train: loss={loss:.6f}")

    # 保存模型
    torch.save(model.state_dict(), os.path.join(output_dir, 'baseline_model.pt'))
    np.savez(os.path.join(output_dir, 'training_history.npz'),
             **{k: np.array(v) for k, v in history.items()})

    # ---- 4. 测试 ----
    print(f"[4/4] Testing on >= 20 cases...")
    test_cases = []
    for N_test in [16, 20, 25]:
        for theta0_test in [45, 60, 75, 90, 105, 120, 135]:
            test_cases.append((N_test, theta0_test, 25))

    amp_errors = []
    phase_errors = []
    sll_values = []
    bw_values = []
    peak_errors = []

    for N_test, theta0_test, SLL_test in test_cases:
        X_test, Y_test = create_dataset(
            [N_test], [theta0_test], [SLL_test], reference='Taylor')
        input_seq = torch.as_tensor(X_test[:1], dtype=torch.float32)

        generated, raw = predict_sequence(
            model, input_seq, config['output_dim'],
            config['amp_range'], config['phase_range'],
            max_steps=config['N_units_max'] + 5)

        if len(generated) == 0:
            print(f"  N={N_test} theta0={theta0_test}: NO OUTPUT")
            continue

        amp_pred = np.array([g[0] for g in generated])
        phase_pred = np.array([g[1] for g in generated])

        pos = uniform_linear_array_pos(N_test)
        amp_ref = taylor_excitation(N_test * 0.5, pos, SLL_test)
        phase_ref = beam_steering_phase(pos, theta0_test) % (2 * np.pi)

        n_match = min(len(amp_pred), N_test)
        amp_err = np.abs(amp_pred[:n_match] - amp_ref[:n_match])
        phase_err = np.abs(phase_pred[:n_match] - phase_ref[:n_match])
        phase_err = np.minimum(phase_err, 2 * np.pi - phase_err)

        amp_errors.extend(amp_err.tolist())
        phase_errors.extend(phase_err.tolist())

        theta = np.linspace(0, 180, 361)
        pos_pred = uniform_linear_array_pos(len(amp_pred))
        pat_pred = calculate_1d_pattern(
            pos_pred, amp_pred, phase_pred, theta).numpy()
        pat_ref = calculate_1d_pattern(
            pos, amp_ref, phase_ref, theta).numpy()

        peak_pred_idx = int(np.argmax(pat_pred))
        peak_pred = theta[peak_pred_idx]
        sll_pred = get_sll_1d(pat_pred, theta, theta0_test, exclude_half_width=8.0)
        sll_ref = get_sll_1d(pat_ref, theta, theta0_test, exclude_half_width=8.0)
        bw_pred = get_3db_beamwidth_1d(pat_pred, theta, theta0_test)

        sll_values.append(sll_pred)
        bw_values.append(bw_pred)
        peak_errors.append(abs(peak_pred - theta0_test))

    # ---- 汇总 ----
    print(f"\n{'='*60}")
    print(f"BASELINE RESULTS ({len(amp_errors)} predictions)")
    print(f"{'='*60}")

    amp_errors = np.array(amp_errors)
    phase_errors = np.array(phase_errors)
    phase_errors_deg = np.degrees(phase_errors)

    print(f"\nAmplitude error:")
    print(f"  mean={amp_errors.mean():.4f}  max={amp_errors.max():.4f}  "
          f"std={amp_errors.std():.4f}")
    print(f"\nPhase error (degrees):")
    print(f"  mean={phase_errors_deg.mean():.2f}  max={phase_errors_deg.max():.2f}  "
          f"std={phase_errors_deg.std():.2f}")
    print(f"\nPeak direction error (degrees):")
    print(f"  mean={np.mean(peak_errors):.2f}  max={np.max(peak_errors):.2f}")
    print(f"\nSLL (dB):")
    print(f"  mean={np.mean(sll_values):.1f}  min={np.min(sll_values):.1f}  "
          f"max={np.max(sll_values):.1f}")
    print(f"  (Taylor {SLL_test}dB reference SLL ~ -{SLL_test}dB)")
    print(f"\n3dB beamwidth (degrees):")
    print(f"  mean={np.mean(bw_values):.2f}  range=[{np.min(bw_values):.2f}, "
          f"{np.max(bw_values):.2f}]")

    print(f"\n{'='*60}")
    print("BASELINE COMPLETE")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
