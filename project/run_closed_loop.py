"""小规模闭环训练验证。

目标：用小数据 + 少量 epochs 验证完整管线：
  生成数据 → 训练 → 保存最佳模型 → 重新加载 → 自回归推理 → 解码幅相 → 计算方向图 → 输出 SLL

验收指标（不是竞赛指标，而是"管线正确性"指标）：
  - 幅值 MAE ≤ 0.1（小数据过拟合应达到）
  - 相位圆周 MAE ≤ 10°
  - 输出阵元数正确率 ≥ 80%
  - 方向图 SLL 与 Taylor 参考相差 ≤ 5 dB
"""

import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mylib.dataset import create_dataset, prepare_training_data, get_dataset_config
from mylib.models import Seq2SeqModel, count_parameters, predict_sequence
from mylib.train import train_model, get_device, masked_mse_loss, custom_accuracy
from mylib.antenna_calc import (
    uniform_linear_array_pos, taylor_excitation, beam_steering_phase,
    calculate_1d_pattern, get_sll_1d,
)
from mylib.embedding import map_vec_to_num_np

output_dir = os.path.join(os.path.dirname(__file__), 'outputs')


def decode_prediction(model, input_seq, output_dim, amp_range, phase_range, device, max_steps=35):
    """自回归推理并解码为幅值/相位数值。"""
    generated, raw = predict_sequence(
        model, input_seq, output_dim, amp_range, phase_range, max_steps=max_steps)
    if len(generated) == 0:
        return None, None
    amps = np.array([g[0] for g in generated])
    phases = np.array([g[1] for g in generated])
    return amps, phases


def circular_mae(pred, ref):
    """圆周 MAE（弧度→度）。"""
    diff = np.abs(pred - ref)
    diff = np.minimum(diff, 2 * np.pi - diff)
    return np.mean(np.degrees(diff))


def main():
    os.makedirs(output_dir, exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)
    dev = get_device()

    # 小数据集：单一 SLL=25, 少量 N 和角度
    N_list = np.arange(15, 26, 2)  # 6 values
    theta0_list = np.arange(60, 121, 15)  # 5 values
    SLL_list = [25]  # 固定 SLL
    n = len(N_list) * len(theta0_list) * len(SLL_list)
    print(f"\n[1] Dataset: {n} samples (small, single SLL=25)")

    X, Y = create_dataset(N_list, theta0_list, SLL_list, reference='Taylor')
    enc_in, dec_in, dec_out = prepare_training_data(X, Y)
    config = get_dataset_config(N_list)

    # 小模型
    main_n = [256, 256]
    branch_n = [128, 128, 64, 64]
    dense_n = [32, 16, 8]
    model = Seq2SeqModel(32, 32, [main_n, branch_n, branch_n, dense_n, dense_n])
    print(f"[2] Model: {count_parameters(model):,} params")

    # 短训练
    save_path = os.path.join(output_dir, 'model_small_test.pt')
    print(f"[3] Training: 50 epochs, batch=16, lr=1e-3")
    t0 = time.time()
    model, history = train_model(
        model, enc_in, dec_in, dec_out,
        batch_size=16, epochs=50, learning_rate=1e-3,
        patience_lr=3, patience_stop=10, verbose=True, device=dev,
        save_path=save_path)
    t1 = time.time()
    print(f"\n    {len(history['loss'])} epochs in {t1-t0:.1f}s")
    print(f"    Best val_loss: {min(history['val_loss']):.6f}")

    # 重新加载最佳模型
    if os.path.exists(save_path):
        model.load_state_dict(
            torch.load(save_path, map_location='cpu', weights_only=False))
        model = model.to(dev)
        model.eval()
        print(f"    Reloaded best model")

    # 闭环测试：自回归推理 → 方向图
    print(f"\n[4] Closed-loop test (autoregressive inference + pattern):")
    test_cases = [(N, t, 25) for N in [16, 20] for t in [60, 90, 120]]
    amp_maes = []
    phase_maes = []
    seq_len_correct = 0
    sll_diffs = []

    for N_t, theta_t, SLL_t in test_cases:
        X_t, _ = create_dataset([N_t], [theta_t], [SLL_t], reference='Taylor')
        input_seq = torch.as_tensor(X_t[:1], dtype=torch.float32, device=dev)

        amps, phases = decode_prediction(
            model, input_seq, 32, (0., 1.), (0., 2*np.pi), dev, max_steps=35)

        if amps is None:
            print(f"  N={N_t} θ={theta_t}: NO OUTPUT")
            continue

        n_out = len(amps)
        n_expected = N_t
        if n_out == n_expected:
            seq_len_correct += 1

        n_match = min(n_out, n_expected)
        pos = uniform_linear_array_pos(N_t)
        amp_ref = taylor_excitation(N_t * 0.5, pos, SLL_t)
        phase_ref = beam_steering_phase(pos, theta_t) % (2 * np.pi)

        amp_mae = np.mean(np.abs(amps[:n_match] - amp_ref[:n_match]))
        phase_mae = circular_mae(phases[:n_match], phase_ref[:n_match])
        amp_maes.append(amp_mae)
        phase_maes.append(phase_mae)

        # 方向图
        if n_out > 0:
            pos_pred = uniform_linear_array_pos(n_out)
            theta = np.linspace(0, 180, 361)
            pat = calculate_1d_pattern(
                pos_pred, amps, phases, theta).numpy()
            sll = get_sll_1d(pat, theta, theta_t, 8.0)

            pat_ref = calculate_1d_pattern(
                pos, amp_ref, phase_ref, theta).numpy()
            sll_ref = get_sll_1d(pat_ref, theta, theta_t, 8.0)
            sll_diff = abs(sll - sll_ref)
            sll_diffs.append(sll_diff)

            print(f"  N={N_t} θ={theta_t}: out={n_out}/{n_expected} "
                  f"amp_mae={amp_mae:.3f} phase_mae={phase_mae:.1f}° "
                  f"SLL={sll:.1f}dB (ref={sll_ref:.1f}, Δ={sll_diff:.1f})")
        else:
            print(f"  N={N_t} θ={theta_t}: out={n_out}/{n_expected} (no output)")

    # 汇总
    print(f"\n{'='*60}")
    print("CLOSED-LOOP VALIDATION SUMMARY")
    print(f"{'='*60}")
    n_cases = len(test_cases)
    if amp_maes:
        print(f"  Amp MAE:        mean={np.mean(amp_maes):.4f}  max={np.max(amp_maes):.4f}")
        print(f"  Phase MAE:      mean={np.mean(phase_maes):.1f}°  max={np.max(phase_maes):.1f}°")
    print(f"  Seq length:     {seq_len_correct}/{n_cases} correct")
    if sll_diffs:
        print(f"  SLL diff:       mean={np.mean(sll_diffs):.1f} dB  max={np.max(sll_diffs):.1f} dB")

    # 管线正确性门槛（不是竞赛指标）
    if amp_maes and np.mean(amp_maes) <= 0.1:
        print("  ✓ Amp MAE ≤ 0.1")
    else:
        print("  ✗ Amp MAE > 0.1 (network needs more training)")

    if amp_maes and np.mean(phase_maes) <= 10:
        print("  ✓ Phase MAE ≤ 10°")
    else:
        print("  ✗ Phase MAE > 10° (network needs more training)")

    print(f"\n{'='*60}")


if __name__ == '__main__':
    main()
