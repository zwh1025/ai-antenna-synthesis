"""NPU 全规模训练（分阶段执行）。

13M 模型 + 13776 样本 + batch=128 + 300 epochs
预计 ~15 分钟（NPU 0.81s/epoch @ batch=32）
"""

import os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mylib.dataset import create_dataset, prepare_training_data, get_dataset_config
from mylib.models import Seq2SeqModel, count_parameters, predict_sequence
from mylib.train import train_model, evaluate_model, get_device
from mylib.antenna_calc import (
    uniform_linear_array_pos, taylor_excitation, beam_steering_phase,
    calculate_1d_pattern, get_sll_1d,
)

output_dir = os.path.join(os.path.dirname(__file__), 'outputs')

def main():
    os.makedirs(output_dir, exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)
    dev = get_device()

    # 全规模数据集
    N_list = np.arange(15, 31, 1)
    theta0_list = np.arange(30, 151, 3)
    SLL_list = np.arange(10, 31, 1)
    n = len(N_list) * len(theta0_list) * len(SLL_list)
    print(f"\nDataset: {n} samples")
    X, Y = create_dataset(N_list, theta0_list, SLL_list, reference='Taylor')
    enc_in, dec_in, dec_out = prepare_training_data(X, Y)
    config = get_dataset_config(N_list)

    # 13M 模型
    main_n = [512, 512]; branch_n = [256, 256, 128, 128]; dense_n = [64, 32, 16]
    model = Seq2SeqModel(32, 32, [main_n, branch_n, branch_n, dense_n, dense_n])
    print(f"Model: {count_parameters(model):,} params")

    # 加载之前的 checkpoint（如果有）
    ckpt = os.path.join(output_dir, 'model_full_npu.pt')
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=False))
        print(f"Loaded checkpoint: {ckpt}")

    print(f"Training: 120 epochs, batch=128, lr=5e-4")
    save_path = os.path.join(output_dir, 'model_final.pt')
    t0 = time.time()
    model, history = train_model(
        model, enc_in, dec_in, dec_out,
        batch_size=128, epochs=120, learning_rate=5e-4,
        patience_lr=5, patience_stop=30, verbose=True, device=dev,
        save_path=save_path, save_every=30)
    t1 = time.time()
    n_ep = len(history['loss'])
    print(f"\nDone: {n_ep} epochs in {t1-t0:.0f}s ({(t1-t0)/n_ep:.2f}s/epoch)")
    print(f"Best acc: {max(history['accuracy']):.4f}")

    torch.save(model.state_dict(), os.path.join(output_dir, 'model_final.pt'))
    np.savez(os.path.join(output_dir, 'history_final.npz'),
             **{k: np.array(v) for k, v in history.items()})

    # 1D 测试
    print(f"\n=== 1D Test ===")
    amp_errs = []; phase_errs = []; slls = []
    for N_t in [16, 20, 25]:
        for th_t in [45, 60, 75, 90, 105, 120, 135]:
            X_t, _ = create_dataset([N_t], [th_t], [30], reference='Taylor')
            inp = torch.as_tensor(X_t[:1], dtype=torch.float32, device=dev)
            gen, _ = predict_sequence(model, inp, 32, (0.,1.), (0.,2*np.pi), max_steps=35)
            if len(gen) < N_t:
                continue
            ap = np.array([g[0] for g in gen[:N_t]])
            pp = np.array([g[1] for g in gen[:N_t]])
            pos = uniform_linear_array_pos(N_t)
            ar = taylor_excitation(N_t*0.5, pos, 30)
            pr = beam_steering_phase(pos, th_t) % (2*np.pi)
            amp_errs.extend(np.abs(ap-ar).tolist())
            pe = np.abs(pp-pr); pe = np.minimum(pe, 2*np.pi-pe)
            phase_errs.extend(pe.tolist())
            theta = np.linspace(0, 180, 361)
            pat = calculate_1d_pattern(pos, ap, pp, theta).numpy()
            slls.append(get_sll_1d(pat, theta, th_t, 8.0))

    ae = np.array(amp_errs); pe = np.degrees(np.array(phase_errs))
    print(f"  Cases: {len(slls)}")
    print(f"  Amp err:   mean={ae.mean():.4f} max={ae.max():.4f}")
    print(f"  Phase err: mean={pe.mean():.2f}° max={pe.max():.2f}°")
    print(f"  1D SLL:    mean={np.mean(slls):.1f} range=[{np.min(slls):.1f}, {np.max(slls):.1f}]")

if __name__ == '__main__':
    main()
