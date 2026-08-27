"""曲面阵列 + 量化/失效联合实验。

在曲面阵列上测试 Taylor vs SOCP vs AI 在非理想条件下的鲁棒性。

条件:
  1. 理想（无量化无失效）
  2. 0.5dB + 6bit 量化
  3. 5% 阵元失效
  4. 10% 阵元失效
  5. 量化 + 5% 失效

使用 v1 教师标签的测试集样本 + v1 DeepSets 模型。
"""

import os, sys, time, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mylib.deepsets import DeepSetsModel
from mylib.train import get_device
from run_curved_verify import eval_dense_3d
from run_deepsets_train import load_teacher_labels, WEIGHT_SCALE, _get_null_dirs

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
N_TEST = 10
FAILURE_RATES = [0.0, 0.05, 0.10, 0.20]
QUANT_AMP_DB = 0.5
QUANT_PHASE_BITS = 6


def quantize_complex_weights(w, amp_step_db=0.5, phase_bits=6):
    """复权值量化: 幅度0.5dB步进 + 相位6bit。"""
    amp = np.abs(w)
    phase = np.angle(w)
    amp_db_val = 20 * np.log10(np.clip(amp, 1e-10, None))
    amp_q_db = np.round(amp_db_val / amp_step_db) * amp_step_db
    amp_q = 10 ** (amp_q_db / 20)
    step = 2 * np.pi / (2 ** phase_bits)
    phase_q = (np.round(phase / step) * step) % (2 * np.pi)
    return amp_q * np.exp(1j * phase_q)


def apply_failure(w, rate, rng):
    """阵元失效: 权值置零。"""
    if rate <= 0:
        return w.copy()
    n = len(w)
    mask = np.zeros(n, dtype=bool)
    mask[rng.choice(n, int(n * rate), replace=False)] = True
    w_f = w.copy()
    w_f[mask] = 0
    return w_f


def main():
    device = get_device('cpu')

    print("=" * 70)
    print("Curved Array + Quantization/Failure Joint Experiment")
    print(f"  Samples: {N_TEST} from v1 test set")
    print(f"  Failure rates: {FAILURE_RATES}")
    print(f"  Quantization: {QUANT_AMP_DB}dB amp + {QUANT_PHASE_BITS}bit phase")
    print("=" * 70)

    data = load_teacher_labels()
    test_feat, _, test_meta = data['test']

    model_path = os.path.join(OUTPUT_DIR, 'deepsets_model.pt')
    model = DeepSetsModel(input_dim=9, hidden_dim=128, output_dim=2)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"  Model loaded: {model_path}")
    else:
        print(f"  ERROR: Model not found at {model_path}")
        return
    model = model.to(device)
    model.eval()

    rng = np.random.RandomState(99)
    results = []

    for i in range(N_TEST):
        px = test_meta['px'][i]
        py = test_meta['py'][i]
        pz = test_meta['pz'][i]
        th0 = float(test_meta['theta0'][i])
        ph0 = float(test_meta['phi0'][i])
        null_dirs = _get_null_dirs(th0, ph0)

        w_taylor = test_meta['w_taylor_re'][i] + 1j * test_meta['w_taylor_im'][i]
        w_socp = test_meta['w_socp_re'][i] + 1j * test_meta['w_socp_im'][i]

        with torch.no_grad():
            x = torch.as_tensor(test_feat[i:i+1], dtype=torch.float32, device=device)
            delta = model(x)[0].cpu().numpy()
        w_ai = (test_meta['w_taylor_re'][i] + delta[:, 0] / WEIGHT_SCALE) + \
               1j * (test_meta['w_taylor_im'][i] + delta[:, 1] / WEIGHT_SCALE)

        for rate in FAILURE_RATES:
            for use_quant in [False, True]:
                for name, w in [('Taylor', w_taylor), ('SOCP', w_socp), ('AI', w_ai)]:
                    w_mod = w.copy()
                    if use_quant:
                        w_mod = quantize_complex_weights(w_mod)
                    w_mod = apply_failure(w_mod, rate, rng)

                    sll, _, _, _ = eval_dense_3d(
                        w_mod, px, py, pz, th0, ph0, null_dirs)
                    sll_val = float(sll) if not np.isnan(sll) else -100.0

                    results.append({
                        'sample': i, 'method': name,
                        'failure_rate': rate, 'quantized': use_quant,
                        'sll': sll_val,
                    })

        if (i + 1) % 2 == 0:
            print(f"  Sample {i+1}/{N_TEST} done")

    print(f"\n{'='*70}")
    print("RESULTS: SLL (dB) by method and condition")
    print(f"{'='*70}")
    print(f"{'Condition':<25} {'Taylor':>10} {'SOCP':>10} {'AI':>10} {'AI-Taylor':>10}")
    print("-" * 65)

    for rate in FAILURE_RATES:
        for use_quant in [False, True]:
            label = f"fail={rate:.0%}"
            if use_quant:
                label += " + quant"
            else:
                label += " only"

            subset = [r for r in results
                      if r['failure_rate'] == rate and r['quantized'] == use_quant]
            t_vals = [r['sll'] for r in subset if r['method'] == 'Taylor']
            s_vals = [r['sll'] for r in subset if r['method'] == 'SOCP']
            a_vals = [r['sll'] for r in subset if r['method'] == 'AI']

            t_mean = np.mean(t_vals)
            s_mean = np.mean(s_vals)
            a_mean = np.mean(a_vals)

            print(f"{label:<25} {t_mean:>10.1f} {s_mean:>10.1f} {a_mean:>10.1f} "
                  f"{a_mean - t_mean:>+10.1f}")

    with open(os.path.join(OUTPUT_DIR, 'curved_nonideal.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved: {os.path.join(OUTPUT_DIR, 'curved_nonideal.json')}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
