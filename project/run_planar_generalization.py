"""平面阵分布外泛化测试：曲面训练的 DeepSets 模型能否覆盖平面阵。

背景：v2 教师数据 280 个样本全部为曲面（alpha∈[0.08,0.15]，|z|max≥9.7λ），
训练分布中不含任何平面样本。本脚本在理想 32×32 平面阵（pz=0，无扰动）上
直接推理，回答"同一模型统一覆盖平面+曲面"是否成立。

对比口径与训练管线完全一致：
  特征 = (x/8, y/8, z/8, Re(w_taylor)*1024, Im(w_taylor)*1024, u0, v0, w0, 35/50)
  AI 权值 = w_taylor + delta_AI/1024
  评估 = eval_dense_3d（81×81 密集 uv 网格，3×3dB_BW 主瓣排除）

输出: outputs/planar_generalization.json
"""

import os, sys, time, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mylib.antenna_calc import uniform_linear_array_pos, taylor_2d_separable
from mylib.deepsets import DeepSetsModel
from run_curved_verify import coordinate_taylor_3d, eval_dense_3d
from run_generate_teacher import normalize_weights
from run_deepsets_train import _get_null_dirs, COORD_NORM, SLL_NORM, WEIGHT_SCALE

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
ASSETS = os.path.join(OUTPUT_DIR, 'ascend_return_20260831', 'github_991916e7_assets')
MODEL_PATH = os.path.join(ASSETS, 'deepsets_model_v2_256.pt')

NX = NY = 32
SLL = 35
THETA_CHOICES = [0.0, 15.0, 30.0, 45.0, 60.0]
PHI_CHOICES = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]


def main():
    posx_ideal = uniform_linear_array_pos(NX)
    posy_ideal = uniform_linear_array_pos(NY)
    amp_x, amp_y = taylor_2d_separable(NX, NY, SLL)

    px = np.tile(posx_ideal[:, None], (1, NY)).ravel()
    py = np.tile(posy_ideal[None, :], (NX, 1)).ravel()
    pz = np.zeros(NX * NY)

    model = DeepSetsModel(input_dim=9, hidden_dim=256, output_dim=2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    print(f"Model loaded: {MODEL_PATH}")

    results = []
    print("=" * 78)
    print("Planar OOD Generalization Test (ideal 32x32, pz=0)")
    print(f"{'theta':>6} {'phi':>5} | {'SLL_Taylor':>10} {'SLL_AI':>8} {'diff':>7} "
          f"{'|dW|/|W|':>9} {'ptErr':>6}")
    print("-" * 78)

    for theta0 in THETA_CHOICES:
        for phi0 in PHI_CHOICES:
            u0 = np.sin(np.deg2rad(theta0)) * np.cos(np.deg2rad(phi0))
            v0 = np.sin(np.deg2rad(theta0)) * np.sin(np.deg2rad(phi0))
            w0 = np.cos(np.deg2rad(theta0))

            w_taylor = coordinate_taylor_3d(px, py, pz, amp_x, amp_y, theta0, phi0)
            w_taylor = normalize_weights(w_taylor, px, py, pz, u0, v0, w0)

            feat = np.stack([
                px / COORD_NORM, py / COORD_NORM, pz / COORD_NORM,
                w_taylor.real * WEIGHT_SCALE, w_taylor.imag * WEIGHT_SCALE,
                np.full(NX * NY, u0), np.full(NX * NY, v0),
                np.full(NX * NY, w0), np.full(NX * NY, SLL / SLL_NORM),
            ], axis=-1).astype(np.float32)

            with torch.no_grad():
                delta = model(torch.as_tensor(feat[None]))[0].numpy()

            w_ai = w_taylor + (delta[:, 0] + 1j * delta[:, 1]) / WEIGHT_SCALE
            ratio = np.linalg.norm(delta) / (np.linalg.norm(w_taylor) * WEIGHT_SCALE)

            null_dirs = _get_null_dirs(theta0, phi0)
            sll_t, pt_t, _, _ = eval_dense_3d(w_taylor, px, py, pz, theta0, phi0, null_dirs)
            sll_a, pt_a, nd_a, _ = eval_dense_3d(w_ai, px, py, pz, theta0, phi0, null_dirs)

            results.append({
                'theta0': theta0, 'phi0': phi0,
                'sll_taylor': float(sll_t), 'sll_ai': float(sll_a),
                'diff': float(sll_a - sll_t),
                'delta_ratio': float(ratio),
                'point_err_taylor': float(pt_t), 'point_err_ai': float(pt_a),
                'worst_null_ai': float(min(nd_a)) if nd_a else None,
            })
            print(f"{theta0:6.0f} {phi0:5.0f} | {sll_t:10.2f} {sll_a:8.2f} "
                  f"{sll_a - sll_t:+7.2f} {ratio:9.4f} {pt_a:6.2f}")

    sll_t_arr = np.array([r['sll_taylor'] for r in results])
    sll_a_arr = np.array([r['sll_ai'] for r in results])
    diff_arr = np.array([r['diff'] for r in results])

    summary = {
        'model': 'deepsets_model_v2_256.pt (trained on curved only, alpha 0.08-0.15)',
        'test_array': 'ideal 32x32 planar (pz=0), 40 scan directions',
        'n_directions': len(results),
        'sll_taylor': {'mean': float(sll_t_arr.mean()), 'worst': float(sll_t_arr.min()),
                       'best': float(sll_t_arr.max())},
        'sll_ai': {'mean': float(sll_a_arr.mean()), 'worst': float(sll_a_arr.min()),
                   'best': float(sll_a_arr.max())},
        'ai_minus_taylor': {'mean': float(diff_arr.mean()),
                            'worst': float(diff_arr.max()),
                            'n_ai_not_worse': int((diff_arr <= 0).sum())},
        'degradation_within_1dB': int((diff_arr <= 1.0).sum()),
        'results': results,
    }

    print("-" * 78)
    print(f"Taylor: mean={sll_t_arr.mean():.2f} worst={sll_t_arr.min():.2f} dB")
    print(f"AI:     mean={sll_a_arr.mean():.2f} worst={sll_a_arr.min():.2f} dB")
    print(f"AI-Taylor(dB, 正=退化): mean={diff_arr.mean():+.2f} max={diff_arr.max():+.2f}; "
          f"AI不劣于Taylor {int((diff_arr <= 0).sum())}/40 方向; "
          f"退化≤1dB {int((diff_arr <= 1.0).sum())}/40 方向")

    out_path = os.path.join(OUTPUT_DIR, 'planar_generalization.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")

    by_theta = {}
    for th in THETA_CHOICES:
        mask = np.array([r['theta0'] == th for r in results])
        by_theta[str(th)] = {
            'taylor_mean': float(sll_t_arr[mask].mean()),
            'ai_mean': float(sll_a_arr[mask].mean()),
            'degradation_mean': float(diff_arr[mask].mean()),
            'delta_ratio_mean': float(np.array(
                [r['delta_ratio'] for r in results])[mask].mean()),
        }
        print(f"  theta={th:.0f}: Taylor={by_theta[str(th)]['taylor_mean']:.2f} "
              f"AI={by_theta[str(th)]['ai_mean']:.2f} "
              f"退化={by_theta[str(th)]['degradation_mean']:+.2f} dB "
              f"|dW|/|W|={by_theta[str(th)]['delta_ratio_mean']:.3f}")
    summary['by_theta'] = by_theta
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("=" * 78)


if __name__ == '__main__':
    main()
