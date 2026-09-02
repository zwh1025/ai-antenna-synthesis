"""平面样本增广修复训练：解决 v2 模型在平面阵上的捷径学习。

问题：v2 训练集（280 曲面样本）中扫描角完美预测修正需求——
  theta<=15: SOCP 对 Taylor 零改善（dW=0）
  theta>=30: 改善 3.5-4 dB（|dW|/|W|~=0.65）
模型学到 "theta>=30 就输出大修正" 的捷径，忽略 z 坐标，
导致平面阵 theta>=30 推理时输出伪修正（退化 10-16 dB）。

修复：加入平面样本（alpha=0），目标 dW=0（平面阵 Taylor 已最优，
无需 SOCP 求解，数据生成免费）。平面样本覆盖 theta=0-60 全角度，
打破 "扫描角 -> 修正量" 的伪相关，迫使模型依据 z 坐标判断。

数据：曲面 280（原始 v2）+ 平面 120（新）= 400 样本
输出：outputs/teacher_labels_v3.npz, outputs/deepsets_model_v3_256.pt
验证：平面 40 方向 OOD + 曲面测试集（对照 v2 基线）
"""

import os, sys, time, json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mylib.antenna_calc import uniform_linear_array_pos, taylor_2d_separable
from mylib.deepsets import DeepSetsModel, count_parameters
from mylib.train import EarlyStopping
from run_curved_verify import generate_curved_array, coordinate_taylor_3d, eval_dense_3d
from run_generate_teacher import normalize_weights
from run_deepsets_train import _get_null_dirs

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
ASSETS = os.path.join(OUTPUT_DIR, 'ascend_return_20260831', 'github_991916e7_assets')
V2_DATA = os.path.join(ASSETS, 'teacher_labels_v2.npz')

NX = NY = 32
SLL = 35
COORD_NORM = 8.0
SLL_NORM = 50.0
WEIGHT_SCALE = 1024.0

THETA_CHOICES = [0.0, 15.0, 30.0, 45.0, 60.0]
PHI_CHOICES = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]

N_PLANAR = 120
SEED = 456
EPOCHS = 300
BATCH = 16
LR = 1e-3
HIDDEN = 256


def generate_planar_samples(rng, posx_ideal, posy_ideal, amp_x, amp_y):
    """平面样本：pz=0（带小扰动多样性），目标 dW=0。"""
    samples = []
    combos = [(t, p) for t in THETA_CHOICES for p in PHI_CHOICES]
    for i in range(N_PLANAR):
        theta0, phi0 = combos[i % len(combos)]
        theta0 = float(theta0)
        phi0 = float(phi0) + rng.uniform(-20, 20)

        px = np.tile(posx_ideal[:, None], (1, NY))
        py = np.tile(posy_ideal[None, :], (NX, 1))
        px = (px + rng.uniform(-0.02, 0.02, (NX, NY))).ravel()
        py = (py + rng.uniform(-0.02, 0.02, (NX, NY))).ravel()
        pz = np.zeros(NX * NY)

        u0 = np.sin(np.deg2rad(theta0)) * np.cos(np.deg2rad(phi0))
        v0 = np.sin(np.deg2rad(theta0)) * np.sin(np.deg2rad(phi0))
        w0 = np.cos(np.deg2rad(theta0))

        w_taylor = coordinate_taylor_3d(px, py, pz, amp_x, amp_y, theta0, phi0)
        w_taylor = normalize_weights(w_taylor, px, py, pz, u0, v0, w0)

        null_dirs = _get_null_dirs(theta0, phi0)
        sll_t, _, _, _ = eval_dense_3d(w_taylor, px, py, pz, theta0, phi0, null_dirs)

        samples.append({
            'px': px, 'py': py, 'pz': pz,
            'w_taylor_re': w_taylor.real, 'w_taylor_im': w_taylor.imag,
            'w_socp_re': w_taylor.real, 'w_socp_im': w_taylor.imag,
            'sll_taylor': float(sll_t), 'sll_socp': float(sll_t),
            'alpha': 0.0,
            'theta0': theta0, 'phi0': phi0 % 360,
            'u0': float(u0), 'v0': float(v0), 'w0': float(w0),
        })
    return samples


def make_features(px, py, pz, w_re, w_im, u0, v0, w0):
    feat = np.stack([
        np.asarray(px, dtype=np.float64) / COORD_NORM,
        np.asarray(py, dtype=np.float64) / COORD_NORM,
        np.asarray(pz, dtype=np.float64) / COORD_NORM,
        np.asarray(w_re, dtype=np.float64) * WEIGHT_SCALE,
        np.asarray(w_im, dtype=np.float64) * WEIGHT_SCALE,
        np.broadcast_to(np.float64(u0), np.shape(px)),
        np.broadcast_to(np.float64(v0), np.shape(px)),
        np.broadcast_to(np.float64(w0), np.shape(px)),
        np.full(np.shape(px), SLL / SLL_NORM),
    ], axis=-1).astype(np.float32)
    return feat


def build_dataset():
    v2 = np.load(V2_DATA)
    curved = {k: v2[k] for k in ['px', 'py', 'pz', 'w_taylor_re', 'w_taylor_im',
                                  'w_socp_re', 'w_socp_im', 'sll_taylor', 'sll_socp',
                                  'alpha', 'theta0', 'phi0', 'u0', 'v0', 'w0']}
    n_curved = len(curved['theta0'])

    posx_ideal = uniform_linear_array_pos(NX)
    posy_ideal = uniform_linear_array_pos(NY)
    amp_x, amp_y = taylor_2d_separable(NX, NY, SLL)
    rng = np.random.RandomState(SEED)
    planar = generate_planar_samples(rng, posx_ideal, posy_ideal, amp_x, amp_y)

    merged = {}
    for k in curved:
        merged[k] = np.concatenate(
            [curved[k]] + [np.array([s[k] for s in planar])], axis=0)
    merged['split'] = np.concatenate([
        v2['split'], np.full(N_PLANAR, 0, dtype=np.int32)])
    merged['n_elements'] = NX * NY
    merged['version'] = 'v3_planar_augmented'

    return merged, n_curved


def train(model, feat, target, val_feat, val_target, save_path):
    model = model.to('cpu')
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=8)
    stop = EarlyStopping(patience=30)
    crit = nn.MSELoss()

    ds = TensorDataset(torch.as_tensor(feat), torch.as_tensor(target))
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True)
    vx = torch.as_tensor(val_feat)
    vy = torch.as_tensor(val_target)

    best = float('inf')
    t0 = time.time()
    for epoch in range(EPOCHS):
        model.train()
        tot, nb = 0.0, 0
        for bx, by in loader:
            opt.zero_grad()
            loss = crit(model(bx), by)
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        model.eval()
        with torch.no_grad():
            vl = crit(model(vx), vy).item()
        sched.step(vl)
        if vl < best:
            best = vl
            torch.save(model.state_dict(), save_path)
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  ep{epoch+1:3d}: loss={tot/nb:.6f} val={vl:.6f} lr={opt.param_groups[0]['lr']:.1e}")
        if stop.step(vl):
            print(f"  early stop at ep{epoch+1} (best val={best:.6f})")
            break
    return time.time() - t0, best


def eval_planar(model, posx_ideal, posy_ideal, amp_x, amp_y):
    """理想平面阵 40 方向 OOD 评估（与 run_planar_generalization 同口径）。"""
    model.eval()
    px = np.tile(posx_ideal[:, None], (1, NY)).ravel()
    py = np.tile(posy_ideal[None, :], (NX, 1)).ravel()
    pz = np.zeros(NX * NY)

    by_theta = {t: [] for t in THETA_CHOICES}
    all_diff = []
    for theta0 in THETA_CHOICES:
        for phi0 in PHI_CHOICES:
            u0 = np.sin(np.deg2rad(theta0)) * np.cos(np.deg2rad(phi0))
            v0 = np.sin(np.deg2rad(theta0)) * np.sin(np.deg2rad(phi0))
            w0 = np.cos(np.deg2rad(theta0))

            w_t = coordinate_taylor_3d(px, py, pz, amp_x, amp_y, theta0, phi0)
            w_t = normalize_weights(w_t, px, py, pz, u0, v0, w0)

            feat = make_features(px, py, pz, w_t.real, w_t.imag, u0, v0, w0)
            with torch.no_grad():
                delta = model(torch.as_tensor(feat[None]))[0].numpy()
            w_ai = w_t + (delta[:, 0] + 1j * delta[:, 1]) / WEIGHT_SCALE

            null_dirs = _get_null_dirs(theta0, phi0)
            sll_t, _, _, _ = eval_dense_3d(w_t, px, py, pz, theta0, phi0, null_dirs)
            sll_a, _, _, _ = eval_dense_3d(w_ai, px, py, pz, theta0, phi0, null_dirs)
            by_theta[theta0].append(sll_a - sll_t)
            all_diff.append(sll_a - sll_t)

    stats = {f"theta={t:.0f}": {
        'taylor': None,
        'degradation_mean': float(np.mean(v)),
        'degradation_max': float(np.max(v)),
        'degradation_within_1dB': int(np.sum(np.array(v) <= 1.0)),
    } for t, v in by_theta.items()}
    return stats, float(np.mean(all_diff)), float(np.max(all_diff))


def eval_curved_test(model, data):
    """曲面测试集（v2 原始 50 样本）恢复率评估。"""
    model.eval()
    split = data['split']
    idx = np.where(split == 2)[0]
    sll_ai, sll_t, sll_s = [], [], []
    for i in idx:
        px, py, pz = data['px'][i], data['py'][i], data['pz'][i]
        w_re, w_im = data['w_taylor_re'][i], data['w_taylor_im'][i]
        feat = make_features(px, py, pz, w_re, w_im,
                             data['u0'][i], data['v0'][i], data['w0'][i])
        with torch.no_grad():
            delta = model(torch.as_tensor(feat[None]))[0].numpy()
        w_ai = (w_re + 1j * w_im) + (delta[:, 0] + 1j * delta[:, 1]) / WEIGHT_SCALE

        th0, ph0 = float(data['theta0'][i]), float(data['phi0'][i])
        null_dirs = _get_null_dirs(th0, ph0)
        sll, _, _, _ = eval_dense_3d(w_ai, px, py, pz, th0, ph0, null_dirs)
        sll_ai.append(sll)
        sll_t.append(data['sll_taylor'][i])
        sll_s.append(data['sll_socp'][i])

    t_m, s_m, a_m = np.mean(sll_t), np.mean(sll_s), np.mean(sll_ai)
    recovery = (a_m - t_m) / (s_m - t_m) * 100 if abs(s_m - t_m) > 0.01 else 0.0
    return {'taylor_mean': float(t_m), 'socp_mean': float(s_m),
            'ai_mean': float(a_m), 'recovery_pct': float(recovery)}


def main():
    print("=" * 78)
    print("v3: Planar-Augmented DeepSets Training (fix shortcut learning)")
    print("=" * 78)

    data, n_curved = build_dataset()
    split = data['split']
    tr_idx, va_idx, te_idx = np.where(split == 0)[0], np.where(split == 1)[0], np.where(split == 2)[0]
    print(f"Dataset: {len(split)} samples = {n_curved} curved (v2) + {N_PLANAR} planar")
    print(f"Train {len(tr_idx)} / Val {len(va_idx)} / Test {len(te_idx)}")

    def pack(idx):
        px, py, pz = data['px'][idx], data['py'][idx], data['pz'][idx]
        w_re, w_im = data['w_taylor_re'][idx], data['w_taylor_im'][idx]
        feats, targets = [], []
        for j in range(len(idx)):
            feats.append(make_features(px[j], py[j], pz[j], w_re[j], w_im[j],
                                       data['u0'][idx[j]], data['v0'][idx[j]],
                                       data['w0'][idx[j]]))
            d_re = (data['w_socp_re'][idx][j] - w_re[j]) * WEIGHT_SCALE
            d_im = (data['w_socp_im'][idx][j] - w_im[j]) * WEIGHT_SCALE
            targets.append(np.stack([d_re, d_im], axis=-1).astype(np.float32))
        return np.array(feats), np.array(targets)

    tr_f, tr_y = pack(tr_idx)
    va_f, va_y = pack(va_idx)
    print(f"Features: {tr_f.shape}, targets: {tr_y.shape}")

    model = DeepSetsModel(input_dim=9, hidden_dim=HIDDEN, output_dim=2)
    print(f"Model: {count_parameters(model):,} params (hidden={HIDDEN})")
    save_path = os.path.join(OUTPUT_DIR, 'deepsets_model_v3_256.pt')
    t_train, best_vl = train(model, tr_f, tr_y, va_f, va_y, save_path)
    print(f"\nTraining: {t_train:.0f}s, best val loss={best_vl:.6f}")

    model.load_state_dict(torch.load(save_path, map_location='cpu', weights_only=True))

    posx_ideal = uniform_linear_array_pos(NX)
    posy_ideal = uniform_linear_array_pos(NY)
    amp_x, amp_y = taylor_2d_separable(NX, NY, SLL)

    print("\n--- Planar OOD (ideal 32x32, 40 directions) ---")
    p_stats, p_mean, p_max = eval_planar(model, posx_ideal, posy_ideal, amp_x, amp_y)
    for k, v in p_stats.items():
        print(f"  {k}: 退化 mean={v['degradation_mean']:+.2f} dB "
              f"max={v['degradation_max']:+.2f} dB "
              f"({v['degradation_within_1dB']}/8 方向 ≤1dB)")
    print(f"  Overall: mean={p_mean:+.2f} dB, max={p_max:+.2f} dB")

    print("\n--- Curved test set (50 original v2 samples) ---")
    c_stats = eval_curved_test(model, data)
    print(f"  Taylor: {c_stats['taylor_mean']:.2f} dB")
    print(f"  SOCP:   {c_stats['socp_mean']:.2f} dB")
    print(f"  AI:     {c_stats['ai_mean']:.2f} dB")
    print(f"  Recovery: {c_stats['recovery_pct']:.1f}% (v2 模型为 82.4%)")

    out = {
        'description': 'v3 planar-augmented model, fixes shortcut learning',
        'dataset': {'curved': int(n_curved), 'planar_added': N_PLANAR,
                    'seed': SEED},
        'training': {'epochs_max': EPOCHS, 'batch': BATCH, 'lr': LR,
                     'hidden': HIDDEN, 'time_s': float(t_train),
                     'best_val_loss': float(best_vl)},
        'planar_ood': {'by_theta': p_stats, 'degradation_mean': p_mean,
                       'degradation_max': p_max},
        'curved_test': c_stats,
        'reference': {'v2_planar_degradation': 'theta>=30: +10.2~+15.7 dB',
                      'v2_curved_recovery_pct': 82.4},
    }
    out_path = os.path.join(OUTPUT_DIR, 'planar_fix_v3.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    data_path = os.path.join(OUTPUT_DIR, 'teacher_labels_v3.npz')
    save_dict = {k: v for k, v in data.items() if k != 'version'}
    save_dict['version'] = 'v3_planar_augmented'
    np.savez(data_path, **save_dict)
    print(f"\nSaved: {data_path}")
    print(f"Saved: {out_path}")
    print(f"Saved: {save_path}")
    print("=" * 78)


if __name__ == '__main__':
    main()
