"""v4 规模增广训练：修复 v3 在 64×64 曲面上的规模外推失效。

背景（run_scale_4096.py 实测）：
  - v3 平面 64×64 泛化近乎完美（语义缩放最差 +0.32 dB）；
  - v3 曲面 64×64 全面退化（+2.2 至 +14.3 dB）：修正量与训练孔径耦合。

修复：混合规模训练——
  既有 400 个 1024 样本（280曲面+120平面，特征 scale=N=1024 与原训练完全
  等价）+ 新生成 100 个 4096 样本（60曲面 SOCP 教师 + 40平面零成本），
  4096 样本特征 scale=N=4096（相对幅度语义）。

4096 曲面 SOCP 教师生成：α~U[0.02,0.15]（覆盖"匹配口径"到"同α"全部
曲率域）、θ∈{0,15,30,45,60}、φ 随机；切平面网格与轮数沿用训练配方，
主瓣排除用 nx=64 的波束宽度（原函数硬编码 32）。

断点续跑：分阶段落盘（教师生成 → 训练 → 评估），重跑自动跳过已完成阶段。

输出: outputs/teacher_labels_4096.npz, outputs/deepsets_model_v4_256.pt,
      outputs/scale_fix_v4.json
"""

import os, sys, time, json
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mylib.antenna_calc import uniform_linear_array_pos, taylor_2d_separable
from mylib.deepsets import DeepSetsModel, count_parameters
from mylib.train import EarlyStopping
from run_curved_verify import (
    coordinate_taylor_3d, eval_dense_3d, uv_to_uvw, solve_socp_3d,
)
from run_deepsets_train import _get_null_dirs
from run_scale_4096 import (
    _normalize_weights_torch, model_predict as _mp_unused, eval_case as _ec,
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
MODEL_V3 = os.path.join(OUTPUT_DIR, 'deepsets_model_v3_256.pt')
TEACHER_4096 = os.path.join(OUTPUT_DIR, 'teacher_labels_4096.npz')
MODEL_V4 = os.path.join(OUTPUT_DIR, 'deepsets_model_v4_256.pt')

COORD_NORM = 8.0
SLL_NORM = 50.0
HIDDEN = 256
EPOCHS = 200
LR = 1e-3

N_CURVED_4096 = 60
N_PLANAR_4096 = 40
ALPHA_LO, ALPHA_HI = 0.02, 0.15
THETA_CHOICES = [0.0, 15.0, 30.0, 45.0, 60.0]
N_COARSE_GRID = 15
N_CUTTING_ITERS = 5
SEED = 789

NX = NY = 64


def socp_cutting_4096(px, py, pz, theta0, phi0, null_dirs,
                      sll_taylor, w_taylor_n):
    """切平面 SOCP（与训练配方一致，主瓣排除按 nx=64 波束宽度）。"""
    null_u = [np.sin(np.deg2rad(t)) * np.cos(np.deg2rad(p))
              for t, p in null_dirs]
    null_v = [np.sin(np.deg2rad(t)) * np.sin(np.deg2rad(p))
              for t, p in null_dirs]
    null_w = [np.cos(np.deg2rad(t)) for t, p in null_dirs]

    u0 = np.sin(np.deg2rad(theta0)) * np.cos(np.deg2rad(phi0))
    v0 = np.sin(np.deg2rad(theta0)) * np.sin(np.deg2rad(phi0))
    w0 = np.cos(np.deg2rad(theta0))

    u_c = np.linspace(-1, 1, N_COARSE_GRID)
    v_c = np.linspace(-1, 1, N_COARSE_GRID)
    ug, vg = np.meshgrid(u_c, v_c, indexing='ij')
    vis = (ug ** 2 + vg ** 2) <= 1.0
    wg = uv_to_uvw(ug, vg)
    bw = 0.886 * 2.0 / NX * 180 / np.pi          # nx=64
    exc = np.sin(np.deg2rad(3.0 * bw / max(np.cos(np.deg2rad(theta0)), 0.1)))
    dist = np.sqrt((ug - u0) ** 2 + (vg - v0) ** 2)
    m = (dist >= exc) & vis
    sl_u, sl_v, sl_w = list(ug[m]), list(vg[m]), list(wg[m])

    best_sll, best_w = sll_taylor, w_taylor_n.copy()
    for _ in range(N_CUTTING_ITERS):
        w_opt = solve_socp_3d(px, py, pz, u0, v0, w0, sl_u, sl_v, sl_w,
                              (null_u, null_v), null_w)
        if w_opt is None:
            break
        w_opt = _normalize_weights_torch(w_opt, px, py, pz, u0, v0, w0)
        sll, _, _, worst = eval_dense_3d(w_opt, px, py, pz, theta0, phi0,
                                         null_dirs)
        if not np.isnan(sll) and sll < best_sll - 0.05:
            best_sll, best_w = sll, w_opt.copy()
        for uw, vw, ww in worst:
            if not any(abs(su - uw) < 0.01 and abs(sv - vw) < 0.01
                       for su, sv in zip(sl_u, sl_v)):
                sl_u.append(uw)
                sl_v.append(vw)
                sl_w.append(ww)
    return best_w, best_sll


def stage1_generate_teacher_4096():
    """生成 60 个 4096 曲面 SOCP 教师 + 40 个 4096 平面样本。"""
    if os.path.exists(TEACHER_4096):
        print(f'[stage1] 已存在, 跳过: {TEACHER_4096}')
        return
    posx = uniform_linear_array_pos(NX)
    posy = uniform_linear_array_pos(NY)
    amp_x, amp_y = taylor_2d_separable(NX, NY, 35)
    rng = np.random.RandomState(SEED)

    samples = []
    n_total = N_CURVED_4096 + N_PLANAR_4096
    t0 = time.time()
    for i in range(n_total):
        planar = i >= N_CURVED_4096
        alpha = 0.0 if planar else rng.uniform(ALPHA_LO, ALPHA_HI)
        theta0 = float(rng.choice(THETA_CHOICES))
        phi0 = float(rng.uniform(0, 360))

        px = np.tile(posx[:, None], (1, NY))
        py = np.tile(posy[None, :], (NX, 1))
        px = (px + rng.uniform(-0.02, 0.02, (NX, NY))).ravel()
        py = (py + rng.uniform(-0.02, 0.02, (NX, NY))).ravel()
        pz = (alpha * (px ** 2 + py ** 2) if not planar
              else np.zeros(NX * NY))

        u0 = np.sin(np.deg2rad(theta0)) * np.cos(np.deg2rad(phi0))
        v0 = np.sin(np.deg2rad(theta0)) * np.sin(np.deg2rad(phi0))
        w0 = np.cos(np.deg2rad(theta0))
        w_t = coordinate_taylor_3d(px, py, pz, amp_x, amp_y, theta0, phi0)
        w_t = _normalize_weights_torch(w_t, px, py, pz, u0, v0, w0)
        null_dirs = _get_null_dirs(theta0, phi0)
        sll_t, _, _, _ = eval_dense_3d(w_t, px, py, pz, theta0, phi0, null_dirs)

        if planar:
            w_s, sll_s = w_t, sll_t
        else:
            w_s, sll_s = socp_cutting_4096(px, py, pz, theta0, phi0,
                                           null_dirs, sll_t, w_t)
        samples.append({
            'px': px, 'py': py, 'pz': pz,
            'w_taylor_re': w_t.real, 'w_taylor_im': w_t.imag,
            'w_socp_re': w_s.real, 'w_socp_im': w_s.imag,
            'sll_taylor': float(sll_t), 'sll_socp': float(sll_s),
            'alpha': float(alpha), 'theta0': theta0, 'phi0': phi0,
            'u0': float(u0), 'v0': float(v0), 'w0': float(w0),
        })
        if (i + 1) % 5 == 0 or i == 0:
            el = time.time() - t0
            print(f'[stage1] {i+1}/{n_total} a={alpha:.3f} th={theta0:.0f} '
                  f'T={sll_t:.1f} S={sll_s:.1f} '
                  f'({el:.0f}s, ETA {(n_total-i-1)*el/(i+1):.0f}s)', flush=True)

    d = {'n_elements': NX * NY, 'n_curved': N_CURVED_4096,
         'n_planar': N_PLANAR_4096, 'seed': SEED}
    for k in ['px', 'py', 'pz', 'w_taylor_re', 'w_taylor_im',
              'w_socp_re', 'w_socp_im']:
        d[k] = np.array([s[k] for s in samples])
    for k in ['sll_taylor', 'sll_socp', 'alpha', 'theta0', 'phi0',
              'u0', 'v0', 'w0']:
        d[k] = np.array([s[k] for s in samples])
    np.savez(TEACHER_4096, **d)
    print(f'[stage1] saved: {TEACHER_4096}')


def _features(px, py, pz, w_re, w_im, u0, v0, w0, scale):
    n = len(px)
    return np.stack([
        px / COORD_NORM, py / COORD_NORM, pz / COORD_NORM,
        w_re * scale, w_im * scale,
        np.full(n, u0), np.full(n, v0), np.full(n, w0),
        np.full(n, 35.0 / SLL_NORM),
    ], axis=-1).astype(np.float32)


def _extra_planar(n_per_size):
    """定向补平面样本: theta∈{45,60}(带少量15/30), 覆盖全phi, 零成本。
    修复 v4 在 1024 平面 theta=60 个别方向的伪修正回退。"""
    out = {}
    for size, n in n_per_size.items():
        nx = int(round(size ** 0.5))
        posx = uniform_linear_array_pos(nx)
        posy = uniform_linear_array_pos(nx)
        ax, ay = taylor_2d_separable(nx, nx, 35)
        rng = np.random.RandomState(1000 + nx)
        rows = []
        thetas = ([45.0] * (n // 4) + [60.0] * (n // 2) +
                  [15.0] * (n // 8) + [30.0] * (n - n // 4 - n // 2 - n // 8))
        for i in range(n):
            theta0 = float(thetas[i % len(thetas)]) + rng.uniform(-5, 5)
            theta0 = min(max(theta0, 0.0), 60.0)
            phi0 = float(rng.uniform(0, 360))
            px = np.tile(posx[:, None], (1, nx))
            py = np.tile(posy[None, :], (nx, 1))
            px = (px + rng.uniform(-0.02, 0.02, (nx, nx))).ravel()
            py = (py + rng.uniform(-0.02, 0.02, (nx, nx))).ravel()
            pz = np.zeros(nx * nx)
            u0 = np.sin(np.deg2rad(theta0)) * np.cos(np.deg2rad(phi0))
            v0 = np.sin(np.deg2rad(theta0)) * np.sin(np.deg2rad(phi0))
            w0 = np.cos(np.deg2rad(theta0))
            w_t = coordinate_taylor_3d(px, py, pz, ax, ay, theta0, phi0)
            w_t = _normalize_weights_torch(w_t, px, py, pz, u0, v0, w0)
            rows.append((px, py, pz, w_t, u0, v0, w0))
        out[size] = rows
    return out


def stage2_train():
    if os.path.exists(MODEL_V4):
        print(f'[stage2] 已存在, 跳过: {MODEL_V4}')
        return
    v2 = np.load(os.path.join(OUTPUT_DIR, 'ascend_return_20260831',
                              'github_991916e7_assets', 'teacher_labels_v2.npz'))
    t4 = np.load(TEACHER_4096)

    extra = _extra_planar({1024: 60, 4096: 60})

    # 1024 组: scale=N=1024 与原训练特征完全等价
    groups = []
    feat, tgt = [], []
    for j in range(len(v2['theta0'])):
        w_re, w_im = v2['w_taylor_re'][j], v2['w_taylor_im'][j]
        d_re = (v2['w_socp_re'][j] - w_re) * 1024.0
        d_im = (v2['w_socp_im'][j] - w_im) * 1024.0
        feat.append(_features(v2['px'][j], v2['py'][j], v2['pz'][j],
                              w_re, w_im, v2['u0'][j], v2['v0'][j],
                              v2['w0'][j], 1024.0))
        tgt.append(np.stack([d_re, d_im], axis=-1).astype(np.float32))
    for px, py, pz, w_t, u0, v0, w0 in extra[1024]:
        feat.append(_features(px, py, pz, w_t.real, w_t.imag, u0, v0, w0,
                              1024.0))
        tgt.append(np.zeros((len(px), 2), dtype=np.float32))
    groups.append(('1024', np.array(feat), np.array(tgt)))

    feat, tgt = [], []
    for j in range(len(t4['theta0'])):
        w_re, w_im = t4['w_taylor_re'][j], t4['w_taylor_im'][j]
        d_re = (t4['w_socp_re'][j] - w_re) * float(NX * NY)
        d_im = (t4['w_socp_im'][j] - w_im) * float(NX * NY)
        feat.append(_features(t4['px'][j], t4['py'][j], t4['pz'][j],
                              w_re, w_im, t4['u0'][j], t4['v0'][j],
                              t4['w0'][j], float(NX * NY)))
        tgt.append(np.stack([d_re, d_im], axis=-1).astype(np.float32))
    for px, py, pz, w_t, u0, v0, w0 in extra[4096]:
        feat.append(_features(px, py, pz, w_t.real, w_t.imag, u0, v0, w0,
                              float(NX * NY)))
        tgt.append(np.zeros((len(px), 2), dtype=np.float32))
    groups.append(('4096', np.array(feat), np.array(tgt)))

    # 验证集: 1024 原验证 30 个
    split = v2['split']
    va_idx = np.where(split == 1)[0]
    va_feat = np.array([groups[0][1][i] for i in va_idx if i < len(groups[0][1])])
    va_tgt = np.array([groups[0][2][i] for i in va_idx if i < len(groups[0][2])])

    model = DeepSetsModel(input_dim=9, hidden_dim=HIDDEN, output_dim=2)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min',
                                                       factor=0.5, patience=8)
    stop = EarlyStopping(patience=25)
    crit = nn.MSELoss()
    vx = torch.as_tensor(va_feat)
    vy = torch.as_tensor(va_tgt)

    print(f'[stage2] train: 460x1024(400+60planar) + '
          f'{len(groups[1][1])}x4096(100+60planar); '
          f'val: {len(va_feat)}x1024; params={count_parameters(model):,}',
          flush=True)
    t0 = time.time()
    best = float('inf')
    for epoch in range(EPOCHS):
        model.train()
        tot, nb = 0.0, 0
        for name, F, T in groups:
            perm = np.random.permutation(len(F))
            bs = 16 if name == '1024' else 6
            for s in range(0, len(F), bs):
                idx = perm[s:s + bs]
                bx = torch.as_tensor(F[idx])
                by = torch.as_tensor(T[idx])
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
            torch.save(model.state_dict(), MODEL_V4)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'  ep{epoch+1:3d}: loss={tot/nb:.6f} val={vl:.6f} '
                  f'({time.time()-t0:.0f}s)', flush=True)
        if stop.step(vl):
            print(f'  early stop ep{epoch+1} (best {best:.6f})', flush=True)
            break
    print(f'[stage2] saved: {MODEL_V4} (best val {best:.6f})')


def model_predict_v4(model, px, py, pz, amp_x, amp_y, theta0, phi0):
    """v4 推理: scale=N 语义缩放。"""
    scale = float(len(px))
    w_t = coordinate_taylor_3d(px, py, pz, amp_x, amp_y, theta0, phi0)
    u0 = np.sin(np.deg2rad(theta0)) * np.cos(np.deg2rad(phi0))
    v0 = np.sin(np.deg2rad(theta0)) * np.sin(np.deg2rad(phi0))
    w0 = np.cos(np.deg2rad(theta0))
    w_t = _normalize_weights_torch(w_t, px, py, pz, u0, v0, w0)
    feat = _features(px, py, pz, w_t.real, w_t.imag, u0, v0, w0, scale)
    with torch.no_grad():
        delta = model(torch.as_tensor(feat[None]))[0].numpy()
    return w_t + (delta[:, 0] + 1j * delta[:, 1]) / scale


def eval_dir(model, px, py, pz, amp_x, amp_y, theta0, phi0, null_dirs):
    w_t = coordinate_taylor_3d(px, py, pz, amp_x, amp_y, theta0, phi0)
    sll_t, _, _, _ = eval_dense_3d(w_t, px, py, pz, theta0, phi0, null_dirs)
    w_ai = model_predict_v4(model, px, py, pz, amp_x, amp_y, theta0, phi0)
    sll_a, _, _, _ = eval_dense_3d(w_ai, px, py, pz, theta0, phi0, null_dirs)
    return float(sll_t), float(sll_a)


def stage3_eval():
    model = DeepSetsModel(input_dim=9, hidden_dim=HIDDEN, output_dim=2)
    model.load_state_dict(torch.load(MODEL_V4, map_location='cpu',
                                     weights_only=True))
    model.eval()

    out = {'model': 'deepsets_model_v4_256.pt (mixed-scale: 400x1024 + '
                    f'{N_CURVED_4096+N_PLANAR_4096}x4096, scale=N features)'}
    posx64 = uniform_linear_array_pos(NX)
    posy64 = uniform_linear_array_pos(NY)
    ax64, ay64 = taylor_2d_separable(NX, NY, 35)

    def build(alpha, seed):
        rng = np.random.RandomState(seed)
        px = np.tile(posx64[:, None], (1, NY))
        py = np.tile(posy64[None, :], (NX, 1))
        px = (px + rng.uniform(-0.02, 0.02, (NX, NY))).ravel()
        py = (py + rng.uniform(-0.02, 0.02, (NX, NY))).ravel()
        pz = (alpha * (px ** 2 + py ** 2)).ravel()
        return px, py, pz

    for tag, alpha, dirs in [
        ('planar_64', 0.0, [(0, 0), (30, 45), (60, 0), (60, 45)]),
        ('curved64_a0.03', 0.03, [(0, 0), (30, 0), (60, 0)]),
        ('curved64_a0.12', 0.12, [(0, 0), (30, 0), (60, 0)]),
    ]:
        px, py, pz = build(alpha, 4242)
        rows = []
        for theta0, phi0 in dirs:
            nd = _get_null_dirs(theta0, phi0)
            st, sa = eval_dir(model, px, py, pz, ax64, ay64, theta0, phi0, nd)
            rows.append({'dir': [theta0, phi0], 'taylor': st, 'ai': sa,
                         'diff': sa - st})
            print('  [%s] (th=%2d,ph=%3d): Taylor=%6.2f AI=%6.2f (%+.2f)'
                  % (tag, theta0, phi0, st, sa, sa - st), flush=True)
        out[tag] = rows

    # 回归测试: 1024 曲面 test 集(50) + 平面40方向
    v2 = np.load(os.path.join(OUTPUT_DIR, 'ascend_return_20260831',
                              'github_991916e7_assets', 'teacher_labels_v2.npz'))
    te_idx = np.where(v2['split'] == 2)[0]
    slls_t, slls_a = [], []
    for i in te_idx:
        px, py, pz = v2['px'][i], v2['py'][i], v2['pz'][i]
        w_re, w_im = v2['w_taylor_re'][i], v2['w_taylor_im'][i]
        u0, v0, w0 = v2['u0'][i], v2['v0'][i], v2['w0'][i]
        w_t = w_re + 1j * w_im
        th0, ph0 = float(v2['theta0'][i]), float(v2['phi0'][i])
        nd = _get_null_dirs(th0, ph0)
        s_t, _, _, _ = eval_dense_3d(w_t, px, py, pz, th0, ph0, nd)
        feat = _features(px, py, pz, w_re, w_im, u0, v0, w0, 1024.0)
        with torch.no_grad():
            delta = model(torch.as_tensor(feat[None]))[0].numpy()
        w_ai = w_t + (delta[:, 0] + 1j * delta[:, 1]) / 1024.0
        s_a, _, _, _ = eval_dense_3d(w_ai, px, py, pz, th0, ph0, nd)
        slls_t.append(s_t)
        slls_a.append(s_a)
    t_m, s_m = np.mean(slls_t), np.mean(slls_a)
    s_s = float(np.mean([v2['sll_socp'][i] for i in te_idx]))
    rec = (s_m - t_m) / (s_s - t_m) * 100 if abs(s_s - t_m) > 0.01 else 0
    out['regression_1024_curved_test'] = {
        'taylor_mean': float(t_m), 'ai_mean': float(s_m),
        'socp_mean': s_s, 'recovery_pct': float(rec),
        'v3_reference': {'ai_mean': -24.56, 'recovery_pct': 128.0},
    }
    print(f'  [regression 1024 curved] Taylor={t_m:.2f} SOCP={s_s:.2f} '
          f'AI={s_m:.2f} 恢复{rec:.0f}% (v3: -24.56/128%)', flush=True)

    # 1024 平面40方向回归
    posx32 = uniform_linear_array_pos(32)
    posy32 = uniform_linear_array_pos(32)
    ax32, ay32 = taylor_2d_separable(32, 32, 35)
    p32x = np.tile(posx32[:, None], (1, 32)).ravel()
    p32y = np.tile(posy32[None, :], (32, 1)).ravel()
    p32z = np.zeros(1024)
    worst = -100
    for t in [0, 15, 30, 45, 60]:
        for p in [0, 45, 90, 135, 180, 225, 270, 315]:
            st, sa = eval_dir(model, p32x, p32y, p32z, ax32, ay32,
                              float(t), float(p), _get_null_dirs(t, p))
            worst = max(worst, sa - st)
    out['regression_1024_planar40_worst_degradation_db'] = float(worst)
    print(f'  [regression 1024 planar40] worst degradation {worst:+.2f} dB '
          f'(v3: +0.51)', flush=True)

    path = os.path.join(OUTPUT_DIR, 'scale_fix_v4.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f'[stage3] saved: {path}')


def main():
    print('=' * 78, flush=True)
    print('v4: mixed-scale training (1024 + 4096)', flush=True)
    print('=' * 78, flush=True)
    stage1_generate_teacher_4096()
    stage2_train()
    stage3_eval()
    print('=' * 78, flush=True)


if __name__ == '__main__':
    main()
