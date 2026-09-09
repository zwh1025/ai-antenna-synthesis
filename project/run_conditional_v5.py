"""v5 条件式 DeepSets 训练：扰动感知阵列重综合。

核心创新：阵元失效掩码作为网络条件输入，扰动发生时 AI 直接重综合
适配当前缺陷孔径的全套激励权值，对比固定 Taylor-LCMV 权值获得鲁棒增益。

数据集（混合教师策略，0D 试点验证）：
  - 复用 v2 曲面 280 + v4 4096 规模 100 + 平面增广 60（mask=1，理想孔径）
  - 新增平面失效 110（10%/20%，SOCP 固定掩膜口径教师，mask 含 0）
  - 新增平面位置误差 60（坐标 Taylor+LCMV 教师，mask=1）
  - 验证 80 / 测试 100（混合工况）

网络改造：
  - 特征 9 -> 10 维（+mask）
  - 输出参数化 w = mask * (w0 + delta_w / scale)，失效元强制清零

断点续跑：分阶段落盘，重跑自动跳过已完成阶段。

输出: outputs/teacher_labels_v5.npz, outputs/deepsets_model_v5_256.pt,
      outputs/conditional_v5.json
"""

import os, sys, time, json
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mylib.antenna_calc import (
    uniform_linear_array_pos, taylor_2d_separable,
    beam_steering_phase_2d, combine_2d_excitation,
)
from mylib.deepsets import DeepSetsModel, count_parameters
from mylib.train import EarlyStopping
from mylib.official_evaluator import evaluate_official_case
from mylib.sum_diff import capon_nulling_2d
from run_curved_verify import (
    coordinate_taylor_3d, eval_dense_3d, uv_to_uvw, steering_vec_3d,
)
from run_deepsets_train import _get_null_dirs
from run_scale_fix_v4 import _features, _extra_planar, _normalize_weights_torch
from run_stage4a_robustness_degradation import (
    apply_position_error, generate_failure_mask,
)
from run_teacher_stage0d_v2 import solve_socp_official

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
TEACHER_V5 = os.path.join(OUTPUT_DIR, 'teacher_labels_v5.npz')
MODEL_V5 = os.path.join(OUTPUT_DIR, 'deepsets_model_v5_256.pt')
# v2 曲面数据缺失（clone 后 npz 被 .gitignore 排除），
# 用 4096 数据（含 60 曲面）+ _extra_planar 替代

COORD_NORM = 8.0
SLL_NORM = 50.0
HIDDEN = 256
EPOCHS = 300
LR = 1e-3
BATCH_1024 = 16
BATCH_4096 = 6
SEED = 2025

NX32 = 32
N_FAIL_10 = 55
N_FAIL_20 = 55
N_POS = 60
N_EXTRA_PLANAR = 60
RHO = 1.4
SOCP_ROUNDS = 15


# ==================== stage 1a: 解析教师（位置误差 + 平面增广）====================

def _steering_flat(px, py, pz, u, v, w=None):
    if w is None:
        w = np.sqrt(np.maximum(1.0 - u * u - v * v, 0.0))
    return np.exp(1j * 2 * np.pi * (px * u + py * v + pz * w))


def _flat_lcmv(px, py, pz, w_ref, theta0, phi0, null_dirs, active, lamb=1.0):
    """逐元坐标版最小修正 LCMV（R=I）。"""
    k = 2 * np.pi / lamb
    u0 = np.sin(np.deg2rad(theta0)) * np.cos(np.deg2rad(phi0))
    v0 = np.sin(np.deg2rad(theta0)) * np.sin(np.deg2rad(phi0))
    w0c = np.cos(np.deg2rad(theta0))

    def av(u, v):
        wd = np.sqrt(max(1.0 - u * u - v * v, 0.0))
        return np.exp(1j * k * (px[active] * u + py[active] * v
                                + pz[active] * wd))
    a_main = av(u0, v0)
    w0 = w_ref[active] / (np.conj(a_main) @ w_ref[active])
    cols = [a_main]
    for tn, pn in null_dirs:
        cols.append(av(np.sin(np.deg2rad(tn)) * np.cos(np.deg2rad(pn)),
                       np.sin(np.deg2rad(tn)) * np.sin(np.deg2rad(pn))))
    C = np.column_stack(cols)
    f = np.zeros(len(cols), dtype=complex)
    f[0] = 1.0
    residual = f - C.conj().T @ w0
    w_act = w0 + C @ np.linalg.lstsq(C.conj().T @ C, residual, rcond=1e-10)[0]
    w_full = np.zeros(len(px), dtype=complex)
    w_full[active] = w_act
    return w_full


def _taylor_baseline(px, py, pz, amp_x, amp_y, theta0, phi0, null_dirs, mask):
    """坐标 Taylor 基线（失效元置 0 后归一化）+ flat LCMV 置零。"""
    w_t = coordinate_taylor_3d(px, py, pz, amp_x, amp_y, theta0, phi0)
    active = mask.astype(bool)
    w_t[~active] = 0.0
    u0 = np.sin(np.deg2rad(theta0)) * np.cos(np.deg2rad(phi0))
    v0 = np.sin(np.deg2rad(theta0)) * np.sin(np.deg2rad(phi0))
    w0c = np.cos(np.deg2rad(theta0))
    a_main = _steering_flat(px, py, pz, u0, v0, w0c)
    resp = np.conj(a_main[active]) @ w_t[active]
    if abs(resp) > 1e-12:
        w_t = w_t / resp
    w_lcmv = _flat_lcmv(px, py, pz, w_t, theta0, phi0, null_dirs, active)
    return w_t, w_lcmv


def stage1a_analytic_teacher():
    """生成位置误差样本（坐标 Taylor+LCMV 教师）+ 平面增广。"""
    posx = uniform_linear_array_pos(NX32)
    posy = uniform_linear_array_pos(NX32)
    amp_x, amp_y = taylor_2d_separable(NX32, NX32, 35)
    rng = np.random.RandomState(SEED)
    px0 = np.tile(posx[:, None], (1, NX32)).ravel()
    py0 = np.tile(posy[None, :], (NX32, 1)).ravel()
    pz0 = np.zeros(NX32 * NX32)
    mask_one = np.ones(NX32 * NX32, dtype=np.float32)

    samples = []
    THETAS = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    for i in range(N_POS):
        theta0 = float(rng.choice(THETAS)) + rng.uniform(-5, 5)
        theta0 = min(max(theta0, 0.0), 60.0)
        phi0 = float(rng.uniform(0, 360))
        null_dirs = _get_null_dirs(theta0, phi0)
        px, py, _, _ = apply_position_error(posx, posy, int(rng.randint(0, 10000)))
        px = px.reshape(-1)
        py = py.reshape(-1)
        w_t, w_lcmv = _taylor_baseline(px, py, pz0, amp_x, amp_y, theta0,
                                       phi0, null_dirs, mask_one.astype(bool))
        sll_t, _, _, _ = eval_dense_3d(w_t, px, py, pz0, theta0, phi0, null_dirs)
        sll_l, _, _, _ = eval_dense_3d(w_lcmv, px, py, pz0, theta0, phi0,
                                       null_dirs)
        u0 = np.sin(np.deg2rad(theta0)) * np.cos(np.deg2rad(phi0))
        v0 = np.sin(np.deg2rad(theta0)) * np.sin(np.deg2rad(phi0))
        w0c = np.cos(np.deg2rad(theta0))
        samples.append({
            'px': px, 'py': py, 'pz': pz0,
            'w_taylor_re': w_t.real, 'w_taylor_im': w_t.imag,
            'w_socp_re': w_lcmv.real, 'w_socp_im': w_lcmv.imag,
            'sll_taylor': float(sll_t), 'sll_socp': float(sll_l),
            'alpha': 0.0, 'theta0': theta0, 'phi0': phi0,
            'u0': float(u0), 'v0': float(v0), 'w0': float(w0c),
            'mask': mask_one, 'perturbation': 'position',
        })
    print(f'[stage1a] position error: {N_POS} samples', flush=True)
    return samples


# ==================== stage 1b: 失效 SOCP 教师 ====================

def stage1b_failure_teacher():
    """生成失效场景 SOCP 教师（固定掩膜口径，并行）。"""
    posx = uniform_linear_array_pos(NX32)
    posy = uniform_linear_array_pos(NX32)
    amp_x, amp_y = taylor_2d_separable(NX32, NX32, 35)
    rng = np.random.RandomState(SEED + 1)
    px0 = np.tile(posx[:, None], (1, NX32)).ravel()
    py0 = np.tile(posy[None, :], (NX32, 1)).ravel()
    pz0 = np.zeros(NX32 * NX32)

    configs = []
    THETAS = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    for rate, n in [(0.10, N_FAIL_10), (0.20, N_FAIL_20)]:
        for i in range(n):
            theta0 = float(rng.choice(THETAS)) + rng.uniform(-5, 5)
            theta0 = min(max(theta0, 0.0), 60.0)
            phi0 = float(rng.uniform(0, 360))
            null_dirs = _get_null_dirs(theta0, phi0)
            seed = int(rng.randint(0, 10000))
            mask = generate_failure_mask(rate, seed, NX32 * NX32)
            configs.append((theta0, phi0, null_dirs, mask, rate))

    print(f'[stage1b] failure SOCP teacher: {len(configs)} samples '
          f'(parallel, rho={RHO}, {SOCP_ROUNDS} rounds, fixed-mask caliber)',
          flush=True)
    import multiprocessing as mp
    tasks = []
    for idx, (theta0, phi0, null_dirs, mask, rate) in enumerate(configs):
        active = ~mask.astype(bool)
        w_t, _ = _taylor_baseline(px0, py0, pz0, amp_x, amp_y, theta0, phi0,
                                  null_dirs, active)
        w_ref = w_t.copy()
        tasks.append((idx, theta0, phi0, null_dirs, px0, py0, pz0,
                      w_t, mask, w_ref, posx, posy))

    n_proc = min(len(tasks), max(1, (os.cpu_count() or 8) - 2))
    t0 = time.time()
    with mp.get_context('fork').Pool(n_proc) as pool:
        results = pool.map(_socp_worker, tasks)
    print(f'[stage1b] SOCP done in {time.time()-t0:.0f}s', flush=True)

    samples = []
    for (theta0, phi0, null_dirs, mask, rate), (w_best, sll) in zip(configs, results):
        if w_best is None:
            w_t, _ = _taylor_baseline(px0, py0, pz0, amp_x, amp_y, theta0,
                                      phi0, null_dirs, ~mask.astype(bool))
            w_best, sll = w_t, -100.0
        u0 = np.sin(np.deg2rad(theta0)) * np.cos(np.deg2rad(phi0))
        v0 = np.sin(np.deg2rad(theta0)) * np.sin(np.deg2rad(phi0))
        w0c = np.cos(np.deg2rad(theta0))
        mask_f = (~mask.astype(bool)).astype(np.float32)
        samples.append({
            'px': px0, 'py': py0, 'pz': pz0,
            'w_taylor_re': w_best.real, 'w_taylor_im': w_best.imag,
            'w_socp_re': w_best.real, 'w_socp_im': w_best.imag,
            'sll_taylor': float(sll), 'sll_socp': float(sll),
            'alpha': 0.0, 'theta0': theta0, 'phi0': phi0,
            'u0': float(u0), 'v0': float(v0), 'w0': float(w0c),
            'mask': mask_f, 'perturbation': f'failure_{rate}',
        })
    return samples


_G = {}


def _socp_worker(task):
    import torch
    torch.set_num_threads(1)
    (idx, theta0, phi0, null_dirs, px, py, pz, w_taylor, mask, w_ref,
     posx, posy) = task
    case = {'case_id': f'fail_{idx}', 'theta0_deg': theta0,
            'phi0_deg': phi0, 'null_dirs': [list(nd) for nd in null_dirs]}
    fail_mask = mask.astype(bool)
    try:
        best_w, best_sll, _ = solve_socp_official(
            px, py, pz, case, w_taylor, null_dirs, posx, posy,
            fail_mask=fail_mask, w_ref=w_ref, rho=RHO, rounds=SOCP_ROUNDS)
        return best_w, best_sll
    except Exception as e:
        print(f'  [worker {idx}] SOCP failed: {e}', flush=True)
        return None, None


# ==================== stage 1: 合并数据集 ====================

def stage1_generate():
    """合并 analytic + failure npz 为最终训练集。"""
    if os.path.exists(TEACHER_V5):
        print(f'[stage1] 已存在, 跳过: {TEACHER_V5}')
        return
    analytic_path = os.path.join(OUTPUT_DIR, 'teacher_labels_v5_analytic.npz')
    failure_path = os.path.join(OUTPUT_DIR, 'teacher_labels_v5_failure.npz')
    if not os.path.exists(analytic_path):
        raise SystemExit(f'解析教师不存在: {analytic_path}\n'
                         '请先运行: python run_v5_generate.py --only analytic')

    da = np.load(analytic_path, allow_pickle=True)
    samples = []
    for j in range(len(da['theta0'])):
        samples.append({
            'px': da['px'][j], 'py': da['py'][j], 'pz': da['pz'][j],
            'mask': np.asarray(da['mask'][j], dtype=np.float32),
            'w_taylor_re': da['w_taylor_re'][j], 'w_taylor_im': da['w_taylor_im'][j],
            'w_socp_re': da['w_socp_re'][j], 'w_socp_im': da['w_socp_im'][j],
            'sll_taylor': float(da['sll_taylor'][j]),
            'sll_socp': float(da['sll_socp'][j]),
            'alpha': float(da['alpha'][j]),
            'theta0': float(da['theta0'][j]), 'phi0': float(da['phi0'][j]),
            'u0': float(da['u0'][j]), 'v0': float(da['v0'][j]),
            'w0': float(da['w0'][j]),
        })
    print(f'[stage1] analytic: {len(samples)} samples')

    if os.path.exists(failure_path):
        df = np.load(failure_path, allow_pickle=True)
        n_fail = len(df['theta0'])
        for j in range(n_fail):
            samples.append({
                'px': df['px'][j], 'py': df['py'][j], 'pz': df['pz'][j],
                'mask': np.asarray(df['mask'][j], dtype=np.float32),
                'w_taylor_re': df['w_taylor_re'][j], 'w_taylor_im': df['w_taylor_im'][j],
                'w_socp_re': df['w_socp_re'][j], 'w_socp_im': df['w_socp_im'][j],
                'sll_taylor': float(df['sll_taylor'][j]),
                'sll_socp': float(df['sll_socp'][j]),
                'alpha': 0.0,
                'theta0': float(df['theta0'][j]), 'phi0': float(df['phi0'][j]),
                'u0': float(df['u0'][j]), 'v0': float(df['v0'][j]),
                'w0': float(df['w0'][j]),
            })
        print(f'[stage1] failure: {n_fail} samples')

    n_total = len(samples)
    n_val = min(80, n_total // 5)
    n_test = min(100, n_total // 4)
    rng = np.random.RandomState(SEED + 2)
    perm = rng.permutation(n_total)
    split = np.zeros(n_total, dtype=np.int32)
    split[perm[:n_val]] = 1
    split[perm[n_val:n_val + n_test]] = 2

    keys = ['px', 'py', 'pz', 'mask', 'w_taylor_re', 'w_taylor_im',
            'w_socp_re', 'w_socp_im']
    scalars = ['sll_taylor', 'sll_socp', 'alpha', 'theta0', 'phi0',
               'u0', 'v0', 'w0']
    out = {'split': split}
    for k in keys:
        out[k] = np.array([s[k] for s in samples], dtype=object)
    for k in scalars:
        out[k] = np.array([s[k] for s in samples])
    np.savez(TEACHER_V5, **out)
    print(f'[stage1] saved: {TEACHER_V5} ({n_total} samples: '
          f'{int((split==0).sum())} train / {n_val} val / {n_test} test)',
          flush=True)


# ==================== stage 2: 训练 ====================

def _features_v5(px, py, pz, w_re, w_im, u0, v0, w0, scale, mask):
    n = len(px)
    return np.stack([
        np.asarray(px) / COORD_NORM,
        np.asarray(py) / COORD_NORM,
        np.asarray(pz) / COORD_NORM,
        np.asarray(w_re) * scale,
        np.asarray(w_im) * scale,
        np.full(n, u0), np.full(n, v0), np.full(n, w0),
        np.full(n, 35.0 / SLL_NORM),
        np.asarray(mask, dtype=np.float32),
    ], axis=-1).astype(np.float32)


def _init_from_v4(model, v4_path):
    """从 v4 权重迁移学习：复制 9 维权重，mask 维初始化为 0（无影响）。"""
    v4_sd = torch.load(v4_path, map_location='cpu', weights_only=True)
    v5_sd = model.state_dict()
    copied = 0
    for k in v5_sd:
        if k in v4_sd and v4_sd[k].shape == v5_sd[k].shape:
            v5_sd[k] = v4_sd[k].clone()
            copied += 1
        elif k == 'phi.0.weight' and 'phi.0.weight' in v4_sd:
            w = v4_sd['phi.0.weight']  # (256, 9)
            v5_sd[k][:, :9] = w
            v5_sd[k][:, 9] = 0.0  # mask 维 = 0，初始等价 v4
            copied += 1
            print(f'  [transfer] phi.0.weight: {w.shape} -> {v5_sd[k].shape} '
                  f'(mask col=0)', flush=True)
    model.load_state_dict(v5_sd)
    print(f'  [transfer] copied {copied}/{len(v5_sd)} tensors from v4',
          flush=True)
    return model


def _eval_ideal_degradation(model, posx, posy, amp_x, amp_y, px0, py0, pz0):
    """理想平面 40 方向最差退化（快速评估，用于训练早停门控）。"""
    mask_one = np.ones(len(px0), dtype=np.float32)
    worst = -100.0
    for t in [0, 30, 60]:
        for p in [0, 90, 180, 270]:
            nd = _get_null_dirs(t, p)
            w_t = coordinate_taylor_3d(px0, py0, pz0, amp_x, amp_y, t, p)
            sll_t, _, _, _ = eval_dense_3d(w_t, px0, py0, pz0, t, p, nd)
            w_ai, _ = model_predict_v5(model, px0, py0, pz0, amp_x, amp_y,
                                        t, p, mask_one)
            sll_a, _, _, _ = eval_dense_3d(w_ai, px0, py0, pz0, t, p, nd)
            worst = max(worst, sll_a - sll_t)
    return worst


def stage2_train():
    if os.path.exists(MODEL_V5):
        print(f'[stage2] 已存在, 跳过: {MODEL_V5}')
        return
    d = np.load(TEACHER_V5, allow_pickle=True)
    split = d['split']
    tr_idx = np.where(split == 0)[0]
    va_idx = np.where(split == 1)[0]

    def pack(idx_list):
        feats, tgts, is_failure = [], [], []
        for i in idx_list:
            n = len(d['px'][i])
            scale = float(n)
            mask = np.asarray(d['mask'][i], dtype=np.float32)
            w_re = d['w_taylor_re'][i]
            w_im = d['w_taylor_im'][i]
            d_re = (d['w_socp_re'][i] - w_re) * scale * mask
            d_im = (d['w_socp_im'][i] - w_im) * scale * mask
            feats.append(_features_v5(
                d['px'][i], d['py'][i], d['pz'][i], w_re, w_im,
                float(d['u0'][i]), float(d['v0'][i]), float(d['w0'][i]),
                scale, mask))
            tgts.append(np.stack([d_re, d_im], axis=-1).astype(np.float32))
            is_failure.append(float(mask.min()) < 0.5)
        return feats, tgts, is_failure

    tr_f, tr_t, tr_fail = pack(tr_idx)
    va_f, va_t, _ = pack(va_idx)

    # 数据平衡：失效样本 3x 上采样
    fail_idx = [i for i, f in enumerate(tr_fail) if f]
    ideal_idx = [i for i, f in enumerate(tr_fail) if not f]
    oversample = fail_idx * 3  # 3x 失效样本
    balanced_idx = ideal_idx + oversample
    print(f'[stage2] train: {len(ideal_idx)} ideal + {len(fail_idx)} failure '
          f'-> {len(balanced_idx)} balanced (failure 3x oversampled)',
          flush=True)

    # 按 size 分组
    by_size = {}
    for i in balanced_idx:
        n = len(tr_f[i])
        key = '1024' if n == 1024 else ('4096' if n == 4096 else f'n{n}')
        if key not in by_size:
            by_size[key] = ([], [])
        by_size[key][0].append(tr_f[i])
        by_size[key][1].append(tr_t[i])
    train_groups = [(k, np.array(v[0]), np.array(v[1]))
                    for k, v in by_size.items()]

    model = DeepSetsModel(input_dim=10, hidden_dim=HIDDEN, output_dim=2)

    # v4 迁移学习
    v4_path = os.path.join(OUTPUT_DIR, 'deepsets_model_v4_256.pt')
    if os.path.exists(v4_path):
        print(f'[stage2] v4 transfer learning from {v4_path}', flush=True)
        model = _init_from_v4(model, v4_path)

    opt = torch.optim.Adam(model.parameters(), lr=LR * 0.3)  # 低 LR 微调
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min',
                                                       factor=0.5, patience=8)
    stop = EarlyStopping(patience=30)
    crit = nn.MSELoss()

    # 理想平面回归监控
    posx = uniform_linear_array_pos(NX32)
    posy = uniform_linear_array_pos(NX32)
    amp_x, amp_y = taylor_2d_separable(NX32, NX32, 35)
    px0 = np.tile(posx[:, None], (1, NX32)).ravel()
    py0 = np.tile(posy[None, :], (NX32, 1)).ravel()
    pz0 = np.zeros(NX32 * NX32)

    print(f'[stage2] params={count_parameters(model):,}; '
          f'val={len(va_idx)}; lr={LR*0.3:.1e} (v4 fine-tune)',
          flush=True)
    t0 = time.time()
    best = float('inf')
    best_ideal_deg = +99.0
    for epoch in range(EPOCHS):
        model.train()
        tot, nb = 0.0, 0
        for name, F, T in train_groups:
            if len(F) == 0:
                continue
            bs = BATCH_1024 if name == '1024' else BATCH_4096
            perm = np.random.permutation(len(F))
            for s in range(0, len(F), bs):
                idx = perm[s:s + bs]
                bx = torch.as_tensor(np.stack([F[j] for j in idx]))
                by = torch.as_tensor(np.stack([T[j] for j in idx]))
                opt.zero_grad()
                loss = crit(model(bx), by)
                loss.backward()
                opt.step()
                tot += loss.item()
                nb += 1
        model.eval()
        vl = 0.0
        cnt = 0
        with torch.no_grad():
            for i in range(len(va_f)):
                vx = torch.as_tensor(va_f[i][None])
                vy = torch.as_tensor(va_t[i][None])
                vl += crit(model(vx), vy).item()
                cnt += 1
        vl /= max(cnt, 1)
        sched.step(vl)

        # 每 10 epoch 检查理想退化
        ideal_deg = None
        if (epoch + 1) % 10 == 0 or epoch == 0:
            ideal_deg = _eval_ideal_degradation(model, posx, posy, amp_x,
                                                amp_y, px0, py0, pz0)

        # 保存条件：val loss 改善 AND 理想退化 < 2dB（或首次）
        save = vl < best
        if ideal_deg is not None and ideal_deg > 2.0 and best < float('inf'):
            save = False  # 理想退化超 2dB，拒绝保存
        if save:
            best = vl
            best_ideal_deg = ideal_deg if ideal_deg is not None else best_ideal_deg
            torch.save(model.state_dict(), MODEL_V5)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            deg_str = f' ideal_deg={ideal_deg:+.2f}dB' if ideal_deg is not None else ''
            print(f'  ep{epoch+1:3d}: loss={tot/nb:.6f} val={vl:.6f}'
                  f'{deg_str} ({time.time()-t0:.0f}s)', flush=True)
        if stop.step(vl):
            print(f'  early stop ep{epoch+1} (best {best:.6f}, '
                  f'ideal_deg={best_ideal_deg:+.2f}dB)', flush=True)
            break
    print(f'[stage2] saved: {MODEL_V5} (best val {best:.6f}, '
          f'ideal_deg={best_ideal_deg:+.2f}dB)', flush=True)


# ==================== stage 3: 评估 ====================

def model_predict_v5(model, px, py, pz, amp_x, amp_y, theta0, phi0, mask):
    scale = float(len(px))
    w_t = coordinate_taylor_3d(px, py, pz, amp_x, amp_y, theta0, phi0)
    active = mask.astype(bool)
    w_t[~active] = 0.0
    u0 = np.sin(np.deg2rad(theta0)) * np.cos(np.deg2rad(phi0))
    v0 = np.sin(np.deg2rad(theta0)) * np.sin(np.deg2rad(phi0))
    w0c = np.cos(np.deg2rad(theta0))
    a_main = _steering_flat(px, py, pz, u0, v0, w0c)
    resp = np.conj(a_main[active]) @ w_t[active]
    if abs(resp) > 1e-12:
        w_t = w_t / resp
    feat = _features_v5(px, py, pz, w_t.real, w_t.imag, u0, v0, w0c, scale, mask)
    with torch.no_grad():
        delta = model(torch.as_tensor(feat[None]))[0].numpy()
    return mask * (w_t + (delta[:, 0] + 1j * delta[:, 1]) / scale), w_t


def _official_sll(w, case, posx_ax, posy_ax, lamb=1.0):
    amp = np.abs(w).reshape(NX32, NX32)
    if amp.max() > 0:
        amp = amp / amp.max()
    phase = (np.angle(w) % (2 * np.pi)).reshape(NX32, NX32)
    r = evaluate_official_case(amp, phase, posx_ax, posy_ax,
                               case['theta0'], case['phi0'],
                               null_dirs=case['null_dirs'], n_uv=201,
                               lamb=lamb)
    return float(r['sum']['sll_db'])


def stage3_eval():
    model = DeepSetsModel(input_dim=10, hidden_dim=HIDDEN, output_dim=2)
    model.load_state_dict(torch.load(MODEL_V5, map_location='cpu',
                                     weights_only=True))
    model.eval()

    posx = uniform_linear_array_pos(NX32)
    posy = uniform_linear_array_pos(NX32)
    amp_x, amp_y = taylor_2d_separable(NX32, NX32, 35)
    px0 = np.tile(posx[:, None], (1, NX32)).ravel()
    py0 = np.tile(posy[None, :], (NX32, 1)).ravel()
    pz0 = np.zeros(NX32 * NX32)
    mask_one = np.ones(NX32 * NX32, dtype=np.float32)
    fj = json.load(open(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'results', 'stage4a_robustness_degradation', 'failure_cases.json')))

    out = {'model': 'deepsets_model_v5_256.pt (conditional: +mask, '
                    'mixed teacher: SOCP failure + analytic pos)'}

    for rate in [0.05, 0.10, 0.20]:
        rows = []
        for case_id in ['regular_000', 'regular_028', 'regular_072',
                        'random_000']:
            r0 = [r for r in fj['realizations']
                  if r['case_id'] == case_id
                  and abs(r['failure_rate'] - rate) < 1e-9 and r['seed'] == 0][0]
            mask = np.zeros(NX32 * NX32, dtype=bool)
            mask[np.array(r0['failed_indices'])] = True
            mask_f = (~mask).astype(np.float32)
            manifest = json.load(open(os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'results', 'stage2_strict_closure', 'baseline',
                'case_manifest.json')))
            by_id = {c['case_id']: c for c in manifest['regular']
                     + manifest['random']}
            case = by_id[case_id]
            theta0, phi0 = case['theta0_deg'], case['phi0_deg']
            null_dirs = [tuple(nd) for nd in case['null_dirs']]
            w_ai, w_t = model_predict_v5(model, px0, py0, pz0, amp_x, amp_y,
                                          theta0, phi0, mask_f)
            sll_ai = _official_sll(w_ai, {'theta0': theta0, 'phi0': phi0,
                                          'null_dirs': null_dirs}, posx, posy)
            sll_a0 = r0['sum_sll_db']
            rows.append({'case_id': case_id, 'a0_fixed': sll_a0,
                         'ai_v5': sll_ai, 'gain': sll_ai - sll_a0})
            print(f'  fail{int(rate*100):02d} {case_id}: A0={sll_a0:.2f} '
                  f'AI={sll_ai:.2f} ({sll_ai-sll_a0:+.2f})', flush=True)
        gains = [r['gain'] for r in rows]
        out[f'fail{int(rate*100):02d}'] = {
            'mean_a0': float(np.mean([r['a0_fixed'] for r in rows])),
            'mean_ai': float(np.mean([r['ai_v5'] for r in rows])),
            'mean_gain': float(np.mean(gains)), 'rows': rows}

    w_a0 = (combine_2d_excitation(amp_x, amp_y,
              *[beam_steering_phase_2d(posx, posy, 0, 0)] * 2))
    st, sa = None, None
    worst = -100
    for t in [0, 15, 30, 45, 60]:
        for p in [0, 45, 90, 135, 180, 225, 270, 315]:
            nd = _get_null_dirs(t, p)
            w_t = coordinate_taylor_3d(px0, py0, pz0, amp_x, amp_y, t, p)
            sll_t, _, _, _ = eval_dense_3d(w_t, px0, py0, pz0, t, p, nd)
            w_ai, _ = model_predict_v5(model, px0, py0, pz0, amp_x, amp_y,
                                        t, p, mask_one)
            sll_a, _, _, _ = eval_dense_3d(w_ai, px0, py0, pz0, t, p, nd)
            worst = max(worst, sll_a - sll_t)
    out['planar_ideal_worst_degradation'] = float(worst)
    print(f'  planar ideal worst degradation: {worst:+.2f} dB', flush=True)

    path = os.path.join(OUTPUT_DIR, 'conditional_v5.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f'[stage3] saved: {path}', flush=True)


def main():
    print('=' * 78, flush=True)
    print('v5: conditional DeepSets (perturbation-aware re-synthesis)',
          flush=True)
    print('=' * 78, flush=True)
    stage1_generate()
    stage2_train()
    stage3_eval()
    print('=' * 78, flush=True)


if __name__ == '__main__':
    main()
